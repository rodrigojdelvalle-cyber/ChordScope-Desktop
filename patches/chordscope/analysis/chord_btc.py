from __future__ import annotations

import gc
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np

from .audio import load_mono
from .control import AnalysisCancelled, check_cancel, is_cancelled
from .librosa_compat import patch_third_party_librosa_loader
from .resources import models_root
from .types import ChordSegment

Progress = Callable[[int, str], None]

_MODEL_CFG = dict(
    feature_size=144,
    timestep=108,
    input_dropout=0.2,
    layer_dropout=0.2,
    attention_dropout=0.2,
    relu_dropout=0.2,
    num_layers=8,
    num_heads=4,
    hidden_size=128,
    total_key_depth=128,
    total_value_depth=128,
    filter_size=128,
    loss="ce",
    probs_out=False,
)
_FEATURE_CFG = dict(song_hz=22050, inst_len=10.0, n_bins=144, bins_per_octave=24, hop_length=2048)
_TIMESTEP = 108


def _write_json_atomic(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _release_torch_cache() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class _LocalBTCModel:
    """BTC local loader without Transformers AutoModel.

    v2.0.1 could stall at AutoModel.from_pretrained inside Nuitka onefile.
    The offline BTC snapshot already vendors inference code and checkpoints, so
    v2.0.2 loads btc_src directly and keeps the runtime dependency surface small.
    """

    def __init__(self, repo_root: Path, device: str, large_voca: bool = True, stage=None):
        import torch
        self.repo_root = repo_root.resolve()
        self.large_voca = bool(large_voca)
        if stage:
            stage(8, "verificando modelo offline")
        required = [
            self.repo_root / "btc_src" / "btc_model.py",
            self.repo_root / "btc_src" / "features.py",
            self.repo_root / ("btc_model_large_voca.pt" if self.large_voca else "btc_model.pt"),
        ]
        missing = [str(p.name) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError("BTC offline incompleto: " + ", ".join(missing))
        patch_third_party_librosa_loader()
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))
        if stage:
            stage(18, "importando BTC local")
        BTC_model = importlib.import_module("btc_src.btc_model").BTC_model
        self.features = importlib.import_module("btc_src.features")
        self.device = "cuda:0" if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        num_chords = 170 if self.large_voca else 25
        if stage:
            stage(32, f"creando red · {self.device.upper()}")
        self.model = BTC_model(config=dict(_MODEL_CFG, num_chords=num_chords)).to(self.device)
        checkpoint = self.repo_root / ("btc_model_large_voca.pt" if self.large_voca else "btc_model.pt")
        if stage:
            stage(48, "cargando pesos")
        ckpt = torch.load(str(checkpoint), map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.mean = ckpt["mean"]
        self.std = ckpt["std"]
        self.idx_to_chord = self.features.idx2voca_chord() if self.large_voca else self.features.idx2chord
        if stage:
            stage(60, "modelo listo")

    def predict(self, audio: np.ndarray, stage=None) -> list[dict]:
        import torch
        if stage:
            stage(64, "calculando log-CQT")
        feat = self.features.audio_to_features(
            audio,
            sr_target=_FEATURE_CFG["song_hz"],
            inst_len=_FEATURE_CFG["inst_len"],
            n_bins=_FEATURE_CFG["n_bins"],
            bins_per_octave=_FEATURE_CFG["bins_per_octave"],
            hop_length=_FEATURE_CFG["hop_length"],
        )
        feat = feat.T
        feat = (feat - self.mean) / self.std
        n = _TIMESTEP
        num_pad = n - (feat.shape[0] % n)
        feat = np.pad(feat, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
        num_instance = feat.shape[0] // n
        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)
        lines: list[dict] = []
        start_time = 0.0
        prev = None
        time_unit = _FEATURE_CFG["inst_len"] / _TIMESTEP
        if stage:
            stage(72, f"inferencia · {num_instance} bloques")
        with torch.no_grad():
            for t in range(num_instance):
                attn_out, _ = self.model.self_attn_layers(x[:, n*t:n*(t+1), :])
                pred, _ = self.model.output_layer(attn_out)
                pred = pred.squeeze()
                for i in range(n):
                    if t == 0 and i == 0:
                        prev = pred[i].item()
                        continue
                    cur = pred[i].item()
                    if cur != prev:
                        end = time_unit * (n*t + i)
                        lines.append({"start": round(start_time,3), "end": round(end,3), "chord": self.idx_to_chord[prev]})
                        start_time = end
                        prev = cur
                    if t == num_instance - 1 and i + num_pad == n:
                        end = time_unit * (n*t + i)
                        if start_time != end:
                            lines.append({"start": round(start_time,3), "end": round(end,3), "chord": self.idx_to_chord[prev]})
                        break
                if stage:
                    stage(72 + int(25*(t+1)/max(1,num_instance)), f"inferencia · bloque {t+1}/{num_instance}")
        if stage:
            stage(100, f"{len(lines)} segmentos")
        return lines


class BTCChordEngine:
    name = "BTC Transformer"

    def __init__(self, device="cpu"):
        self.device = device
        self._model: _LocalBTCModel | None = None

    @staticmethod
    def _convert(results) -> list[ChordSegment]:
        return [ChordSegment(
            start=float(r.get("start", r.get("start_time", 0))),
            end=float(r.get("end", r.get("end_time", 0))),
            label=str(r.get("chord", "N")),
            engine="BTC Transformer",
            confidence=float(r.get("confidence", 1.0) or 1.0),
        ) for r in (results or [])]

    def _direct_analyze(self, audio_path, progress=None) -> list[ChordSegment]:
        local = models_root() / "btc-chord"
        if progress:
            progress(5, f"{self.name} · loader local")
        if self._model is None:
            self._model = _LocalBTCModel(local, self.device, large_voca=True, stage=progress)
        y, _ = load_mono(audio_path, sr=22050)
        if progress:
            progress(62, f"{self.name} · audio listo")
        return self._convert(self._model.predict(y, stage=progress))

    def analyze(self, audio_path, progress=None, *, timeout_seconds: int = 90, cancel_event=None, isolated: bool = True):
        check_cancel(cancel_event)
        if not isolated or timeout_seconds <= 0:
            return self._direct_analyze(audio_path, progress)
        return self._analyze_isolated(audio_path, progress, int(timeout_seconds), cancel_event)

    def _analyze_isolated(self, audio_path, progress, timeout_seconds: int, cancel_event=None):
        with tempfile.TemporaryDirectory(prefix="chordscope_btc_worker_") as td:
            td = Path(td)
            output_path = td / "result.json"
            progress_path = td / "progress.json"
            if "__compiled__" in globals():
                cmd = [sys.executable, "--btc-worker", str(audio_path), str(output_path), str(progress_path), str(self.device)]
            else:
                main_py = Path(__file__).resolve().parents[2] / "main.py"
                cmd = [sys.executable, str(main_py), "--btc-worker", str(audio_path), str(output_path), str(progress_path), str(self.device)]
            kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
            if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            proc = subprocess.Popen(cmd, **kwargs)
            started = time.monotonic()
            last_stage = "iniciando proceso aislado"
            last_pct = 5
            try:
                while proc.poll() is None:
                    if is_cancelled(cancel_event):
                        proc.terminate()
                        try:
                            proc.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        raise AnalysisCancelled("Análisis cancelado por el usuario")
                    elapsed = time.monotonic() - started
                    if elapsed > timeout_seconds:
                        proc.terminate()
                        try:
                            proc.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        raise TimeoutError(f"BTC excedió {timeout_seconds}s y fue omitido de forma segura")
                    if progress_path.exists():
                        try:
                            d = json.loads(progress_path.read_text(encoding="utf-8"))
                            last_stage = str(d.get("stage") or last_stage)
                            last_pct = int(d.get("percent") or last_pct)
                        except Exception:
                            pass
                    if progress:
                        progress(min(94,max(5,last_pct)), f"{self.name} · {last_stage} · {elapsed:.0f}s/{timeout_seconds}s")
                    time.sleep(0.45)
                elapsed = time.monotonic() - started
                if not output_path.exists():
                    raise RuntimeError(f"BTC worker terminó sin resultado (exit={proc.returncode})")
                payload = json.loads(output_path.read_text(encoding="utf-8"))
                if not payload.get("ok"):
                    raise RuntimeError(payload.get("error") or "BTC worker falló")
                if progress:
                    progress(100, f"{self.name} · listo · {elapsed:.1f}s")
                return self._convert(payload.get("segments") or [])
            finally:
                if proc.poll() is None:
                    proc.kill()
                    try:
                        proc.wait(timeout=2)
                    except Exception:
                        pass


def run_btc_worker(audio_path: str, output_path: str, progress_path: str, device: str = "cpu") -> int:
    out = Path(output_path)
    prog = Path(progress_path)
    def stage(percent: int, message: str):
        _write_json_atomic(prog, {"percent": int(percent), "stage": str(message), "time": time.time()})
    try:
        stage(2, "arrancando")
        engine = BTCChordEngine(device=device)
        segments = engine._direct_analyze(audio_path, stage)
        _write_json_atomic(out, {"ok": True, "segments": [
            {"start": s.start, "end": s.end, "chord": s.label, "confidence": s.confidence} for s in segments
        ]})
        stage(100, "listo")
        return 0
    except Exception as exc:
        _write_json_atomic(out, {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        stage(100, "error")
        return 2
    finally:
        _release_torch_cache()
