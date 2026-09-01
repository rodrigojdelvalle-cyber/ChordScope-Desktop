from __future__ import annotations

import json
import os
import tempfile
import time
import traceback
import wave
from pathlib import Path

import numpy as np

from .analysis.resources import build_variant, models_root


def _fixture(path: Path, sr: int = 22050, duration: float = 4.0) -> None:
    n = int(sr * duration)
    t = np.arange(n, dtype=np.float64) / sr
    y = np.zeros(n, dtype=np.float64)
    tones_a = (261.6256, 329.6276, 391.9954)
    tones_b = (220.0, 261.6256, 329.6276)
    mid = n // 2
    for freq in tones_a:
        y[:mid] += 0.055 * np.sin(2 * np.pi * freq * t[:mid])
    for freq in tones_b:
        y[mid:] += 0.055 * np.sin(2 * np.pi * freq * t[mid:])
    for bt in np.arange(0, duration, 0.5):
        i = int(bt * sr)
        m = min(n - i, int(0.02 * sr))
        if m > 0:
            y[i:i + m] += 0.28 * np.hanning(m)
    pcm = (np.clip(y, -0.95, 0.95) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _marker(payload: dict) -> None:
    target = os.environ.get("CHORDSCOPE_FULL_SMOKE_MARKER")
    if not target:
        return
    path = Path(target)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _timed(name: str, fn, timings: dict):
    start = time.perf_counter()
    value = fn()
    timings[name] = round(time.perf_counter() - start, 3)
    return value


def run_full_runtime_smoke() -> int:
    timings: dict[str, float] = {}
    state = {
        "status": "CHORDSCOPE_FULL_ML_RUNTIME_SMOKE_RUNNING",
        "version": "2.0.2",
        "build_variant": build_variant(),
        "stage": "initializing",
        "demucs_model": "pending",
        "beat_this_model": "pending",
        "lv_chordia_model": "pending",
        "btc_model": "pending",
        "btc_isolation": "pending",
        "timings": timings,
    }

    def mark(stage: str, **updates) -> None:
        state["stage"] = stage
        state.update(updates)
        state["timings"] = dict(timings)
        _marker(state)

    mark("verifying_offline_models")
    try:
        root = models_root()
        required = {
            "btc_config": root / "btc-chord" / "config.json",
            "btc_weights": root / "btc-chord" / "btc_model_large_voca.pt",
            "beat_this": root / "beat_this" / "beat_this-final0.ckpt",
            "demucs_yaml": root / "demucs" / "htdemucs_ft.yaml",
        }
        missing = [name for name, path in required.items() if not path.is_file() or path.stat().st_size == 0]
        if missing:
            raise RuntimeError("Modelos offline faltantes: " + ", ".join(missing))

        with tempfile.TemporaryDirectory(prefix="chordscope_full_smoke_") as td:
            audio = Path(td) / "fixture.wav"
            _fixture(audio)

            from .analysis.separation import SourceSeparator
            separator = SourceSeparator(device="cpu")
            try:
                mark("demucs_loading")
                sep = _timed(
                    "demucs_seconds",
                    lambda: separator.separate(audio, use_demucs=True, shifts=0),
                    timings,
                )
                if not str(sep.engine).startswith("Demucs"):
                    raise RuntimeError(f"Demucs cayó a fallback: {sep.engine}")
                if not sep.harmony.exists() or not sep.bass.exists():
                    raise RuntimeError("Demucs no produjo stems requeridos")
                mark("demucs_ok", demucs_model="ok")

                from .analysis.beat_tracker import detect_beats
                mark("beat_this_loading")
                beat = _timed(
                    "beat_this_seconds",
                    lambda: detect_beats(audio, 4.0, prefer_model=True),
                    timings,
                )
                if not str(beat.get("engine", "")).startswith("Beat This"):
                    raise RuntimeError(f"Beat This cayó a fallback: {beat.get('engine')}")
                mark("beat_this_ok", beat_this_model="ok")

                from .analysis.chord_lv import LVChordiaEngine
                mark("lv_chordia_loading")
                lv = LVChordiaEngine(device="cpu")
                lv_out = _timed(
                    "lv_chordia_seconds",
                    lambda: lv.analyze_variants(sep.harmony, vocabularies=("submission",)),
                    timings,
                )
                if "submission" not in lv_out:
                    raise RuntimeError("LV-Chordia no devolvió la salida submission")
                mark("lv_chordia_ok", lv_chordia_model="ok")

                # Validate BTC inference directly inside the packaged runtime first.
                # This separates model/package correctness from subprocess lifecycle.
                from .analysis.chord_btc import BTCChordEngine
                mark("btc_direct_loading")
                btc = BTCChordEngine(device="cpu")
                btc_out = _timed(
                    "btc_direct_seconds",
                    lambda: btc.analyze(sep.harmony, timeout_seconds=0, isolated=False),
                    timings,
                )
                if btc_out is None:
                    raise RuntimeError("BTC directo no devolvió salida")
                mark("btc_direct_ok", btc_model="ok")

                # Also validate the production isolation boundary used by the app.
                # It receives a separate timeout so a child-process failure is explicit.
                mark("btc_isolated_loading")
                btc_isolated = BTCChordEngine(device="cpu")
                btc_isolated_out = _timed(
                    "btc_isolated_seconds",
                    lambda: btc_isolated.analyze(sep.harmony, timeout_seconds=150, isolated=True),
                    timings,
                )
                if btc_isolated_out is None:
                    raise RuntimeError("BTC aislado no devolvió salida")
                mark("btc_isolated_ok", btc_isolation="ok")
            finally:
                separator.close()

        import torch
        payload = dict(state)
        payload.update({
            "status": "CHORDSCOPE_FULL_ML_RUNTIME_SMOKE_OK",
            "stage": "complete",
            "torch": str(torch.__version__),
            "torch_cuda_build": torch.version.cuda,
            "cuda_available_on_runner": bool(torch.cuda.is_available()),
            "demucs_model": "ok",
            "beat_this_model": "ok",
            "lv_chordia_model": "ok",
            "btc_model": "ok",
            "btc_isolation": "ok",
            "timings": dict(timings),
        })
        _marker(payload)
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    except Exception as exc:
        payload = dict(state)
        payload.update({
            "status": "CHORDSCOPE_FULL_ML_RUNTIME_SMOKE_FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "timings": dict(timings),
            "traceback": traceback.format_exc(),
        })
        _marker(payload)
        print(json.dumps(payload, ensure_ascii=False))
        return 1
