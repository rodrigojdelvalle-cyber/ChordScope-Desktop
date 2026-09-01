from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .audio import load_mono
from .bass import detect_bass_notes
from .beat_tracker import detect_beats
from .chord_btc import BTCChordEngine
from .chord_lv import LVChordiaEngine
from .consensus import attach_bass, decode_sequence
from .control import AnalysisCancelled, check_cancel
from .key import estimate_key, global_chroma
from .separation import SourceSeparator
from .template_engine import extract_beat_features
from .types import AnalysisResult

Progress = Callable[[int, str], None]


@dataclass(frozen=True)
class AnalysisProfile:
    name: str
    use_demucs: bool
    demucs_shifts: int
    prefer_beat_model: bool
    lv_vocabularies: tuple[str, ...]
    btc_mode: str  # off | adaptive | always
    btc_timeout_cpu: int
    btc_timeout_cuda: int


PROFILES = {
    "fast": AnalysisProfile(
        name="Rápido",
        use_demucs=False,
        demucs_shifts=0,
        prefer_beat_model=False,
        lv_vocabularies=(),
        btc_mode="off",
        btc_timeout_cpu=0,
        btc_timeout_cuda=0,
    ),
    "balanced": AnalysisProfile(
        name="Equilibrado",
        use_demucs=True,
        demucs_shifts=0,
        prefer_beat_model=True,
        lv_vocabularies=("submission",),
        btc_mode="adaptive",
        btc_timeout_cpu=90,
        btc_timeout_cuda=60,
    ),
    "deep": AnalysisProfile(
        name="Profundo",
        use_demucs=True,
        demucs_shifts=1,
        prefer_beat_model=True,
        lv_vocabularies=("full", "submission"),
        btc_mode="always",
        btc_timeout_cpu=180,
        btc_timeout_cuda=120,
    ),
}


def normalize_profile(profile: str | None) -> AnalysisProfile:
    return PROFILES.get(str(profile or "balanced").strip().lower(), PROFILES["balanced"])


def device_auto() -> str:
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _subprogress(cb: Progress | None, lo: int, hi: int, prefix: str):
    if cb is None:
        return None

    def f(p: int, msg: str):
        x = lo + (hi - lo) * max(0, min(100, p)) / 100
        cb(int(round(x)), f"{prefix} · {msg}")

    return f


def _preliminary_quality(features, engine_segments, key_root: int, key_mode: str) -> dict:
    cells, _ = decode_sequence(features, engine_segments, key_root, key_mode)
    if not cells:
        return {"confidence": 0.0, "no_chord_ratio": 1.0, "cells": 0}
    confidence = float(np.mean([float(c.confidence) for c in cells]))
    no_chord_ratio = float(np.mean([1.0 if c.noChord else 0.0 for c in cells]))
    return {"confidence": confidence, "no_chord_ratio": no_chord_ratio, "cells": len(cells)}


def _should_run_btc(profile: AnalysisProfile, quality: dict, engine_segments: dict) -> tuple[bool, str]:
    if profile.btc_mode == "off":
        return False, "desactivado por perfil"
    if profile.btc_mode == "always":
        return True, "perfil profundo"
    lv_segments = sum(len(v) for k, v in engine_segments.items() if k.startswith("lv_"))
    if lv_segments == 0:
        return True, "LV-Chordia no produjo segmentos"
    if quality["confidence"] < 0.72:
        return True, f"confianza preliminar {quality['confidence']:.2f}"
    if quality["no_chord_ratio"] > 0.28:
        return True, f"N.C. preliminar {quality['no_chord_ratio']:.0%}"
    return False, f"consenso preliminar suficiente ({quality['confidence']:.2f})"


def analyze_file(
    audio_path: str | Path,
    progress: Progress | None = None,
    *,
    profile: str = "balanced",
    cancel_event=None,
) -> AnalysisResult:
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(path)

    cfg = normalize_profile(profile)
    device = device_auto()
    if progress:
        progress(1, f"ChordScope 2.0.2 · {cfg.name} · motor {device.upper()}")

    check_cancel(cancel_event)
    y_probe, sr_probe = load_mono(path, sr=22050)
    duration = float(len(y_probe) / sr_probe)
    del y_probe

    separator = SourceSeparator(device=device)
    engine_status: dict[str, str] = {
        "profile": cfg.name,
        "device": device,
    }
    try:
        sep = separator.separate(
            path,
            _subprogress(progress, 3, 23, "Separación"),
            use_demucs=cfg.use_demucs,
            shifts=cfg.demucs_shifts,
            cancel_event=cancel_event,
        )
        engine_status["separation"] = sep.engine
        check_cancel(cancel_event)

        beat_info = detect_beats(
            path,
            duration,
            _subprogress(progress, 23, 37, "Tempo/Downbeat"),
            prefer_model=cfg.prefer_beat_model,
            cancel_event=cancel_event,
        )
        beat_times = beat_info["beats"]
        if len(beat_times) < 2:
            raise RuntimeError("No se pudo construir una grilla de beats confiable")
        engine_status["tempo"] = beat_info["engine"]
        check_cancel(cancel_event)

        features = extract_beat_features(
            sep.harmony,
            beat_times,
            duration,
            _subprogress(progress, 37, 49, "Armonía"),
        )
        check_cancel(cancel_event)
        prelim_key = estimate_key(global_chroma(sep.harmony), beats=None)

        engine_segments: dict[str, list] = {}
        if cfg.lv_vocabularies:
            lv = LVChordiaEngine(device=device)
            try:
                lv_outputs = lv.analyze_variants(
                    sep.harmony,
                    vocabularies=cfg.lv_vocabularies,
                    progress=_subprogress(progress, 49, 68, "LV-Chordia"),
                    cancel_event=cancel_event,
                )
                for vocabulary in cfg.lv_vocabularies:
                    key = f"lv_{vocabulary}"
                    engine_segments[key] = lv_outputs.get(vocabulary, [])
                    engine_status[f"lv_chordia_{vocabulary}"] = "ok"
            except AnalysisCancelled:
                raise
            except Exception as exc:
                for vocabulary in cfg.lv_vocabularies:
                    engine_segments[f"lv_{vocabulary}"] = []
                    engine_status[f"lv_chordia_{vocabulary}"] = f"error: {type(exc).__name__}: {exc}"
        else:
            engine_status["lv_chordia"] = "omitido por perfil rápido"
            if progress:
                progress(68, "LV-Chordia · omitido por perfil rápido")

        check_cancel(cancel_event)
        quality = _preliminary_quality(features, engine_segments, prelim_key.root, prelim_key.mode)
        run_btc, btc_reason = _should_run_btc(cfg, quality, engine_segments)
        engine_status["preliminary_confidence"] = f"{quality['confidence']:.3f}"
        engine_status["preliminary_no_chord_ratio"] = f"{quality['no_chord_ratio']:.3f}"

        if run_btc:
            btc = BTCChordEngine(device=device)
            timeout = cfg.btc_timeout_cuda if device.startswith("cuda") else cfg.btc_timeout_cpu
            try:
                if progress:
                    progress(69, f"BTC Transformer · activado · {btc_reason} · límite {timeout}s")
                engine_segments["btc"] = btc.analyze(
                    sep.harmony,
                    _subprogress(progress, 69, 82, "BTC Transformer"),
                    timeout_seconds=timeout,
                    cancel_event=cancel_event,
                    isolated=True,
                )
                engine_status["btc"] = f"ok · {btc_reason}"
            except AnalysisCancelled:
                raise
            except TimeoutError as exc:
                engine_segments["btc"] = []
                engine_status["btc"] = f"timeout controlado: {exc}"
                if progress:
                    progress(82, "BTC Transformer · timeout controlado · continuando sin BTC")
            except Exception as exc:
                engine_segments["btc"] = []
                engine_status["btc"] = f"error controlado: {type(exc).__name__}: {exc}"
                if progress:
                    progress(82, f"BTC Transformer · omitido por error {type(exc).__name__}")
        else:
            engine_segments["btc"] = []
            engine_status["btc"] = f"omitido adaptativamente: {btc_reason}"
            if progress:
                progress(82, f"BTC Transformer · omitido · {btc_reason}")

        check_cancel(cancel_event)
        if not any(engine_segments.values()):
            engine_status["warning"] = "Motores ML externos omitidos/no disponibles; STFT-template + contexto activos"

        cells, _ = decode_sequence(features, engine_segments, prelim_key.root, prelim_key.mode)
        refined_key = estimate_key(global_chroma(sep.harmony), beats=cells)
        cells, _ = decode_sequence(features, engine_segments, refined_key.root, refined_key.mode)
        refined_key = estimate_key(global_chroma(sep.harmony), beats=cells)

        check_cancel(cancel_event)
        bass = detect_bass_notes(
            sep.bass,
            beat_times,
            duration,
            _subprogress(progress, 82, 92, "Walking bass"),
        )
        check_cancel(cancel_event)
        attach_bass(cells, bass)

        if progress:
            progress(95, "Contexto funcional · ii–V–I · dominantes secundarios")
        period = float(beat_info["period"])
        for cell, t in zip(cells, beat_times):
            cell.time = float(t)

        result = AnalysisResult(
            bpm=float(beat_info["bpm"]),
            beatPeriod=period,
            beatOffset=float(beat_times[0] if beat_times else 0.0),
            keyRoot=refined_key.root,
            keyMode=refined_key.mode,
            scaleRoot=refined_key.root,
            scaleType=refined_key.scale_type,
            scaleConfidence=refined_key.confidence,
            keyConfidence=refined_key.confidence,
            keyAlternatives=refined_key.alternatives,
            keyEngine=refined_key.engine,
            tempoConfidence=float(beat_info["confidence"]),
            tempoEngine=str(beat_info["engine"]),
            duration=duration,
            meter=int(beat_info["meter"]),
            beats=cells,
            engineStatus=engine_status,
            separation=sep.engine,
        )
        if progress:
            progress(100, f"Listo · {result.bpm:.1f} BPM · {refined_key.mode} · {cfg.name}")
        return result
    finally:
        separator.close()
