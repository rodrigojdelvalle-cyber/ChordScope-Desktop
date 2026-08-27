from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch

from .audio import load_mono
from .bass import detect_bass_notes
from .beat_tracker import detect_beats
from .chord_btc import BTCChordEngine
from .chord_lv import LVChordiaEngine
from .consensus import attach_bass, decode_sequence
from .key import estimate_key, global_chroma
from .separation import SourceSeparator
from .template_engine import extract_beat_features
from .types import AnalysisResult

Progress = Callable[[int, str], None]


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


def analyze_file(audio_path: str | Path, progress: Progress | None = None) -> AnalysisResult:
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(path)
    device = device_auto()
    if progress:
        progress(1, f"ChordScope Desktop · motor {device.upper()}")

    y_probe, sr_probe = load_mono(path, sr=22050)
    duration = float(len(y_probe) / sr_probe)

    separator = SourceSeparator(device=device)
    engine_status = {}
    try:
        sep = separator.separate(path, _subprogress(progress, 3, 25, "Separación"))
        engine_status["separation"] = sep.engine

        beat_info = detect_beats(path, duration, _subprogress(progress, 25, 38, "Tempo/Downbeat"))
        beat_times = beat_info["beats"]
        if len(beat_times) < 2:
            raise RuntimeError("No se pudo construir una grilla de beats confiable")
        engine_status["tempo"] = beat_info["engine"]

        features = extract_beat_features(sep.harmony, beat_times, duration, _subprogress(progress, 38, 49, "Armonía"))
        prelim_key = estimate_key(global_chroma(sep.harmony), beats=None)

        engine_segments = {}
        lv = LVChordiaEngine(device=device)
        try:
            lv_outputs = lv.analyze_variants(
                sep.harmony,
                vocabularies=("full", "submission"),
                progress=_subprogress(progress, 49, 70, "LV-Chordia"),
            )
            engine_segments["lv_full"] = lv_outputs.get("full", [])
            engine_segments["lv_submission"] = lv_outputs.get("submission", [])
            engine_status["lv_chordia_full"] = "ok"
            engine_status["lv_chordia_submission"] = "ok"
        except Exception as exc:
            engine_segments["lv_full"] = []
            engine_segments["lv_submission"] = []
            engine_status["lv_chordia_full"] = f"error: {type(exc).__name__}: {exc}"
            engine_status["lv_chordia_submission"] = f"error: {type(exc).__name__}: {exc}"

        btc = BTCChordEngine(device=device)
        try:
            engine_segments["btc"] = btc.analyze(sep.harmony, _subprogress(progress, 70, 83, "BTC Transformer"))
            engine_status["btc"] = "ok"
        except Exception as exc:
            engine_segments["btc"] = []
            engine_status["btc"] = f"error: {type(exc).__name__}: {exc}"

        if not any(engine_segments.values()):
            engine_status["warning"] = "Motores ML externos no disponibles; se usará STFT-template + contexto"

        cells, _ = decode_sequence(features, engine_segments, prelim_key.root, prelim_key.mode)
        refined_key = estimate_key(global_chroma(sep.harmony), beats=cells)
        cells, _ = decode_sequence(features, engine_segments, refined_key.root, refined_key.mode)
        refined_key = estimate_key(global_chroma(sep.harmony), beats=cells)

        bass = detect_bass_notes(sep.bass, beat_times, duration, _subprogress(progress, 83, 92, "Walking bass"))
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
            progress(100, f"Listo · {result.bpm:.1f} BPM · {refined_key.mode}")
        return result
    finally:
        separator.close()
