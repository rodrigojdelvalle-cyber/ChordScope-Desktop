from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .dsp import load_audio

Progress = Callable[[int, str], None]


def load_mono(path: str | Path, sr: int | None = 22050) -> tuple[np.ndarray, int]:
    y, rate = load_audio(path, sr=sr)
    y = np.asarray(y, dtype=np.float32)
    if y.size:
        peak = float(np.max(np.abs(y)))
        if peak > 1e-9:
            y = y / max(1.0, peak)
    return y, int(rate)


def probe_audio(path: str | Path, n_peaks: int = 1600, progress: Progress | None = None) -> dict:
    if progress:
        progress(10, "Leyendo audio · decoder nativo")
    y, sr = load_mono(path, sr=22050)
    if progress:
        progress(60, "Calculando forma de onda")
    duration = float(len(y) / sr) if sr else 0.0
    if not len(y):
        peaks: list[list[float]] = []
    else:
        n = max(1, min(n_peaks, len(y)))
        edges = np.linspace(0, len(y), n + 1, dtype=np.int64)
        peaks = []
        for i in range(n):
            chunk = y[edges[i]:edges[i + 1]]
            peaks.append([float(chunk.min()), float(chunk.max())] if chunk.size else [0.0, 0.0])
    if progress:
        progress(100, "Audio listo")
    return {"duration": duration, "sample_rate": sr, "waveform": peaks, "frames": int(len(y))}
