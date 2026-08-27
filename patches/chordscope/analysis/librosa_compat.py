from __future__ import annotations

import sys
import types

import numpy as np
from scipy import signal


def _native_resample(y, *, orig_sr, target_sr, res_type="soxr_hq", fix=True, scale=False, axis=-1, **_kwargs):
    """Small librosa.audio.resample-compatible shim for third-party CQT code."""
    x = np.asarray(y)
    orig = float(orig_sr)
    target = float(target_sr)
    if orig <= 0 or target <= 0:
        raise ValueError("orig_sr and target_sr must be positive")
    if abs(orig - target) <= 1e-12:
        out = np.array(x, copy=True)
    else:
        from fractions import Fraction
        frac = Fraction(target / orig).limit_denominator(10000)
        out = signal.resample_poly(x, frac.numerator, frac.denominator, axis=axis)
    if scale:
        out = out / np.sqrt(target / orig)
    return np.asarray(out, dtype=x.dtype if np.issubdtype(x.dtype, np.floating) else np.float32)


def _native_to_mono(y):
    x = np.asarray(y)
    if x.ndim <= 1:
        return x
    return np.mean(x, axis=tuple(range(x.ndim - 1)))


def patch_third_party_librosa_loader() -> None:
    """Make Librosa usable by bundled third-party chord engines under Nuitka.

    ChordScope itself no longer depends on ``librosa.load``. LV-Chordia and
    BTC still use Librosa CQT. The CQT module imports ``librosa.core.audio``
    for resampling, and that real module contains Numba JIT/stencil decorators
    that are fragile in onefile builds. Install a small native shim before CQT
    imports occur.
    """
    import librosa
    from .audio import load_mono

    def _load(path, *, sr=22050, mono=True, offset=0.0, duration=None, dtype=np.float32, res_type="soxr_hq", **_kwargs):
        y, rate = load_mono(path, sr=None if sr is None else int(sr))
        if offset:
            y = y[int(max(0.0, float(offset)) * rate):]
        if duration is not None:
            y = y[: int(max(0.0, float(duration)) * rate)]
        if dtype is not None:
            y = y.astype(dtype, copy=False)
        return y, rate

    def _get_samplerate(path):
        from .dsp import load_audio
        _y, rate = load_audio(path, sr=None)
        return int(rate)

    def _get_duration(*, y=None, sr=22050, path=None, **_kwargs):
        if path is not None:
            yy, rr = load_mono(path, sr=None)
            return float(len(yy) / rr) if rr else 0.0
        if y is None:
            return 0.0
        return float(np.asarray(y).shape[-1] / float(sr))

    stub = types.ModuleType("librosa.core.audio")
    stub.__dict__.update({
        "load": _load,
        "resample": _native_resample,
        "to_mono": _native_to_mono,
        "get_samplerate": _get_samplerate,
        "get_duration": _get_duration,
        "__all__": ["load", "resample", "to_mono", "get_samplerate", "get_duration"],
    })
    sys.modules["librosa.core.audio"] = stub

    librosa.__dict__["load"] = _load
    librosa.__dict__["resample"] = _native_resample
    librosa.__dict__["to_mono"] = _native_to_mono
    librosa.__dict__["get_samplerate"] = _get_samplerate
    librosa.__dict__["get_duration"] = _get_duration

    core = sys.modules.get("librosa.core")
    if core is not None:
        core.__dict__["audio"] = stub
        core.__dict__["load"] = _load
        core.__dict__["resample"] = _native_resample
