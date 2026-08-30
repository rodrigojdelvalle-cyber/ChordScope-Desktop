from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
from scipy import signal


def _prepare_numba_runtime() -> None:
    """Undo Nuitka's standalone no-JIT default before third-party Librosa imports.

    Librosa 0.11 uses Numba guvectorize/stencil internals that do not import
    correctly with NUMBA_DISABLE_JIT=1. ChordScope's own DSP does not depend on
    Numba, but LV-Chordia/BTC still require Librosa CQT, so enable the normal
    Numba path only at this compatibility boundary and give its cache a writable
    per-user temporary directory.
    """
    os.environ.pop("NUMBA_DISABLE_JIT", None)
    cache_dir = Path(tempfile.gettempdir()) / "ChordScopeNumbaCache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))


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


def _patch_constantq_numba_helper() -> None:
    """Replace Librosa's tiny JIT helper with pure Python in compiled builds.

    Nuitka onefile compiles the Python body wrapped by Numba's CPUDispatcher.
    Numba then attempts to inspect CPython bytecode for that already-compiled
    function and raises ``RuntimeError: Compiled function bytecode used``.
    The helper only counts powers of two, so an equivalent pure-Python function
    avoids that unsupported boundary without changing CQT mathematics.
    """
    try:
        from librosa.core import constantq

        def _num_two_factors(x):
            value = int(x)
            if value <= 0:
                return 0
            count = 0
            while value % 2 == 0:
                count += 1
                value //= 2
            return count

        constantq.__dict__["__num_two_factors"] = _num_two_factors
    except Exception:
        # The runtime smoke test validates this compatibility path end-to-end.
        pass


def patch_third_party_librosa_loader() -> None:
    """Make Librosa usable by bundled third-party chord engines under Nuitka.

    ChordScope itself no longer depends on ``librosa.load``. LV-Chordia and
    BTC still use Librosa CQT. The CQT module imports ``librosa.core.audio``
    for resampling, and that real module contains Numba JIT/stencil decorators
    that are fragile in onefile builds. Install a small native shim before CQT
    imports occur, while leaving Numba enabled for Librosa's CQT utilities.
    """
    _prepare_numba_runtime()
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

    _patch_constantq_numba_helper()
