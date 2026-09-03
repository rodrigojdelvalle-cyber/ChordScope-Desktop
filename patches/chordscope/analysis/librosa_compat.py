from __future__ import annotations

import importlib
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
from scipy import signal


_CQT_EXPORTS = ("cqt", "hybrid_cqt", "pseudo_cqt", "vqt")


def _prepare_numba_runtime() -> None:
    """Prepare a writable Numba runtime for packaged third-party Librosa code."""
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


def _patch_librosa_numba_helpers() -> None:
    """Replace small Librosa Numba helpers that are unsafe after Nuitka compile.

    LV-Chordia reaches Librosa pitch tracking from hybrid_cqt. In a Nuitka
    executable Numba can try to inspect CPython bytecode for functions that are
    already native-compiled and raise ``RuntimeError: Compiled function bytecode
    used``. Keep the math in NumPy for the two helpers observed on this path:
    parabolic interpolation and local maxima detection.
    """
    pitch = importlib.import_module("librosa.core.pitch")
    utils = importlib.import_module("librosa.util.utils")

    def _parabolic_interpolation_numpy(x: np.ndarray, *, axis: int = -2) -> np.ndarray:
        arr = np.asarray(x)
        xi = np.swapaxes(arr, -1, axis)
        shiftsi = np.zeros_like(xi)
        if xi.shape[-1] < 3:
            return np.swapaxes(shiftsi, -1, axis)

        left = xi[..., :-2]
        center = xi[..., 1:-1]
        right = xi[..., 2:]
        a = right + left - 2.0 * center
        b = (right - left) / 2.0

        valid = (np.abs(b) < np.abs(a)) & (a != 0)
        inner = np.zeros_like(center)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(-b, a, out=inner, where=valid)
        shiftsi[..., 1:-1] = inner
        return np.swapaxes(shiftsi, -1, axis)

    def _localmax_numpy(x: np.ndarray, *, axis: int = 0) -> np.ndarray:
        """NumPy equivalent of librosa.util.localmax edge semantics."""
        arr = np.asarray(x)
        if arr.ndim == 0:
            return np.asarray(False)
        axis_norm = np.core.numeric.normalize_axis_index(axis, arr.ndim)
        xi = np.moveaxis(arr, axis_norm, -1)
        out = np.zeros(xi.shape, dtype=bool)
        n = xi.shape[-1]
        if n == 0:
            return np.moveaxis(out, -1, axis_norm)
        if n == 1:
            out[..., 0] = True
            return np.moveaxis(out, -1, axis_norm)

        # Librosa localmax uses a strict comparison to the previous sample and
        # a non-strict comparison to the following sample. The first element
        # therefore cannot be a local maximum; the final element is compared
        # only against its predecessor.
        if n > 2:
            out[..., 1:-1] = (xi[..., 1:-1] > xi[..., :-2]) & (xi[..., 1:-1] >= xi[..., 2:])
        out[..., -1] = xi[..., -1] > xi[..., -2]
        return np.moveaxis(out, -1, axis_norm)

    pitch.__dict__["_parabolic_interpolation"] = _parabolic_interpolation_numpy
    utils.__dict__["localmax"] = _localmax_numpy
    # pitch imports localmax into its module namespace, so replace that bound
    # reference too; otherwise piptrack would keep calling the gufunc wrapper.
    if "localmax" in pitch.__dict__:
        pitch.__dict__["localmax"] = _localmax_numpy


def _load_constantq_eager(librosa_module):
    """Load librosa.core.constantq without going through Librosa's lazy proxy."""
    module_name = "librosa.core.constantq"
    current = sys.modules.get(module_name)
    if current is not None and not callable(getattr(current, "hybrid_cqt", None)):
        sys.modules.pop(module_name, None)
        core = sys.modules.get("librosa.core")
        if core is not None:
            core.__dict__.pop("constantq", None)

    try:
        constantq = importlib.import_module(module_name)
    except Exception:
        broken = sys.modules.get(module_name)
        if broken is not None and not callable(getattr(broken, "hybrid_cqt", None)):
            sys.modules.pop(module_name, None)
        raise

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

    core = importlib.import_module("librosa.core")
    core.__dict__["constantq"] = constantq
    for name in _CQT_EXPORTS:
        func = getattr(constantq, name, None)
        if callable(func):
            librosa_module.__dict__[name] = func
            core.__dict__[name] = func

    if not callable(librosa_module.__dict__.get("hybrid_cqt")):
        raise RuntimeError("Librosa CQT compatibility bootstrap did not bind hybrid_cqt")
    return constantq


def patch_third_party_librosa_loader():
    """Make Librosa deterministic for LV-Chordia/BTC in Nuitka packages."""
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
    stub.__package__ = "librosa.core"
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

    _load_constantq_eager(librosa)
    _patch_librosa_numba_helpers()
    return librosa
