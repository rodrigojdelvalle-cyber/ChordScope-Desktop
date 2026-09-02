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


def _load_constantq_eager(librosa_module):
    """Load librosa.core.constantq without going through Librosa's lazy proxy.

    Nuitka onefile can leave the lazy-loader route in a partially initialized
    state. Import the real module directly, discard any incomplete residue from
    a previous attempt, patch the tiny Numba helper, and bind public CQT symbols
    eagerly on both librosa and librosa.core.
    """
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
        # Never keep a half-imported module: Librosa's lazy loader would reuse it
        # and report a misleading circular-import AttributeError later.
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

    # Librosa only uses this helper to count powers of two. Replacing the
    # CPUDispatcher avoids Numba trying to inspect Nuitka-compiled bytecode.
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
    """Make Librosa deterministic for LV-Chordia/BTC in Nuitka packages.

    ChordScope itself uses native decoding/DSP. Third-party chord engines still
    need Librosa CQT. We replace the fragile audio submodule with native shims,
    then eagerly import/bind constantq so no packaged code relies on Librosa's
    lazy-loader circular-import path.
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
    return librosa
