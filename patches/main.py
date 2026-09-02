from __future__ import annotations
import json
import os
import sys
import traceback
from pathlib import Path


def _bootstrap_marker(env_name: str, payload: dict) -> None:
    """Write a diagnostic as early as possible, before optional runtime imports."""
    target = os.environ.get(env_name)
    if not target:
        return
    try:
        Path(target).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _run_packaged_smoke(flag: str, env_name: str, smoke_kind: str) -> None:
    """Dispatch CI smoke modes before normal startup shims can interfere.

    Use explicit imports rather than __import__ so Nuitka can see and include the
    two smoke modules in both onefile and standalone distributions.
    """
    if __name__ != "__main__" or flag not in sys.argv:
        return
    _bootstrap_marker(env_name, {
        "status": "CHORDSCOPE_PACKAGED_SMOKE_BOOTSTRAP",
        "flag": flag,
        "argv": list(sys.argv),
        "compiled": "__compiled__" in globals(),
    })
    try:
        if smoke_kind == "basic":
            from chordscope.runtime_smoke import run_runtime_smoke
            func = run_runtime_smoke
        elif smoke_kind == "full":
            from chordscope.full_runtime_smoke import run_full_runtime_smoke
            func = run_full_runtime_smoke
        else:
            raise RuntimeError(f"Unknown smoke kind: {smoke_kind}")
        raise SystemExit(func())
    except SystemExit:
        raise
    except BaseException as exc:
        _bootstrap_marker(env_name, {
            "status": "CHORDSCOPE_PACKAGED_SMOKE_BOOTSTRAP_FAILED",
            "flag": flag,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise SystemExit(1)


_run_packaged_smoke(
    "--runtime-smoke-test",
    "CHORDSCOPE_SMOKE_MARKER",
    "basic",
)
_run_packaged_smoke(
    "--full-runtime-smoke-test",
    "CHORDSCOPE_FULL_SMOKE_MARKER",
    "full",
)


def _patch_numba_onefile_cache() -> None:
    """Apply narrow Numba/Librosa compatibility shims for Nuitka packaged builds."""
    if "__compiled__" not in globals():
        return

    import tempfile

    cache_dir = Path(tempfile.gettempdir()) / "ChordScopeNumbaCache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))

    try:
        from numba.core import dispatcher

        def _no_disk_cache(self):
            return None

        if hasattr(dispatcher, "Dispatcher"):
            dispatcher.Dispatcher.enable_caching = _no_disk_cache

        from numba.np.ufunc import decorators as ufunc_decorators

        def _get_cache_disabled(cls, kwargs):
            kwargs.pop("cache", False)
            return False

        ufunc_decorators._BaseVectorize.get_cache = classmethod(_get_cache_disabled)

        from numba.np.ufunc import ufuncbuilder
        cls = getattr(ufuncbuilder, "UFuncDispatcher", None)
        if cls is not None and hasattr(cls, "enable_caching"):
            cls.enable_caching = _no_disk_cache
    except Exception:
        pass

    try:
        import numpy as np
        import librosa.util.utils as librosa_utils

        def _phasor_angles_numpy(x):
            x_arr = np.asarray(x)
            return np.cos(x_arr) + 1j * np.sin(x_arr)

        librosa_utils._phasor_angles = _phasor_angles_numpy
    except Exception:
        pass


_patch_numba_onefile_cache()

if __name__ == "__main__" and "--btc-worker" in sys.argv:
    from chordscope.analysis.chord_btc import run_btc_worker
    i = sys.argv.index("--btc-worker")
    args = sys.argv[i + 1:i + 5]
    if len(args) != 4:
        raise SystemExit(64)
    raise SystemExit(run_btc_worker(*args))

from chordscope.app import run

if __name__ == "__main__":
    raise SystemExit(run())
