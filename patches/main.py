from __future__ import annotations
import os
import sys


def _patch_numba_onefile_cache() -> None:
    """Apply narrow Numba/Librosa compatibility shims for Nuitka onefile builds.

    Nuitka compiles Python modules and extracts the application into a temporary
    onefile directory. Librosa 0.11 decorates several helpers with Numba. Disk
    caching is only a startup optimisation, so it can be disabled safely.

    In addition, Numba's DUFunc for librosa.util.utils._phasor_angles can reach
    compiled-function bytecode that is not executable from a Nuitka onefile
    payload (RuntimeError: "Compiled function bytecode used").  The helper is
    mathematically just cos(x) + 1j*sin(x), so for packaged builds only we
    replace that single implementation with its NumPy equivalent.  This keeps
    Librosa's public phasor/CQT behaviour intact without weakening normal source
    execution.
    """
    if "__compiled__" not in globals():
        return

    import tempfile
    from pathlib import Path

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
        # The runtime smoke test below remains the authoritative verification.
        pass

    try:
        import numpy as np
        import librosa.util.utils as librosa_utils

        def _phasor_angles_numpy(x):
            x_arr = np.asarray(x)
            return np.cos(x_arr) + 1j * np.sin(x_arr)

        librosa_utils._phasor_angles = _phasor_angles_numpy
    except Exception:
        # Do not mask import/startup failures; packaged smoke will report them.
        pass


_patch_numba_onefile_cache()

if __name__ == "__main__" and "--runtime-smoke-test" in sys.argv:
    from chordscope.runtime_smoke import run_runtime_smoke
    raise SystemExit(run_runtime_smoke())

from chordscope.app import run

if __name__ == "__main__":
    raise SystemExit(run())
