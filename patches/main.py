from __future__ import annotations
import os
import sys


def _patch_numba_onefile_cache() -> None:
    """Keep Numba JIT enabled but disable disk caching in Nuitka onefile builds.

    Nuitka compiles Python modules and extracts the application into a temporary
    onefile directory. Librosa 0.11 decorates several helpers with cache=True;
    Numba then tries to create cache locators from source .py files that are not
    available as normal source modules in the compiled distribution. Disk
    caching is only a startup optimisation, so disabling it is safe while
    preserving the JIT/vectorized implementations Librosa expects.
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

        # @jit(..., cache=True) / @njit(..., cache=True)
        if hasattr(dispatcher, "Dispatcher"):
            dispatcher.Dispatcher.enable_caching = _no_disk_cache

        # @vectorize / @guvectorize(..., cache=True).  For gufuncs the cache
        # flag is also forwarded to build_gufunc_wrapper, so overriding only
        # UFuncDispatcher.enable_caching is not sufficient.  Force the
        # decorators to consume cache=True as cache=False before Librosa loads.
        from numba.np.ufunc import decorators as ufunc_decorators

        def _get_cache_disabled(cls, kwargs):
            kwargs.pop("cache", False)
            return False

        ufunc_decorators._BaseVectorize.get_cache = classmethod(_get_cache_disabled)

        # Defensive fallback for element-wise ufunc dispatchers.
        from numba.np.ufunc import ufuncbuilder
        cls = getattr(ufuncbuilder, "UFuncDispatcher", None)
        if cls is not None and hasattr(cls, "enable_caching"):
            cls.enable_caching = _no_disk_cache
    except Exception:
        # Do not make application startup depend on this compatibility shim.
        # The runtime smoke test will still catch any real packaged failure.
        pass


_patch_numba_onefile_cache()

if __name__ == "__main__" and "--runtime-smoke-test" in sys.argv:
    from chordscope.runtime_smoke import run_runtime_smoke
    raise SystemExit(run_runtime_smoke())

from chordscope.app import run

if __name__ == "__main__":
    raise SystemExit(run())
