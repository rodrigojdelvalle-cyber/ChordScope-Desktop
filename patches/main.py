from __future__ import annotations
import os
import sys

# Nuitka onefile extracts modules to a temporary directory. Librosa 0.11 imports
# Numba-decorated helpers with cache=True, and Numba can fail to locate a stable
# source file there. Disable Numba JIT at process startup for the packaged build,
# before importing any ChordScope/Librosa modules. Keep normal JIT behavior in
# source/development runs.
if "__compiled__" in globals():
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

if __name__ == "__main__" and "--runtime-smoke-test" in sys.argv:
    from chordscope.runtime_smoke import run_runtime_smoke
    raise SystemExit(run_runtime_smoke())

from chordscope.app import run

if __name__ == "__main__":
    raise SystemExit(run())
