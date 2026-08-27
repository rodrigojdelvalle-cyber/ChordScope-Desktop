from __future__ import annotations
import sys

if __name__ == "__main__" and "--runtime-smoke-test" in sys.argv:
    from chordscope.runtime_smoke import run_runtime_smoke
    raise SystemExit(run_runtime_smoke())

from chordscope.app import run

if __name__ == "__main__":
    raise SystemExit(run())
