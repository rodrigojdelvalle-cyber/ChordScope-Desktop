from __future__ import annotations

import os
from pathlib import Path


def app_root() -> Path:
    # Nuitka onefile exposes extracted files through normal __file__ paths.
    return Path(__file__).resolve().parents[2]


def resource_path(*parts: str) -> Path:
    return app_root().joinpath(*parts)


def models_root() -> Path:
    p = resource_path("vendor_models")
    p.mkdir(parents=True, exist_ok=True)
    return p


def writable_cache_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(Path.home())
    p = Path(base) / "ChordScope" / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_variant() -> str:
    """Return the runtime package flavour without requiring a CUDA-capable host."""
    explicit = os.environ.get("CHORDSCOPE_BUILD_VARIANT")
    if explicit:
        return explicit
    try:
        import torch
        return "nvidia-cuda" if torch.version.cuda else "cpu"
    except Exception:
        return "unknown"
