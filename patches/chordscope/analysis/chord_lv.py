from __future__ import annotations

import contextlib
import gc
import importlib.resources
from pathlib import Path
from typing import Callable

from .control import check_cancel
from .librosa_compat import patch_third_party_librosa_loader
from .types import ChordSegment

Progress = Callable[[int, str], None]


def _release_torch_cache() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _patch_lv_chordia_resources() -> None:
    """Resolve lv-chordia 1.1.0 resources from its bundled data directory.

    lv-chordia 1.1.0 calls importlib.resources.path("lv_chordia.data", ...).
    Its ``data`` directory is not a regular Python package, and Nuitka may not
    register it as an importable resource package even when the files are
    correctly bundled. Intercept only that legacy resource lookup and return the
    real bundled file path. Normal importlib.resources behavior is preserved for
    every other package. This works in both CPython source runs and Nuitka builds.
    """
    current = importlib.resources.path
    if getattr(current, "_chordscope_lv_patch", False):
        return
    try:
        import lv_chordia
        data_dir = Path(lv_chordia.__file__).resolve().parent / "data"
        if not data_dir.is_dir():
            return
    except Exception:
        return

    original = current

    @contextlib.contextmanager
    def _resource_path(package, resource):
        package_name = package if isinstance(package, str) else getattr(package, "__name__", "")
        if package_name == "lv_chordia.data":
            target = data_dir / str(resource)
            if not target.is_file():
                raise FileNotFoundError(str(target))
            yield target
            return
        with original(package, resource) as resolved:
            yield resolved

    _resource_path._chordscope_lv_patch = True
    importlib.resources.path = _resource_path


class LVChordiaEngine:
    name = "LV-Chordia"

    def __init__(self, device="cpu"):
        self.device = device

    @staticmethod
    def _convert(results, vocabulary):
        return [
            ChordSegment(
                start=float(r.get("start_time", r.get("start", 0))),
                end=float(r.get("end_time", r.get("end", 0))),
                label=str(r.get("chord", "N")),
                engine=f"LV-Chordia-{vocabulary}",
                confidence=float(r.get("confidence", 1.0) or 1.0),
            )
            for r in (results or [])
        ]

    def analyze_variants(self, audio_path, vocabularies=("full", "submission"), progress=None, cancel_event=None):
        vocabularies = tuple(vocabularies)
        if not vocabularies:
            return {}
        check_cancel(cancel_event)
        patch_third_party_librosa_loader()
        _patch_lv_chordia_resources()
        if progress:
            progress(5, f"{self.name} · cargando ensemble de 5 redes")
        try:
            from lv_chordia import LVChordiaSession
            outputs = {}
            with LVChordiaSession(chord_dict_name=vocabularies[0], device=self.device) as session:
                check_cancel(cancel_event)
                for idx, vocabulary in enumerate(vocabularies):
                    check_cancel(cancel_event)
                    if progress:
                        progress(25 + int(65 * idx / max(1, len(vocabularies))), f"{self.name} · inferencia/HMM {vocabulary}")
                    try:
                        results = session.infer(str(audio_path), vocabulary)
                    except TypeError:
                        results = session.infer(str(audio_path), chord_dict_name=vocabulary)
                    check_cancel(cancel_event)
                    outputs[vocabulary] = self._convert(results, vocabulary)
            if progress:
                progress(100, f"{self.name} · " + " · ".join(f"{k}:{len(v)}" for k, v in outputs.items()))
            return outputs
        except Exception:
            check_cancel(cancel_event)
            _patch_lv_chordia_resources()
            from lv_chordia.chord_recognition import chord_recognition
            outputs = {}
            for idx, vocabulary in enumerate(vocabularies):
                check_cancel(cancel_event)
                if progress:
                    progress(35 + int(55 * idx / max(1, len(vocabularies))), f"{self.name} legacy · {vocabulary}")
                try:
                    results = chord_recognition(audio_path=str(audio_path), chord_dict_name=vocabulary, device=self.device)
                except TypeError as exc:
                    if "unexpected keyword argument 'device'" not in str(exc):
                        raise
                    results = chord_recognition(audio_path=str(audio_path), chord_dict_name=vocabulary)
                check_cancel(cancel_event)
                outputs[vocabulary] = self._convert(results, vocabulary)
            if progress:
                progress(100, f"{self.name} legacy listo")
            return outputs
        finally:
            _release_torch_cache()

    def analyze(self, audio_path, progress=None, vocabulary="full", cancel_event=None):
        return self.analyze_variants(audio_path, (vocabulary,), progress, cancel_event=cancel_event)[vocabulary]
