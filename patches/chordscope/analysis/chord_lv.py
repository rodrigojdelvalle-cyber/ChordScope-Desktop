from __future__ import annotations

import gc
import importlib.machinery
import sys
import types
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


def _ensure_lv_data_namespace() -> None:
    """Expose bundled lv_chordia/data as an importable package in Nuitka builds.

    lv-chordia 1.1.0 ships its resource directory without an __init__.py. Nuitka
    correctly bundles the files via --include-package-data, but importlib.resources
    used by the legacy inference path still imports ``lv_chordia.data`` by name.
    Under the packaged executable that namespace is otherwise absent. Register a
    small namespace package pointing at the already-bundled data directory.
    """
    if "lv_chordia.data" in sys.modules:
        return
    try:
        import lv_chordia
        root = Path(lv_chordia.__file__).resolve().parent
        data_dir = root / "data"
        if not data_dir.is_dir():
            return
        module = types.ModuleType("lv_chordia.data")
        spec = importlib.machinery.ModuleSpec("lv_chordia.data", loader=None, is_package=True)
        spec.submodule_search_locations = [str(data_dir)]
        module.__spec__ = spec
        module.__package__ = "lv_chordia.data"
        module.__path__ = [str(data_dir)]
        module.__file__ = str(data_dir / "__init__.py")
        sys.modules["lv_chordia.data"] = module
        setattr(lv_chordia, "data", module)
    except Exception:
        pass


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
        _ensure_lv_data_namespace()
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
            _ensure_lv_data_namespace()
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
