from __future__ import annotations
from typing import Callable
from .librosa_compat import patch_third_party_librosa_loader
from .types import ChordSegment
Progress=Callable[[int,str],None]


class LVChordiaEngine:
    name="LV-Chordia"
    def __init__(self,device="cpu"):
        self.device=device

    @staticmethod
    def _convert(results,vocabulary):
        return [ChordSegment(start=float(r.get("start_time",r.get("start",0))),end=float(r.get("end_time",r.get("end",0))),label=str(r.get("chord","N")),engine=f"LV-Chordia-{vocabulary}",confidence=float(r.get("confidence",1.0) or 1.0)) for r in (results or [])]

    def analyze_variants(self,audio_path,vocabularies=("full","submission"),progress=None):
        vocabularies=tuple(vocabularies)
        patch_third_party_librosa_loader()
        if progress:
            progress(5,f"{self.name} · cargando ensemble de 5 redes")
        try:
            from lv_chordia import LVChordiaSession
            outputs={}
            with LVChordiaSession(chord_dict_name="submission",device=self.device) as session:
                for idx,vocabulary in enumerate(vocabularies):
                    if progress:
                        progress(25+int(65*idx/max(1,len(vocabularies))),f"{self.name} · HMM {vocabulary}")
                    try:
                        results=session.infer(str(audio_path),vocabulary)
                    except TypeError:
                        results=session.infer(str(audio_path),chord_dict_name=vocabulary)
                    outputs[vocabulary]=self._convert(results,vocabulary)
            if progress:
                progress(100,f"{self.name} · "+" · ".join(f"{k}:{len(v)}" for k,v in outputs.items()))
            return outputs
        except Exception:
            from lv_chordia.chord_recognition import chord_recognition
            outputs={}
            for idx,vocabulary in enumerate(vocabularies):
                if progress:
                    progress(35+int(55*idx/max(1,len(vocabularies))),f"{self.name} legacy · {vocabulary}")
                results=chord_recognition(audio_path=str(audio_path),chord_dict_name=vocabulary,device=self.device)
                outputs[vocabulary]=self._convert(results,vocabulary)
            if progress:
                progress(100,f"{self.name} legacy listo")
            return outputs

    def analyze(self,audio_path,progress=None,vocabulary="full"):
        return self.analyze_variants(audio_path,(vocabulary,),progress)[vocabulary]
