from __future__ import annotations
from typing import Callable
from .audio import load_mono
from .librosa_compat import patch_third_party_librosa_loader
from .resources import models_root
from .types import ChordSegment
Progress=Callable[[int,str],None]


class BTCChordEngine:
    name="BTC Transformer"
    def __init__(self,device="cpu"):
        self.device=device
        self._model=None

    def _load(self):
        if self._model is not None:
            return self._model
        patch_third_party_librosa_loader()
        from transformers import AutoModel
        local=models_root()/"btc-chord"
        source=str(local) if (local/"config.json").exists() else "puar-playground/btc-chord"
        kwargs=dict(trust_remote_code=True,large_voca=True,device=self.device)
        if source==str(local):
            kwargs["local_files_only"]=True
        self._model=AutoModel.from_pretrained(source,**kwargs)
        return self._model

    def analyze(self,audio_path,progress=None):
        if progress:
            progress(8,f"{self.name} · cargando modelo")
        model=self._load()
        if progress:
            progress(35,f"{self.name} · log-CQT")
        y,_=load_mono(audio_path,sr=22050)
        results=model.predict(y)
        out=[ChordSegment(start=float(r.get("start",r.get("start_time",0))),end=float(r.get("end",r.get("end_time",0))),label=str(r.get("chord","N")),engine=self.name,confidence=float(r.get("confidence",1.0) or 1.0)) for r in (results or [])]
        if progress:
            progress(100,f"{self.name} · {len(out)} segmentos")
        return out
