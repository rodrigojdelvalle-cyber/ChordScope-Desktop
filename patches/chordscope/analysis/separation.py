from __future__ import annotations

import gc
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

from .audio import load_mono
from .dsp import hpss_native
from .resources import models_root

Progress = Callable[[int, str], None]

@dataclass(slots=True)
class SeparationResult:
    root: Path
    original: Path
    harmony: Path
    bass: Path
    drums: Optional[Path]
    vocals: Optional[Path]
    engine: str
    sample_rate: int

class SourceSeparator:
    _MODEL = "htdemucs_ft"
    _LOCAL_FILES=("htdemucs_ft.yaml","f7e0c4bc-ba3fe64a.th","d12395a8-e57c48e6.th","92cfc3b6-ef3bcb9c.th","04573f0d-f3cf25b2.th")
    def __init__(self,device:str="cpu"):
        self.device=device; self._separator=None; self._tempdirs=[]
    def close(self):
        self._separator=None; gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception: pass
        while self._tempdirs:
            try:self._tempdirs.pop().cleanup()
            except Exception:pass
    def _temp_root(self):
        td=tempfile.TemporaryDirectory(prefix="chordscope_sep_"); self._tempdirs.append(td); return Path(td.name)
    @classmethod
    def _local_repo_ready(cls,path): return path.is_dir() and all((path/n).is_file() for n in cls._LOCAL_FILES)
    def _build_separator(self,progress):
        from demucs_infer.api import Separator
        local_repo=models_root()/"demucs"
        kwargs=dict(model=self._MODEL,device=self.device,shifts=1,overlap=.25,split=True,progress=False)
        if self._local_repo_ready(local_repo): kwargs["repo"]=local_repo; engine="Demucs HTDemucs-FT · offline"
        else: engine="Demucs HTDemucs-FT · runtime model"
        if progress:progress(6,"Cargando HTDemucs-FT")
        def callback(info):
            if progress is None:return
            try:
                length=max(1,int(info.get("audio_length",1))); offset=max(0,int(info.get("segment_offset",0))); mi=max(0,int(info.get("model_idx_in_bag",0))); models=max(1,int(info.get("models",1)))
                frac=(mi+min(1.0,offset/length))/models; progress(18+int(72*frac),"Separando drums · bass · other · vocals")
            except Exception:pass
        kwargs["callback"]=callback; self._separator=Separator(**kwargs); return self._separator,engine
    @staticmethod
    def _decode_for_separator(audio_path):
        import torch
        y,sr=load_mono(audio_path,sr=None)
        wav=torch.from_numpy(np.ascontiguousarray(y[None,:]))
        return wav,int(sr)
    def separate(self,audio_path,progress=None):
        audio_path=Path(audio_path); root=self._temp_root()
        try:
            if progress:progress(3,"Cargando separación de fuentes")
            separator,engine=self._build_separator(progress)
            if progress:progress(18,"Separando drums · bass · other · vocals")
            wav,input_sr=self._decode_for_separator(audio_path)
            _mixture,stems=separator.separate_tensor(wav,input_sr)
            sr=int(separator.samplerate); written={}
            for name,tensor in stems.items():
                arr=tensor.detach().float().cpu().numpy()
                if arr.ndim==2:arr=arr.T
                p=root/f"{name}.wav"; sf.write(p,arr,sr,subtype="FLOAT"); written[name]=p
            harmony=written.get("other"); bass=written.get("bass")
            if harmony is None or bass is None: raise RuntimeError("Demucs no devolvió stems other/bass")
            if progress:progress(100,"Separación de fuentes lista")
            return SeparationResult(root,audio_path,harmony,bass,written.get("drums"),written.get("vocals"),engine,sr)
        except Exception as exc:
            if progress:progress(26,f"Demucs no disponible · HPSS nativo ({type(exc).__name__})")
            return self._fallback_hpss(audio_path,root,progress)
    def _fallback_hpss(self,audio_path,root,progress):
        y,sr=load_mono(audio_path,sr=44100)
        if progress:progress(45,"Separación armónico/percusiva · DSP nativo")
        harmonic,percussive=hpss_native(y,sr)
        sos=butter(6,260,btype="lowpass",fs=sr,output="sos"); bass=sosfiltfilt(sos,y).astype(np.float32) if len(y)>64 else y.copy()
        hp=root/"harmony_hpss.wav"; bp=root/"bass_lowpass.wav"; dp=root/"drums_hpss.wav"
        sf.write(hp,harmonic,sr,subtype="FLOAT"); sf.write(bp,bass,sr,subtype="FLOAT"); sf.write(dp,percussive,sr,subtype="FLOAT")
        if progress:progress(100,"HPSS nativo listo")
        return SeparationResult(root,audio_path,hp,bp,dp,None,"ChordScope SciPy HPSS fallback",sr)
