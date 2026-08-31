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
from .control import check_cancel
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
    _LOCAL_FILES = (
        "htdemucs_ft.yaml",
        "f7e0c4bc-ba3fe64a.th",
        "d12395a8-e57c48e6.th",
        "92cfc3b6-ef3bcb9c.th",
        "04573f0d-f3cf25b2.th",
    )

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._separator = None
        self._tempdirs: list[tempfile.TemporaryDirectory] = []

    def release_model(self) -> None:
        """Free the heavy Demucs model without deleting generated stems."""
        self._separator = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def close(self) -> None:
        self.release_model()
        while self._tempdirs:
            try:
                self._tempdirs.pop().cleanup()
            except Exception:
                pass

    def _temp_root(self) -> Path:
        td = tempfile.TemporaryDirectory(prefix="chordscope_sep_")
        self._tempdirs.append(td)
        return Path(td.name)

    @classmethod
    def _local_repo_ready(cls, path: Path) -> bool:
        return path.is_dir() and all((path / n).is_file() for n in cls._LOCAL_FILES)

    def _build_separator(self, progress: Progress | None, shifts: int, cancel_event=None):
        check_cancel(cancel_event)
        from demucs_infer.api import Separator
        local_repo = models_root() / "demucs"
        kwargs = dict(model=self._MODEL, device=self.device, shifts=max(0,int(shifts)), overlap=.25, split=True, progress=False)
        if self._local_repo_ready(local_repo):
            kwargs["repo"] = local_repo
            engine = f"Demucs HTDemucs-FT · offline · shifts={max(0,int(shifts))}"
        else:
            engine = f"Demucs HTDemucs-FT · runtime model · shifts={max(0,int(shifts))}"
        if progress:
            progress(6, "Cargando HTDemucs-FT")
        def callback(info):
            check_cancel(cancel_event)
            if progress is None:
                return
            try:
                length=max(1,int(info.get("audio_length",1))); offset=max(0,int(info.get("segment_offset",0)))
                mi=max(0,int(info.get("model_idx_in_bag",0))); models=max(1,int(info.get("models",1)))
                frac=(mi+min(1.0,offset/length))/models
                progress(18+int(72*frac), "Separando drums · bass · other · vocals")
            except Exception:
                pass
        kwargs["callback"] = callback
        self._separator = Separator(**kwargs)
        return self._separator, engine

    @staticmethod
    def _decode_for_separator(audio_path):
        import torch
        y,sr=load_mono(audio_path,sr=None)
        return torch.from_numpy(np.ascontiguousarray(y[None,:])), int(sr)

    def separate(self,audio_path,progress: Progress|None=None,*,use_demucs:bool=True,shifts:int=0,cancel_event=None) -> SeparationResult:
        audio_path=Path(audio_path); root=self._temp_root(); check_cancel(cancel_event)
        if not use_demucs:
            if progress: progress(8,"Modo rápido · HPSS nativo")
            return self._fallback_hpss(audio_path,root,progress,cancel_event)
        try:
            if progress: progress(3,"Cargando separación de fuentes")
            separator,engine=self._build_separator(progress,shifts,cancel_event)
            check_cancel(cancel_event)
            if progress: progress(18,"Separando drums · bass · other · vocals")
            wav,input_sr=self._decode_for_separator(audio_path)
            _mixture,stems=separator.separate_tensor(wav,input_sr)
            check_cancel(cancel_event)
            sr=int(separator.samplerate); written={}
            for name,tensor in stems.items():
                check_cancel(cancel_event)
                arr=tensor.detach().float().cpu().numpy()
                if arr.ndim==2: arr=arr.T
                p=root/f"{name}.wav"; sf.write(p,arr,sr,subtype="FLOAT"); written[name]=p
            harmony=written.get("other"); bass=written.get("bass")
            if harmony is None or bass is None: raise RuntimeError("Demucs no devolvió stems other/bass")
            if progress: progress(96,"Stems listos · liberando modelo Demucs")
            self.release_model()
            if progress: progress(100,"Separación lista · memoria Demucs liberada")
            return SeparationResult(root,audio_path,harmony,bass,written.get("drums"),written.get("vocals"),engine,sr)
        except Exception as exc:
            self.release_model(); check_cancel(cancel_event)
            if progress: progress(26,f"Demucs no disponible · HPSS nativo ({type(exc).__name__})")
            return self._fallback_hpss(audio_path,root,progress,cancel_event)

    def _fallback_hpss(self,audio_path,root,progress=None,cancel_event=None):
        check_cancel(cancel_event)
        y,sr=load_mono(audio_path,sr=44100)
        if progress: progress(45,"Separación armónico/percusiva · DSP nativo")
        harmonic,percussive=hpss_native(y,sr); check_cancel(cancel_event)
        sos=butter(6,260,btype="lowpass",fs=sr,output="sos")
        bass=sosfiltfilt(sos,y).astype(np.float32) if len(y)>64 else y.copy()
        hp=root/"harmony_hpss.wav"; bp=root/"bass_lowpass.wav"; dp=root/"drums_hpss.wav"
        sf.write(hp,harmonic,sr,subtype="FLOAT"); sf.write(bp,bass,sr,subtype="FLOAT"); sf.write(dp,percussive,sr,subtype="FLOAT")
        if progress: progress(100,"HPSS nativo listo")
        return SeparationResult(root,Path(audio_path),hp,bp,dp,None,"ChordScope SciPy HPSS fallback",sr)
