from __future__ import annotations

from collections import Counter
import gc
from pathlib import Path
from typing import Callable

import numpy as np

from .audio import load_mono
from .control import check_cancel
from .dsp import estimate_tempo_native
from .resources import models_root

Progress = Callable[[int, str], None]


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _estimate_meter(beats: np.ndarray, downbeats: np.ndarray) -> int:
    if len(beats) < 4 or len(downbeats) < 2:
        return 4
    counts=[]
    for a,b in zip(downbeats[:-1],downbeats[1:]):
        n=int(np.sum((beats>=a-.08)&(beats<b-.08)))
        if 2<=n<=12: counts.append(n)
    if not counts: return 4
    c=Counter(counts).most_common(1)[0][0]
    return c if c in {3,4,5,6,7,8,9,12} else 4


def _align_grid(beats: np.ndarray, downbeats: np.ndarray, duration: float, meter: int) -> np.ndarray:
    beats=np.asarray(sorted(float(x) for x in beats if 0<=x<=duration),dtype=float)
    if len(beats)<2: return beats
    period=float(np.median(np.diff(beats)))
    if not np.isfinite(period) or period<=0: return beats
    out=beats.tolist()
    while out and out[0]-period>=-.12: out.insert(0,out[0]-period)
    while out and out[-1]+period<=duration+.12: out.append(out[-1]+period)
    out=[max(0.0,x) for x in out if x<=duration]
    grid=np.asarray(out,dtype=float)
    if len(downbeats) and meter>1 and len(grid):
        indices=[]
        for d in downbeats:
            j=int(np.argmin(np.abs(grid-d)))
            if abs(grid[j]-d)<period*.38: indices.append(j)
        if indices:
            phase=Counter([j%meter for j in indices]).most_common(1)[0][0]
            if phase:
                first=grid[0]; prepend=[]
                for k in range(phase,0,-1):
                    t=first-k*period
                    if t>=-.08: prepend.append(max(0.0,t))
                grid=np.asarray(prepend+grid.tolist()) if len(prepend)==phase else grid[phase:]
    dedup=[]
    for x in grid:
        if not dedup or x-dedup[-1]>period*.22: dedup.append(float(x))
    return np.asarray(dedup,dtype=float)


def _release_torch_cache() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception:
        pass


def detect_beats(audio_path: str|Path,duration: float,progress: Progress|None=None,*,prefer_model: bool=True,cancel_event=None) -> dict:
    device=_device(); check_cancel(cancel_event)
    if not prefer_model:
        if progress: progress(10,"Modo rápido · tempo DSP nativo")
        y,sr=load_mono(audio_path,sr=22050); check_cancel(cancel_event)
        bpm,beats,conf=estimate_tempo_native(y,sr)
        grid=_align_grid(np.asarray(beats),np.asarray([]),duration,4); period=60.0/max(1e-6,bpm)
        if progress: progress(100,f"DSP tempo · {bpm:.1f} BPM")
        return {"beats":grid.tolist(),"downbeats":[],"bpm":round(float(bpm),2),"period":period,"meter":4,"confidence":float(conf),"engine":"ChordScope native tempo"}

    tracker=None
    try:
        if progress: progress(5,f"Beat This! · cargando modelo · {device.upper()}")
        from beat_this.inference import File2Beats
        local_ckpt=models_root()/"beat_this"/"beat_this-final0.ckpt"
        checkpoint=str(local_ckpt) if local_ckpt.exists() else "final0"
        tracker=File2Beats(checkpoint_path=checkpoint,device=device,dbn=False,float16=device.startswith("cuda"))
        check_cancel(cancel_event)
        if progress: progress(30,"Beat This! · inferencia")
        beats,downbeats=tracker(str(audio_path)); check_cancel(cancel_event)
        beats=np.asarray(beats,dtype=float); downbeats=np.asarray(downbeats,dtype=float)
        if len(beats)<2: raise RuntimeError("Beat This devolvió menos de dos beats")
        period=float(np.median(np.diff(beats))); bpm=60.0/period; meter=_estimate_meter(beats,downbeats)
        grid=_align_grid(beats,downbeats,duration,meter)
        if progress: progress(100,f"Beat This! · {bpm:.1f} BPM · {meter}/4")
        return {"beats":grid.tolist(),"downbeats":downbeats.tolist(),"bpm":round(bpm,2),"period":period,"meter":meter,"confidence":.92,"engine":"Beat This! final0"}
    except Exception as exc:
        check_cancel(cancel_event)
        if progress: progress(35,f"Beat This! fallback · DSP nativo ({type(exc).__name__})")
        y,sr=load_mono(audio_path,sr=22050); bpm,beats,conf=estimate_tempo_native(y,sr)
        grid=_align_grid(np.asarray(beats),np.asarray([]),duration,4); period=60.0/max(1e-6,bpm)
        if progress: progress(100,f"DSP tempo · {bpm:.1f} BPM")
        return {"beats":grid.tolist(),"downbeats":[],"bpm":round(float(bpm),2),"period":period,"meter":4,"confidence":float(conf),"engine":"ChordScope native tempo fallback"}
    finally:
        tracker=None
        _release_torch_cache()
