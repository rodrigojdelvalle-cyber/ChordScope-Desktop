from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .audio import load_mono
from .dsp import frame_spectral_features
from .music import QUALITY_COMPLEXITY, QUALITY_INTERVALS
from .types import BeatFeature, ChordCandidate

Progress = Callable[[int, str], None]

TEMPLATE_QUALITIES=["maj","min","7","maj7","m7","6","m6","sus2","sus4","dim","m7b5","aug","add9","9","maj9","m9","11","m11","13","m13"]


def _templates():
    out=[]
    for root in range(12):
        for q in TEMPLATE_QUALITIES:
            v=np.full(12,-0.08,dtype=np.float64)
            for interval in QUALITY_INTERVALS[q]:
                pc=(root+interval)%12
                if interval==0:w=1.25
                elif interval in {3,4}:w=1.12
                elif interval in {10,11}:w=1.02
                elif interval==7:w=.88
                else:w=.72
                v[pc]=max(v[pc],w)
            v/=np.linalg.norm(v)+1e-12
            out.append((root,q,v))
    return out

TEMPLATES=_templates()


def extract_beat_features(harmony_path: str | Path, beat_times: list[float], duration: float, progress: Progress | None = None) -> list[BeatFeature]:
    if progress: progress(5,"STFT/Chroma armónico · DSP nativo")
    y,sr=load_mono(harmony_path,sr=22050)
    chroma,rms,flat,times=frame_spectral_features(y,sr,hop=256,n_fft=4096,fmin=32.0,fmax=5000.0)
    activity_raw=rms*(1.0-np.clip(flat,0,1))
    ref=max(float(np.percentile(activity_raw,72)) if activity_raw.size else 0.0,1e-5)
    out=[]
    for i,st in enumerate(beat_times):
        en=beat_times[i+1] if i+1<len(beat_times) else duration
        mask=(times>=st)&(times<en)
        if not np.any(mask):
            c=np.zeros(12); rr=ff=act=0.0
        else:
            c=np.mean(chroma[:,mask],axis=1).astype(float)
            c=np.maximum(c-np.median(c)*0.24,0)
            c/=np.linalg.norm(c)+1e-12
            rr=float(np.mean(rms[mask])); ff=float(np.mean(flat[mask])); act=float(np.mean(activity_raw[mask])/ref)
        best=None; best_score=-1e9
        for root,q,t in TEMPLATES:
            score=float(np.dot(c,t))-0.014*QUALITY_COMPLEXITY.get(q,3)
            if score>best_score:
                best_score=score; best=ChordCandidate(root=root,quality=q,score=score,engine="STFT-template")
        out.append(BeatFeature(time=float(st),end=float(en),chroma=[float(x) for x in c],activity=act,rms=rr,flatness=ff,candidate=best))
        if progress and i % max(1,len(beat_times)//8)==0:
            progress(18+int(78*i/max(1,len(beat_times))),"STFT/Chroma por pulso")
    if progress: progress(100,"Características armónicas listas")
    return out
