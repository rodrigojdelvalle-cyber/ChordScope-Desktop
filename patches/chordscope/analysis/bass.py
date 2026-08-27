from __future__ import annotations

from pathlib import Path
from typing import Callable

from .audio import load_mono
from .dsp import bass_pitch_class_for_segment

Progress = Callable[[int, str], None]


def detect_bass_notes(bass_path: str | Path, beat_times: list[float], duration: float, progress: Progress | None = None) -> list[tuple[int | None, float]]:
    if progress:
        progress(5,"Analizando línea de bajo · DSP nativo")
    y,sr=load_mono(bass_path,sr=22050)
    out=[]
    for i,st in enumerate(beat_times):
        en=beat_times[i+1] if i+1<len(beat_times) else duration
        span_end=st+(en-st)*0.82
        a=max(0,int(st*sr)); b=min(len(y),max(a+1,int(span_end*sr)))
        out.append(bass_pitch_class_for_segment(y[a:b],sr))
        if progress and i % max(1,len(beat_times)//6)==0:
            progress(15+int(80*i/max(1,len(beat_times))),"Walking bass por pulso")
    if progress:
        progress(100,"Línea de bajo lista")
    return out
