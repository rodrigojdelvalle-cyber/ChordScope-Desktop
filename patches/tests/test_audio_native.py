import wave
from pathlib import Path
import numpy as np
from chordscope.analysis.audio import probe_audio, load_mono
from chordscope.analysis.dsp import frame_spectral_features, estimate_tempo_native


def _wav(path: Path, sr=22050, sec=2.0):
    t=np.arange(int(sr*sec))/sr
    y=.25*np.sin(2*np.pi*440*t)
    for x in np.arange(0,sec,.5):
        i=int(x*sr); m=min(int(.015*sr),len(y)-i)
        if m>0:y[i:i+m]+=.5*np.hanning(m)
    pcm=np.clip(y,-.99,.99); pcm=(pcm*32767).astype('<i2')
    with wave.open(str(path),'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(sr);w.writeframes(pcm.tobytes())


def test_native_audio_loader_and_features(tmp_path):
    p=tmp_path/'x.wav'; _wav(p)
    y,sr=load_mono(p,22050)
    assert sr==22050 and len(y)>40000
    meta=probe_audio(p,n_peaks=20)
    assert meta['duration']>1.9 and len(meta['waveform'])==20
    c,r,f,t=frame_spectral_features(y,sr,hop=256,n_fft=2048)
    assert c.shape[0]==12 and c.shape[1]==len(r)==len(f)==len(t)
    bpm,beats,conf=estimate_tempo_native(y,sr)
    assert 50 < bpm < 240 and len(beats)>=2
