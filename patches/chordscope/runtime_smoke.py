from __future__ import annotations
import json, os, tempfile, wave
from pathlib import Path
import numpy as np


def _write_fixture(path: Path, sr: int = 22050, duration: float = 6.0) -> None:
    n=int(sr*duration); t=np.arange(n,dtype=np.float64)/sr
    y=(0.10*np.sin(2*np.pi*261.6256*t)+0.08*np.sin(2*np.pi*329.6276*t)+0.07*np.sin(2*np.pi*391.9954*t)+0.05*np.sin(2*np.pi*440.0*t))
    for bt in np.arange(0,duration,0.5):
        i=int(bt*sr); m=min(n-i,int(.018*sr))
        if m>0:y[i:i+m]+=0.45*np.hanning(m)
    y=np.clip(y,-.95,.95)
    pcm=(y*32767).astype('<i2')
    with wave.open(str(path),'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(sr);w.writeframes(pcm.tobytes())


def run_runtime_smoke() -> int:
    from .analysis.audio import probe_audio,load_mono
    from .analysis.template_engine import extract_beat_features
    from .analysis.key import global_chroma
    from .analysis.bass import detect_bass_notes
    from .analysis.dsp import hpss_native,estimate_tempo_native
    from .analysis.librosa_compat import patch_third_party_librosa_loader
    with tempfile.TemporaryDirectory(prefix='chordscope_smoke_') as td:
        p=Path(td)/'fixture.wav'; _write_fixture(p)
        meta=probe_audio(p,n_peaks=80)
        assert meta['duration']>5.5 and meta['sample_rate']==22050 and len(meta['waveform'])>10
        y,sr=load_mono(p,22050)
        h,perc=hpss_native(y,sr)
        import soundfile as sf
        hp=Path(td)/'harmony.wav'; bp=Path(td)/'bass.wav'; sf.write(hp,h,sr); sf.write(bp,y,sr)
        bpm,beats,conf=estimate_tempo_native(y,sr)
        if len(beats)<3: beats=np.arange(0,meta['duration'],0.5)
        bt=[float(x) for x in beats if x<meta['duration']]
        feats=extract_beat_features(hp,bt,meta['duration'])
        chroma=global_chroma(hp)
        bass=detect_bass_notes(bp,bt,meta['duration'])
        assert len(feats)==len(bt) and chroma.shape==(12,) and len(bass)==len(bt)

        patch_third_party_librosa_loader()
        import librosa
        ly,lsr=librosa.load(str(p),sr=22050,mono=True)
        assert len(ly)>10000 and lsr==22050
        from librosa.core.constantq import hybrid_cqt
        cqt=hybrid_cqt(ly[:sr*2],sr=lsr,hop_length=512,n_bins=36,bins_per_octave=12,tuning=0.0)
        assert cqt.shape[0]==36 and cqt.shape[1]>5

        try:
            import av
            av_status=f"ok:{getattr(av, '__version__', 'unknown')}"
        except Exception as exc:
            raise RuntimeError(f'PyAV packaged import failed: {exc}') from exc

        payload={'status':'CHORDSCOPE_RUNTIME_SMOKE_OK','bpm':round(float(bpm),2),'beats':len(bt),'features':len(feats),'decoder':'soundfile/native-dsp','librosa_external_cqt':'ok','pyav':av_status}
        text=json.dumps(payload,ensure_ascii=False)
        marker=os.environ.get('CHORDSCOPE_SMOKE_MARKER')
        if marker:
            Path(marker).write_text(text,encoding='utf-8')
        print(text)
    return 0
