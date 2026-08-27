from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import median_filter


def _to_mono(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 1:
        return x
    if x.ndim != 2:
        return np.ravel(x).astype(np.float32)
    return np.mean(x, axis=1, dtype=np.float32)


def load_audio(path: str | Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Decode audio without going through librosa's lazy-loaded audio module."""
    p = str(path)
    try:
        data, native_sr = sf.read(p, dtype="float32", always_2d=True)
        y = _to_mono(data)
        native_sr = int(native_sr)
    except Exception as sf_exc:
        try:
            import av  # type: ignore
            container = av.open(p)
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise RuntimeError("el archivo no contiene una pista de audio")
            target_rate = int(sr or stream.codec_context.sample_rate or 44100)
            resampler = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=target_rate)
            chunks: list[np.ndarray] = []
            for frame in container.decode(stream):
                converted = resampler.resample(frame)
                if converted is None:
                    continue
                if not isinstance(converted, (list, tuple)):
                    converted = [converted]
                for out_frame in converted:
                    a = np.asarray(out_frame.to_ndarray(), dtype=np.float32)
                    chunks.append(np.ravel(a))
            container.close()
            if not chunks:
                raise RuntimeError("no se pudieron decodificar muestras de audio")
            y = np.concatenate(chunks).astype(np.float32, copy=False)
            native_sr = target_rate
        except Exception as av_exc:
            raise RuntimeError(
                f"No se pudo decodificar el audio con libsndfile ni PyAV. "
                f"Formato/codec posiblemente no soportado. soundfile={sf_exc}; PyAV={av_exc}"
            ) from av_exc

    if sr is not None and native_sr != int(sr) and y.size:
        target = int(sr)
        g = math.gcd(native_sr, target)
        y = signal.resample_poly(y, target // g, native_sr // g).astype(np.float32, copy=False)
        native_sr = target

    y = np.ascontiguousarray(y, dtype=np.float32)
    if y.size:
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y, native_sr


def frame_spectral_features(
    y: np.ndarray,
    sr: int,
    *,
    hop: int = 256,
    n_fft: int = 4096,
    fmin: float = 27.5,
    fmax: float = 5000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return chroma[12,T], RMS[T], flatness[T], times[T] using NumPy/SciPy."""
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return np.zeros((12, 1), np.float32), np.zeros(1), np.ones(1), np.zeros(1)

    n_fft = int(max(512, n_fft))
    hop = int(max(64, hop))
    noverlap = max(0, n_fft - hop)
    freqs, times, z = signal.stft(
        y,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=noverlap,
        nfft=n_fft,
        boundary="zeros",
        padded=True,
    )
    mag = np.abs(z).astype(np.float64)
    power = mag * mag
    rms = np.sqrt(np.mean(power, axis=0) + 1e-14)
    flat = np.exp(np.mean(np.log(mag + 1e-10), axis=0)) / (np.mean(mag + 1e-10, axis=0) + 1e-10)
    flat = np.clip(flat, 0.0, 1.0)

    valid = (freqs >= fmin) & (freqs <= min(fmax, sr * 0.49))
    vf = freqs[valid]
    vm = mag[valid]
    chroma = np.zeros((12, mag.shape[1]), dtype=np.float64)
    if vf.size:
        midi = 69.0 + 12.0 * np.log2(vf / 440.0)
        pcs = np.mod(np.rint(midi).astype(int), 12)
        weights = np.sqrt(vm + 1e-12) / np.sqrt(np.maximum(vf[:, None], 55.0) / 55.0)
        for pc in range(12):
            mask = pcs == pc
            if np.any(mask):
                chroma[pc] = np.sum(weights[mask], axis=0)
    norm = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-12
    chroma = chroma / norm
    return chroma.astype(np.float32), rms.astype(np.float32), flat.astype(np.float32), np.asarray(times, dtype=np.float64)


def hpss_native(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Median-filter HPSS implemented with SciPy only."""
    y = np.asarray(y, dtype=np.float32)
    if y.size < 32:
        return y.copy(), np.zeros_like(y)
    n_fft = 2048 if y.size >= 2048 else max(256, 2 ** int(np.floor(np.log2(max(32, y.size)))))
    hop = max(64, n_fft // 4)
    _freqs, _times, z = signal.stft(
        y, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft-hop,
        nfft=n_fft, boundary="zeros", padded=True,
    )
    mag = np.abs(z)
    harm = median_filter(mag, size=(1, 31), mode="nearest")
    perc = median_filter(mag, size=(31, 1), mode="nearest")
    eps = 1e-12
    h2 = harm * harm
    p2 = perc * perc
    mh = h2 / (h2 + (2.0 * perc) ** 2 + eps)
    mp = p2 / (p2 + (1.2 * harm) ** 2 + eps)
    _, yh = signal.istft(z * mh, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft-hop, nfft=n_fft, input_onesided=True)
    _, yp = signal.istft(z * mp, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft-hop, nfft=n_fft, input_onesided=True)
    def fit(x):
        x = np.asarray(x[: len(y)], dtype=np.float32)
        if len(x) < len(y):
            x = np.pad(x, (0, len(y) - len(x)))
        return x
    return fit(yh), fit(yp)


def estimate_tempo_native(y: np.ndarray, sr: int) -> tuple[float, np.ndarray, float]:
    """Packaging-safe fallback tempo grid from spectral flux autocorrelation."""
    chroma, rms, _flat, times = frame_spectral_features(y, sr, hop=256, n_fft=1024, fmin=40, fmax=8000)
    er = np.maximum(0.0, np.diff(rms, prepend=rms[:1]))
    cd = np.sum(np.maximum(0.0, np.diff(chroma, axis=1, prepend=chroma[:, :1])), axis=0)
    onset = er / (np.percentile(er, 90) + 1e-9) + 0.35 * cd / (np.percentile(cd, 90) + 1e-9)
    onset = np.nan_to_num(onset)
    if onset.size < 8 or float(np.max(onset)) < 1e-6:
        bpm = 120.0
        period = 0.5
        return bpm, np.arange(0.0, len(y)/sr, period, dtype=float), 0.20
    onset = onset - np.mean(onset)
    ac = signal.correlate(onset, onset, mode="full", method="fft")[len(onset)-1:]
    hop_sec = 256.0 / sr
    lag_lo = max(1, int(round((60.0/220.0)/hop_sec)))
    lag_hi = min(len(ac)-1, int(round((60.0/55.0)/hop_sec)))
    if lag_hi <= lag_lo:
        bpm = 120.0
        period = 0.5
        return bpm, np.arange(0.0, len(y)/sr, period, dtype=float), 0.20
    band = ac[lag_lo:lag_hi+1]
    lag = lag_lo + int(np.argmax(band))
    period = lag * hop_sec
    bpm = 60.0 / max(period, 1e-6)
    while bpm < 70:
        bpm *= 2.0; period /= 2.0
    while bpm > 190:
        bpm /= 2.0; period *= 2.0
    lag = max(1, int(round(period / hop_sec)))
    phase_scores = [float(np.sum(np.maximum(onset[p::lag], 0.0))) for p in range(min(lag, len(onset)))]
    phase = int(np.argmax(phase_scores)) if phase_scores else 0
    start = float(times[min(phase, len(times)-1)]) if len(times) else 0.0
    while start - period >= -0.05:
        start -= period
    start = max(0.0, start)
    duration = len(y) / sr
    beats = np.arange(start, duration + period * 0.25, period, dtype=float)
    peak = float(np.max(band))
    baseline = float(np.median(np.maximum(band, 0))) + 1e-9
    conf = float(np.clip(0.28 + 0.18 * (peak / baseline - 1.0), 0.25, 0.72))
    return float(bpm), beats, conf


def bass_pitch_class_for_segment(y: np.ndarray, sr: int) -> tuple[int | None, float]:
    """Estimate bass pitch class by harmonic summation on a Demucs bass slice."""
    x = np.asarray(y, dtype=np.float32)
    if x.size < max(128, int(sr * 0.04)):
        return None, 0.0
    x = x - float(np.mean(x))
    rms = float(np.sqrt(np.mean(x*x) + 1e-12))
    if rms < 2e-5:
        return None, 0.0
    n = int(2 ** np.ceil(np.log2(max(1024, len(x)))))
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=n))
    freqs = np.fft.rfftfreq(n, 1.0/sr)
    lo, hi = 38.0, 330.0
    scores = np.zeros(12, dtype=float)
    for midi in range(28, 65):
        f0 = 440.0 * (2.0 ** ((midi - 69) / 12.0))
        if not (lo <= f0 <= hi):
            continue
        s = 0.0
        for h, w in ((1, 1.0), (2, 0.55), (3, 0.32), (4, 0.18)):
            f = f0 * h
            if f >= sr * 0.49:
                break
            j = int(np.argmin(np.abs(freqs - f)))
            a = float(np.max(spec[max(0,j-1): min(len(spec), j+2)]))
            s += w * a
        s /= 1.0 + 0.012 * max(0, midi - 40)
        scores[midi % 12] += s
    if not np.any(scores > 0):
        return None, 0.0
    order = np.argsort(scores)[::-1]
    top = float(scores[order[0]])
    second = float(scores[order[1]]) + 1e-12
    share = top / (float(np.sum(scores)) + 1e-12)
    ratio = top / second
    conf = float(np.clip(0.42 * min(1.0, share * 4.0) + 0.58 * min(1.0, (ratio - 1.0) / 1.7), 0.0, 1.0))
    if conf < 0.30:
        return None, conf
    return int(order[0]), conf
