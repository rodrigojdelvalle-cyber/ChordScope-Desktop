from __future__ import annotations

from pathlib import Path

import numpy as np

from .audio import load_mono
from .dsp import frame_spectral_features
from .music import expected_family, quality_family
from .types import BeatCell, KeyEstimate

MAJOR_PROFILE = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88], dtype=float)
MINOR_PROFILE = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17], dtype=float)


def global_chroma(harmony_path: str | Path) -> np.ndarray:
    y, sr = load_mono(harmony_path, sr=22050)
    c, rms, flat, _times = frame_spectral_features(y, sr, hop=512, n_fft=4096, fmin=32.0, fmax=5000.0)
    if c.size == 0:
        return np.ones(12) / 12
    activity = rms * (1.0 - np.clip(flat, 0, 1))
    if activity.size:
        threshold = np.percentile(activity, 35)
        keep = activity >= threshold
        if np.any(keep):
            c = c[:, keep]
    v = np.median(c, axis=1) if c.shape[1] else np.ones(12)
    v = np.maximum(v, 0)
    return v / (np.linalg.norm(v) + 1e-12)


def _corr(chroma: np.ndarray, profile: np.ndarray, root: int) -> float:
    p = np.roll(profile, root)
    a = chroma - chroma.mean()
    b = p - p.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > 1e-12 else 0.0


def chroma_key_candidates(chroma: np.ndarray) -> list[dict]:
    out = []
    for root in range(12):
        out.append({"root": root, "mode": "major", "profile": _corr(chroma, MAJOR_PROFILE, root)})
        out.append({"root": root, "mode": "minor", "profile": _corr(chroma, MINOR_PROFILE, root)})
    return sorted(out, key=lambda x: x["profile"], reverse=True)


def _segments(beats: list[BeatCell]) -> list[dict]:
    out = []
    cur = None
    for i, b in enumerate(beats):
        if b.noChord:
            cur = None
            continue
        fam = quality_family(b.quality)
        if cur and cur["root"] == b.root and cur["family"] == fam and cur["end"] == i - 1:
            cur["end"] = i
            cur["length"] += 1
        else:
            cur = {"root": b.root, "quality": b.quality, "family": fam, "start": i, "end": i, "length": 1}
            out.append(cur)
    return out


def progression_score(beats: list[BeatCell], root: int, mode: str) -> float:
    segs = _segments(beats)
    if not segs:
        return 0.0
    score = 0.0
    tonic_family = "major" if mode == "major" else "minor"
    total = 0.0
    for s in segs:
        w = min(4, s["length"])
        total += w
        rel = (s["root"] - root) % 12
        if s["family"] in expected_family(mode, rel):
            score += 1.0 * w
        if rel == 0:
            score += (1.25 if s["family"] == tonic_family else -0.55) * w
        if s["family"] == "dominant":
            target_rel = ((s["root"] + 5) - root) % 12
            diatonic = {0,2,4,5,7,9,11} if mode == "major" else {0,2,3,5,7,8,10,11}
            if target_rel in diatonic:
                score += 0.65 * w
    if segs[0]["root"] == root and segs[0]["family"] == tonic_family:
        score += 5.0
    if segs[-1]["root"] == root and segs[-1]["family"] == tonic_family:
        score += 6.5
    for a, b in zip(segs[:-1], segs[1:]):
        ar = (a["root"] - root) % 12
        br = (b["root"] - root) % 12
        if ar == 7 and br == 0 and a["family"] == "dominant" and b["family"] == tonic_family:
            score += 4.6
        if ar == 2 and br == 7 and a["family"] in ({"minor"} if mode == "major" else {"diminished"}) and b["family"] == "dominant":
            score += 1.8
    for a, b, c in zip(segs[:-2], segs[1:-1], segs[2:]):
        ar, br, cr = (a["root"]-root)%12, (b["root"]-root)%12, (c["root"]-root)%12
        ii_ok = a["family"] in ({"minor"} if mode == "major" else {"diminished"})
        if ar == 2 and br == 7 and cr == 0 and ii_ok and b["family"] == "dominant" and c["family"] == tonic_family:
            score += 5.5
    return score / max(1.0, total * 0.55)


def estimate_key(chroma: np.ndarray, beats: list[BeatCell] | None = None) -> KeyEstimate:
    candidates = chroma_key_candidates(chroma)
    ranked = []
    for c in candidates:
        prog = progression_score(beats or [], c["root"], c["mode"]) if beats else 0.0
        score = 2.7 * c["profile"] + 1.45 * prog
        ranked.append({**c, "progression": prog, "score": score})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else best
    gap = max(0.0, best["score"] - second["score"])
    conf = max(0.38, min(0.98, 0.52 + gap * 0.18 + max(0, best["progression"]) * 0.025))
    scale = "ionian" if best["mode"] == "major" else "aeolian"
    return KeyEstimate(
        root=int(best["root"]), mode=str(best["mode"]), scale_type=scale,
        confidence=conf,
        alternatives=[
            {"root": int(x["root"]), "mode": x["mode"], "score": round(float(x["score"]), 4)}
            for x in ranked[:4]
        ],
        engine="Krumhansl + cadencias + dominantes secundarios",
    )
