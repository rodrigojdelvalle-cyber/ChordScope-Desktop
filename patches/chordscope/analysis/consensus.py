from __future__ import annotations

from collections import defaultdict

import numpy as np

from .music import (
    ParsedChord, QUALITY_COMPLEXITY, chord_pcs, key_prior, parse_chord_label,
    quality_family, transition_score,
)
from .types import BeatCell, BeatFeature, ChordCandidate, ChordSegment

ENGINE_WEIGHTS = {
    "LV-Chordia-full": 1.32,
    "LV-Chordia-submission": 1.04,
    "BTC Transformer": 1.24,
    "CQT-template": 0.58,
    "STFT-template": 0.62,
}


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _segment_vote(segments: list[ChordSegment], st: float, en: float) -> tuple[ParsedChord, float, str] | None:
    best = None
    best_ov = 0.0
    for seg in segments:
        if seg.end <= st:
            continue
        if seg.start >= en:
            break
        ov = _overlap(st, en, seg.start, seg.end)
        if ov > best_ov:
            best_ov = ov
            best = seg
    if best is None or best_ov <= 0:
        return None
    ratio = best_ov / max(1e-6, en - st)
    return parse_chord_label(best.label), ratio * best.confidence, best.engine


def _pc_support(chroma: list[float], root: int, quality: str) -> float:
    if not chroma:
        return 0.0
    pcs = chord_pcs(root, quality)
    total = sum(max(0.0, x) for x in chroma) + 1e-12
    inside = sum(max(0.0, chroma[p]) for p in pcs)
    return inside / total


def _candidate_key(root: int, quality: str) -> tuple[int, str]:
    return root % 12, quality


def collect_candidates(i: int, feature: BeatFeature, engine_segments: dict[str, list[ChordSegment]], key_root: int, key_mode: str) -> tuple[list[ChordCandidate], dict[str, str]]:
    score_by: dict[tuple[int, str], float] = defaultdict(float)
    engine_by: dict[tuple[int, str], set[str]] = defaultdict(set)
    source_votes: dict[str, str] = {}
    no_score = 0.0
    st, en = feature.time, feature.end

    for engine_name, segs in engine_segments.items():
        vote = _segment_vote(segs, st, en)
        if not vote:
            continue
        parsed, strength, actual_engine = vote
        weight = ENGINE_WEIGHTS.get(actual_engine, ENGINE_WEIGHTS.get(engine_name, 1.0))
        source_votes[engine_name] = parsed.raw or "N"
        if parsed.no_chord:
            no_score += weight * max(0.35, strength)
            continue
        key = _candidate_key(parsed.root, parsed.quality)
        score_by[key] += weight * (0.55 + 0.45 * min(1.0, strength))
        engine_by[key].add(engine_name)

    if feature.candidate is not None:
        c = feature.candidate
        score_by[(c.root, c.quality)] += ENGINE_WEIGHTS.get(c.engine, ENGINE_WEIGHTS["CQT-template"]) * max(0.0, min(1.0, c.score))
        engine_by[(c.root, c.quality)].add("template")
        source_votes["template"] = f"{c.root}:{c.quality}"

    root_groups: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for (root, q), score in score_by.items():
        root_groups[root].append((q, score))
    for root, items in root_groups.items():
        if len(items) >= 2:
            total = sum(s for _, s in items)
            fam_scores = defaultdict(float)
            for q, s in items:
                fam_scores[quality_family(q)] += s
            fam, fam_score = max(fam_scores.items(), key=lambda x: x[1])
            if fam in {"major", "minor", "dominant"}:
                q0 = "maj" if fam == "major" else "min" if fam == "minor" else "7"
                score_by[(root, q0)] += total * 0.16

    candidates = []
    for (root, q), score in score_by.items():
        support = _pc_support(feature.chroma, root, q)
        score += support * 0.46
        score += key_prior(root, q, key_root, key_mode) * 0.42
        score -= QUALITY_COMPLEXITY.get(q, 3) * 0.035
        exact_engines = len(engine_by[(root, q)])
        if exact_engines >= 2:
            score += 0.34 + 0.14 * (exact_engines - 2)
        candidates.append(ChordCandidate(root, q, score, "+".join(sorted(engine_by[(root, q)])) or "consensus"))

    activity = feature.activity
    flatness = feature.flatness
    if activity < 0.12:
        no_score += 2.4
    elif activity < 0.22:
        no_score += 1.45
    elif activity < 0.34:
        no_score += 0.65
    if flatness > 0.54:
        no_score += 0.65
    if no_score > 0:
        candidates.append(ChordCandidate(0, "maj", no_score, "N.C.", no_chord=True))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:10], source_votes


def _as_parsed(c: ChordCandidate) -> ParsedChord:
    return ParsedChord(c.no_chord, c.root, c.quality)


def _transition(a: ChordCandidate, b: ChordCandidate, key_root: int, key_mode: str) -> float:
    if a.no_chord and b.no_chord:
        return 0.36
    if a.no_chord or b.no_chord:
        return -0.13
    return transition_score(_as_parsed(a), _as_parsed(b), key_root, key_mode)


def decode_sequence(features: list[BeatFeature], engine_segments: dict[str, list[ChordSegment]], key_root: int, key_mode: str) -> tuple[list[BeatCell], list[dict[str, str]]]:
    lattice = []
    votes = []
    for i, f in enumerate(features):
        cand, src = collect_candidates(i, f, engine_segments, key_root, key_mode)
        if not cand:
            cand = [ChordCandidate(0, "maj", 1.0, "N.C.", no_chord=True)]
        lattice.append(cand)
        votes.append(src)

    dp = []
    back = []
    for i, cand_list in enumerate(lattice):
        dp.append(np.full(len(cand_list), -1e18, dtype=float))
        back.append(np.full(len(cand_list), -1, dtype=int))
        for j, c in enumerate(cand_list):
            emission = c.score
            if i == 0:
                dp[i][j] = emission
                continue
            for k, p in enumerate(lattice[i - 1]):
                s = dp[i - 1][k] + emission + _transition(p, c, key_root, key_mode)
                if s > dp[i][j]:
                    dp[i][j] = s
                    back[i][j] = k

    j = int(np.argmax(dp[-1])) if dp else 0
    path: list[ChordCandidate] = [lattice[-1][j]] if lattice else []
    for i in range(len(lattice) - 1, 0, -1):
        j = int(back[i][j]) if back[i][j] >= 0 else 0
        path.append(lattice[i - 1][j])
    path.reverse()

    cells = []
    for i, c in enumerate(path):
        sorted_scores = sorted((x.score for x in lattice[i]), reverse=True)
        gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 1.0
        conf = max(0.18, min(0.99, 0.54 + gap * 0.17))
        cells.append(BeatCell(time=features[i].time, root=c.root, quality=c.quality, confidence=conf, noChord=c.no_chord, sourceVotes=votes[i], consensus=c.engine))
    return cells, votes


def attach_bass(cells: list[BeatCell], bass: list[tuple[int | None, float]]) -> None:
    for cell, (pc, conf) in zip(cells, bass):
        if cell.noChord and conf < 0.84:
            cell.bassNote = None
            cell.bassNoteConfidence = 0.0
        else:
            cell.bassNote = pc
            cell.bassNoteConfidence = float(conf)
