from chordscope.analysis.pipeline import PROFILES, normalize_profile, _should_run_btc


def test_balanced_is_default_and_btc_is_adaptive():
    p = normalize_profile("does-not-exist")
    assert p is PROFILES["balanced"]
    assert p.btc_mode == "adaptive"


def test_adaptive_btc_skips_strong_consensus_and_runs_on_weak_consensus():
    p = PROFILES["balanced"]
    run, _ = _should_run_btc(p, {"confidence": .84, "no_chord_ratio": .05}, {"lv_submission": [1, 2]})
    assert not run
    run, _ = _should_run_btc(p, {"confidence": .60, "no_chord_ratio": .05}, {"lv_submission": [1, 2]})
    assert run


def test_fast_never_runs_btc_and_deep_always_does():
    strong = {"confidence": .95, "no_chord_ratio": 0.0}
    assert not _should_run_btc(PROFILES["fast"], strong, {"lv_submission": [1]})[0]
    assert _should_run_btc(PROFILES["deep"], strong, {"lv_submission": [1]})[0]
