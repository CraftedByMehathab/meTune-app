import numpy as np
from metune_pipeline.config import PipelineConfig
from metune_pipeline.dsp import dsp_features
from metune_pipeline.fuse import fuse
from metune_pipeline.rules import baseline, rule_fields
from metune_pipeline.vad import silence_gaps

CFG = PipelineConfig()
FACTS = {"clip_pct": 0.001, "longest_gap_s": 7.3, "spikes_per_min": 5.0, "speech_ratio": 0.6}


def vote(**kw):
    v = {"emotional_tone": "upset", "emotional_intensity": "high", "background_noise_present": False,
         "background_noise_type": "", "background_noise_severity": "none", "audio_quality": "clear",
         "speaker_overlap_present": False}
    return {**v, **kw}


def test_rules_thresholds_match_label_audit():
    assert rule_fields(FACTS, CFG) == {"long_silence_present": False, "audio_quality": "clear"}
    assert rule_fields({**FACTS, "longest_gap_s": 12}, CFG)["long_silence_present"] is True
    assert rule_fields({**FACTS, "clip_pct": 1.0}, CFG)["audio_quality"] == "slightly_impaired"
    assert rule_fields({**FACTS, "clip_pct": 5.0}, CFG)["audio_quality"] == "severely_impaired"


def test_baseline_static_from_spikes():
    assert baseline({**FACTS, "spikes_per_min": 30}, CFG).background_noise_type == "static"
    assert baseline(FACTS, CFG).background_noise_present is False


def test_fuse_majority_and_confidence():
    r = fuse(FACTS, [vote(), vote(), vote(emotional_tone="frustrated")], CFG)
    assert r.emotional_tone == "upset"
    assert 0.35 < r.confidence < 0.95


def test_fuse_consistency_rules():
    r = fuse(FACTS, [vote(background_noise_type="TV", background_noise_severity="medium")], CFG)
    assert (r.background_noise_type, r.background_noise_severity) == ("", "none")
    r = fuse(FACTS, [vote(background_noise_present=True, background_noise_type="TV")], CFG)
    assert r.background_noise_severity == "low"


def test_fuse_dsp_owns_silence_and_quality():
    r = fuse({**FACTS, "longest_gap_s": 15}, [vote(audio_quality="slightly_impaired")], CFG)
    assert r.long_silence_present is True
    assert r.audio_quality == "clear"           # LLM only overrides with severely_impaired


def test_fuse_measured_static_fills_llm_no_noise():
    r = fuse({**FACTS, "spikes_per_min": 30}, [vote(), vote()], CFG)
    assert (r.background_noise_present, r.background_noise_type, r.background_noise_severity) == (True, "static", "medium")
    r = fuse({**FACTS, "spikes_per_min": 30}, [vote(background_noise_present=True, background_noise_type="TV")], CFG)
    assert r.background_noise_type == "TV"      # an LLM-heard noise is kept
    assert fuse(FACTS, [vote()], CFG).background_noise_present is False


def test_fuse_caps_confidence_when_all_votes_are_defaults():
    default = vote(emotional_tone="neutral", emotional_intensity="low")
    assert fuse(FACTS, [default, default], CFG).confidence == CFG.default_vote_confidence
    assert fuse(FACTS, [vote(), vote()], CFG).confidence > CFG.default_vote_confidence
    # one vote heard something → the cap does not apply
    assert fuse(FACTS, [default, {**default, "speaker_overlap_present": True}], CFG).confidence > CFG.default_vote_confidence


def test_fuse_without_votes_is_baseline():
    assert fuse(FACTS, [], CFG, degraded=True).confidence == 0.2


def test_silence_gaps_ignore_edges():
    step = 0.1
    mask = np.array([1] * 30 + [0] * 120 + [1] * 30 + [0] * 25, bool)  # 12 s gap mid, 2.5 s at end
    assert silence_gaps(mask, step, edge_s=2.0) == [(3.0, 12.0)]


def test_dsp_counts_spikes():
    sr = CFG.sample_rate
    x = np.random.default_rng(0).normal(0, 1e-4, sr * 10).astype(np.float32)
    for t in range(2, 10):
        x[t * sr: t * sr + 200] = 0.5                  # 8 clicks
    f = dsp_features(x, CFG)
    assert f["spikes_per_min"] > 40
    assert f["clip_pct"] == 0


def ser(tone="upset", score=0.9):
    return {"tone": tone, "emotional_score": score, "probs": {}, "windows": 5}


def test_fuse_ser_owns_tone_and_fixes_contradicting_intensity():
    llm_neutral = vote(emotional_tone="neutral", emotional_intensity="low")
    r = fuse(FACTS, [llm_neutral] * 3, CFG, ser=ser("upset"))
    assert (r.emotional_tone, r.emotional_intensity) == ("upset", "medium")    # not "upset, low"
    r = fuse(FACTS, [vote()] * 3, CFG, ser=ser("frustrated"))
    assert (r.emotional_tone, r.emotional_intensity) == ("frustrated", "high")  # LLM heard emotion: keep its
    r = fuse(FACTS, [vote()] * 3, CFG, ser=ser("neutral", 0.2))
    assert r.emotional_tone == "neutral"


def test_fuse_ser_without_llm_and_confidence():
    r = fuse(FACTS, [], CFG, ser=ser("satisfied", 0.95))
    assert r.emotional_tone == "satisfied" and r.background_noise_present is False
    sure, unsure = fuse(FACTS, [], CFG, ser=ser("upset", 0.98)), fuse(FACTS, [], CFG, ser=ser("upset", 0.52))
    assert sure.confidence > unsure.confidence
    assert fuse(FACTS, [], CFG).emotional_tone == "neutral"          # tone_source=llm path unchanged


def test_ser_aggregate_and_missing_model():
    import pytest
    from metune_pipeline.schema import TONES
    from metune_pipeline.ser import aggregate, ser_facts

    calm, angry = np.eye(5)[TONES.index("neutral")], np.eye(5)[TONES.index("upset")]
    assert aggregate([(3.0, calm), (1.0, angry)], 1.0)[TONES.index("upset")] == 0.25   # duration-weighted
    assert aggregate([(3.0, calm), (1.0, angry)], 0.5)[TONES.index("upset")] == 1.0    # most emotional half
    assert aggregate([], 1.0)[TONES.index("neutral")] == 1.0
    with pytest.raises(FileNotFoundError):
        ser_facts(np.zeros(16000, np.float32), CFG.model_copy(update={"ser_model_path": "/nope.onnx"}))
