"""Combine LLM votes with DSP rules into one schema-valid Result."""
from collections import Counter

from .config import PipelineConfig
from .rules import baseline, rule_fields, static_present
from .schema import Result

LLM_MAJORITY = ["emotional_tone", "emotional_intensity", "background_noise_present",
                "background_noise_severity", "speaker_overlap_present"]


def majority(votes: list[dict], key: str):
    return Counter(v[key] for v in votes).most_common(1)[0]


def fuse(facts: dict, votes: list[dict], cfg: PipelineConfig, degraded: bool = False) -> Result:
    if not votes:
        return baseline(facts, cfg, confidence=0.2 if degraded else 0.3)
    rules = rule_fields(facts, cfg)
    out = {k: majority(votes, k)[0] for k in LLM_MAJORITY}
    types = [v["background_noise_type"] for v in votes
             if v["background_noise_present"] == out["background_noise_present"]]
    out["background_noise_type"] = Counter(types).most_common(1)[0][0].strip() if types else ""
    out["long_silence_present"] = rules["long_silence_present"]
    llm_q = majority(votes, "audio_quality")[0]
    out["audio_quality"] = "severely_impaired" if llm_q == "severely_impaired" else rules["audio_quality"]

    # measured static wins over an LLM "no noise"; an LLM-heard noise (TV, music…) is kept as is
    if not out["background_noise_present"] and static_present(facts, cfg):
        out["background_noise_present"], out["background_noise_type"] = True, "static"
        out["background_noise_severity"] = "medium"

    # consistency rules from the output schema
    if not out["background_noise_present"]:
        out["background_noise_type"], out["background_noise_severity"] = "", "none"
    else:
        if out["background_noise_severity"] == "none":
            out["background_noise_severity"] = "low"
        if not out["background_noise_type"]:
            out["background_noise_type"] = "unspecified"

    # confidence: tone agreement, scaled down when there is little speech (DR-009; uncalibrated)
    agree = majority(votes, "emotional_tone")[1] / len(votes)
    speech_pen = min(1.0, facts["speech_ratio"] / 0.3)
    out["confidence"] = round(0.35 + 0.6 * agree * speech_pen, 2)
    # votes that all fall back to the defaults agree trivially; that agreement is not evidence
    if all(_is_default(v) for v in votes):
        out["confidence"] = min(out["confidence"], cfg.default_vote_confidence)
    return Result(**out)


def _is_default(v: dict) -> bool:
    return (v["emotional_tone"] == "neutral" and not v["background_noise_present"]
            and not v["speaker_overlap_present"])
