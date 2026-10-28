"""Combine LLM votes with DSP rules into one schema-valid Result."""
from collections import Counter

from .config import PipelineConfig
from .rules import baseline, rule_fields, static_present
from .schema import Result

LLM_MAJORITY = ["emotional_tone", "emotional_intensity", "background_noise_present",
                "background_noise_severity", "speaker_overlap_present"]


def majority(votes: list[dict], key: str):
    return Counter(v[key] for v in votes).most_common(1)[0]


def fuse(facts: dict, votes: list[dict], cfg: PipelineConfig, degraded: bool = False,
         ser: dict | None = None, noise: dict | None = None, overlap: dict | None = None) -> Result:
    """`ser` (tone_source=ser): the speech-emotion model owns tone, also without any LLM votes.
    `noise` (noise_source=ast): the sound classifier owns the noise fields, also without any LLM votes.
    `overlap` (overlap_detector): ORed with the LLM's overlap answer; alone without LLM votes (E31)."""
    if not votes:
        out = baseline(facts, cfg, confidence=0.2 if degraded else 0.3).model_dump()
        if ser:
            out = _apply_ser(out, ser, facts, llm_emotional=False)
        if noise:
            out = _apply_noise(out, noise, facts, cfg)
        if overlap:
            out["speaker_overlap_present"] = overlap["present"]
        return Result(**out)
    rules = rule_fields(facts, cfg)
    out = {k: majority(votes, k)[0] for k in LLM_MAJORITY}
    types = [v["background_noise_type"] for v in votes
             if v["background_noise_present"] == out["background_noise_present"]]
    out["background_noise_type"] = Counter(types).most_common(1)[0][0].strip() if types else ""
    out["long_silence_present"] = rules["long_silence_present"]
    llm_q = majority(votes, "audio_quality")[0]
    out["audio_quality"] = "severely_impaired" if llm_q == "severely_impaired" else rules["audio_quality"]
    if noise:
        out = _apply_noise(out, noise, facts, cfg)
    if overlap:
        out["speaker_overlap_present"] = out["speaker_overlap_present"] or overlap["present"]

    # measured static wins over an LLM "no noise"; an LLM-heard noise (TV, music…) is kept as is
    if not out["background_noise_present"] and static_present(facts, cfg) and not noise:
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
    if ser:
        out = _apply_ser(out, ser, facts, llm_emotional=out["emotional_tone"] != "neutral")
    return Result(**out)


def _apply_ser(out: dict, ser: dict, facts: dict, llm_emotional: bool) -> dict:
    out = {**out, "emotional_tone": ser["tone"]}
    # The SER score does not separate low/medium/high (E16 val calls), so intensity stays the LLM's unless
    # that would contradict the tone (LLM heard neutral → its "low"); then the most common label, medium.
    if ser["tone"] != "neutral" and not llm_emotional:
        out["emotional_intensity"] = "medium"
    # confidence: distance from the neutral/emotional boundary, scaled down for little speech (uncalibrated)
    certainty = max(ser["emotional_score"], 1 - ser["emotional_score"])
    out["confidence"] = round(0.35 + 0.6 * certainty * min(1.0, facts["speech_ratio"] / 0.3), 2)
    return out


def _apply_noise(out: dict, noise: dict, facts: dict, cfg: PipelineConfig) -> dict:
    out = {**out, "background_noise_present": noise["present"], "background_noise_type": noise["type"],
           "background_noise_severity": noise["severity"]}
    # measured static still wins over a "no noise", when AST also hears clicks / static (E28)
    if not noise["present"] and static_present(facts, cfg) and noise["static_confirmed"]:
        out.update(background_noise_present=True, background_noise_type="static", background_noise_severity="medium")
    return out


def _is_default(v: dict) -> bool:
    return (v["emotional_tone"] == "neutral" and not v["background_noise_present"]
            and not v["speaker_overlap_present"])
