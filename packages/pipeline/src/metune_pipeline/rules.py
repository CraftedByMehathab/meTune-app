"""DSP-owned fields (DR-006) and the rules-only baseline."""
from .config import PipelineConfig
from .schema import Result


def rule_fields(f: dict, cfg: PipelineConfig) -> dict:
    q = "clear"
    if f["clip_pct"] > cfg.clip_pct_slight:
        q = "slightly_impaired"
    if f["clip_pct"] > cfg.clip_pct_severe:
        q = "severely_impaired"
    return {"long_silence_present": f["longest_gap_s"] >= cfg.long_silence_s, "audio_quality": q}


def static_present(f: dict, cfg: PipelineConfig) -> bool:
    """Click/crackle rate high enough to call it static. Measured, so it holds with or without an LLM."""
    return f["spikes_per_min"] >= cfg.spike_rate_static


def baseline(f: dict, cfg: PipelineConfig, confidence: float = 0.3) -> Result:
    """Majority-class tone + spike-based static. Used without an LLM, or when every LLM vote failed."""
    static = static_present(f, cfg)
    return Result(
        emotional_tone="neutral", emotional_intensity="medium",
        background_noise_present=static, background_noise_type="static" if static else "",
        background_noise_severity="medium" if static else "none",
        speaker_overlap_present=False, confidence=confidence, **rule_fields(f, cfg),
    )
