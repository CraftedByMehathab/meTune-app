import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class PipelineConfig(BaseModel):
    sample_rate: int = 16000
    frame_ms: int = 20
    skip_start_s: float = 1.0
    edge_ignore_s: float = 2.0
    long_silence_s: float = 10.0
    vad_threshold: float = 0.5
    clip_pct_slight: float = 0.5
    clip_pct_severe: float = 3.0
    spike_jump_db: float = 25.0
    spike_rate_static: float = 20.0
    default_vote_confidence: float = 0.4

    tone_source: Literal["llm", "ser"] = "llm"
    ser_models: list[str] = [".data/models/ser_e16.onnx"]   # several = ensemble (mean probabilities)
    ser_top_fraction: float = 1.0
    ser_tau: float = 0.5
    ser_customer_only: bool = False      # E21: score narrowband (phone-line) windows only
    ser_narrowband_max: float = 1e-3     # high-band share below this = narrowband
    ser_bias: dict[str, float] | None = None   # E33: tone = argmax log p + bias (replaces the ser_tau rule when set)

    noise_source: Literal["llm", "ast"] = "llm"     # ast = AudioSet classifier + trained head owns noise (E25)
    noise_model: str = ".data/models/ast_audioset.int8.onnx"

    overlap_detector: bool = False       # E31: pyannote overlap, ORed with the LLM's answer
    overlap_frame_p: float = 0.4
    overlap_min_s: float = 0.42

    llm_model: str = "gemini-3.5-flash-lite"
    llm_prompt: str = "v2"
    llm_votes: int = 3
    llm_temperatures: list[float] = [0.0, 0.4, 0.7]
    llm_thinking_level: str = "minimal"
    llm_timeout_s: float = 120
    llm_rpm: int = 0
    llm_retries: int = 2
    price_per_1m: dict[str, float] = {"audio_in": 0.30, "text_in": 0.30, "text_out": 2.50}


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """Load config.yaml from `path`, $METUNE_CONFIG, or ./config.yaml; defaults if none found."""
    path = Path(path or os.environ.get("METUNE_CONFIG", "config.yaml"))
    if not path.exists():
        return PipelineConfig()
    return PipelineConfig(**(yaml.safe_load(path.read_text()) or {}))


def config_hash(cfg: PipelineConfig) -> str:
    return hashlib.sha256(json.dumps(cfg.model_dump(), sort_keys=True).encode()).hexdigest()[:12]
