"""Output schema from the trial brief. Field order = output order."""
from typing import Literal

from pydantic import BaseModel, Field

Tone = Literal["neutral", "satisfied", "frustrated", "upset", "distressed"]
Level = Literal["low", "medium", "high"]
Severity = Literal["none", "low", "medium", "high"]
Quality = Literal["clear", "slightly_impaired", "severely_impaired"]

TONES: list[str] = list(Tone.__args__)


class LLMOut(BaseModel):
    """What the LLM is allowed to decide. Silence + confidence are NOT the LLM's job."""
    emotional_tone: Tone
    emotional_intensity: Level
    background_noise_present: bool
    background_noise_type: str
    background_noise_severity: Severity
    audio_quality: Quality
    speaker_overlap_present: bool


class Heard(BaseModel):
    heard: str = Field(description="2-4 short sentences on what is actually audible, written before any label")


class LLMOutHeard(LLMOut, Heard):
    """Prompt v2 (DR-017): `heard` is generated first so the labels are conditioned on a description.
    Base order matters: pydantic lists the last base's fields first."""


class Result(LLMOut):
    long_silence_present: bool
    confidence: float = Field(ge=0, le=1)


FIELDS: list[str] = list(Result.model_fields)
# Fields scored against labels (free-text noise type and confidence are excluded)
SCORED_FIELDS: list[str] = [f for f in FIELDS if f not in ("background_noise_type", "confidence")]
