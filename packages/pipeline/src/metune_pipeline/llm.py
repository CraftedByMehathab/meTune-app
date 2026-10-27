"""Audio-LLM adapter (Gemini). Votes run in parallel; every call logs tokens, cost and latency."""
import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass

from pydantic import ValidationError

from .config import PipelineConfig
from .schema import LLMOut, LLMOutHeard

# NOTE: on the 3 trial calls this prompt returns "neutral" for every call even though the model
# hears the emotion in free-text mode. Fix via proxy-set experiments, not by tuning on scored calls.
PROMPT = """You are an expert call-audio analyst. Analyze this customer-support call recording.

Rate the CUSTOMER only (not the agent). Decide who the customer is from context.
emotional_tone (customer's primary tone):
- neutral: no clear positive or negative emotion
- satisfied: pleased, relieved, appreciative, clearly positive
- frustrated: annoyed, impatient, dissatisfied, without strong anger
- upset: clearly angry, agitated, strongly dissatisfied
- distressed: highly emotional, overwhelmed, panicked, crying
emotional_intensity: low (subtle) | medium (clear, sustained) | high (strong, escalated)
background_noise_*: meaningful NON-SPEECH-OF-PARTICIPANTS sound (TV, music, chatter, road, static, typing, wind...).
  Ignore the short connect tone at the very start. Barely perceptible artifacts do not count.
  severity: none | low (audible, no interference) | medium (occasionally interferes) | high (materially impairs)
audio_quality: technical quality only (distortion, clipping, echo, robotic, packet loss, muffled).
speaker_overlap_present: people talk at the same time enough to affect understanding.
Rules: do NOT infer emotion from loudness alone. Do NOT call poor audio quality "background noise".

Measured facts (trust these): {facts}
Return JSON only."""

# v2 (DR-017): same rubric; the model must describe what it hears before it labels. In free-text mode
# the model distinguishes the calls; with labels first it falls back to the defaults (DR-014).
PROMPT_V2 = PROMPT.replace("Return JSON only.", """First fill `heard`: 2-4 short sentences on what is actually audible — how the
customer's voice sounds (pitch, pace, loudness changes, word choice), every non-speech sound you notice,
and any moment where voices overlap. Then choose every label so that it is consistent with `heard`.
Return JSON only.""")

# v3 (DR-017, E8): `heard` texts in E7 judged tone from the words ("requesting a new alarm clock" → neutral)
# and once took a crying baby in the background as the customer's emotion.
PROMPT_V3 = PROMPT_V2.replace("Rules: do NOT infer emotion from loudness alone.", """Judge emotion from HOW the customer speaks — pitch, energy, pace, tension, trembling, sighs,
  sobbing, laughter — not from WHAT they say: routine or polite words are often said angrily, fearfully or happily.
  Choose neutral only when the voice itself carries no emotional colour.
  Sounds from other people or things in the background (a crying baby, a TV) are noise, never the customer's tone.
Rules: do NOT infer emotion from loudness alone.""")

PROMPTS = {"v1": (PROMPT, LLMOut), "v2": (PROMPT_V2, LLMOutHeard), "v3": (PROMPT_V3, LLMOutHeard)}

RETRYABLE = {408, 429, 500, 502, 503, 504}


@dataclass
class LLMCall:
    vote: int
    model_id: str
    audio_in: int = 0
    text_in: int = 0
    text_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    error: str | None = None
    output: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def make_client(api_key: str | None = None):
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    from google import genai
    return genai.Client(api_key=key)


_next_slot = 0.0
_slot_lock: asyncio.Lock | None = None


async def _rate_limit(rpm: int) -> None:
    """Space requests evenly to stay under a requests-per-minute quota (DR-018). rpm=0 → no limit."""
    global _next_slot, _slot_lock
    if rpm <= 0:
        return
    if _slot_lock is None:
        _slot_lock = asyncio.Lock()
    async with _slot_lock:
        now = time.monotonic()
        wait = _next_slot - now
        _next_slot = max(now, _next_slot) + 60 / rpm
    if wait > 0:
        await asyncio.sleep(wait)


def _cost(c: LLMCall, cfg: PipelineConfig) -> float:
    p = cfg.price_per_1m
    return (c.audio_in * p["audio_in"] + c.text_in * p["text_in"] + c.text_out * p["text_out"]) / 1e6


async def _one_vote(client, audio: bytes, mime: str, facts: dict, cfg: PipelineConfig, i: int) -> LLMCall:
    from google.genai import errors, types

    call = LLMCall(vote=i, model_id=cfg.llm_model)
    prompt, schema = PROMPTS[cfg.llm_prompt]
    contents = [types.Part.from_bytes(data=audio, mime_type=mime), prompt.format(facts=json.dumps(facts))]
    gen = types.GenerateContentConfig(
        response_mime_type="application/json", response_schema=schema,
        temperature=cfg.llm_temperatures[i % len(cfg.llm_temperatures)],
        thinking_config=types.ThinkingConfig(thinking_level=cfg.llm_thinking_level),
    )
    t0 = time.perf_counter()
    for attempt in range(cfg.llm_retries + 1):
        backoff = 2 ** attempt
        try:
            await _rate_limit(cfg.llm_rpm)
            r = await asyncio.wait_for(
                client.aio.models.generate_content(model=cfg.llm_model, contents=contents, config=gen),
                cfg.llm_timeout_s,
            )
            call.output = schema.model_validate_json(r.text).model_dump()
            u = r.usage_metadata
            for d in (u.prompt_tokens_details or []):
                if "AUDIO" in str(d.modality):
                    call.audio_in += d.token_count or 0
                else:
                    call.text_in += d.token_count or 0
            call.text_out = (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)
            call.error = None
            break
        except errors.APIError as e:
            call.error = f"{e.code}: {e.message}"[:500]
            if e.code not in RETRYABLE:
                break
            if e.code == 429:
                backoff = 20 * (attempt + 1)     # per-minute quotas need tens of seconds, not 1-2 s
        except (TimeoutError, ValidationError) as e:
            call.error = f"{type(e).__name__}: {e}"[:500]
        if attempt < cfg.llm_retries:
            await asyncio.sleep(backoff)
    call.latency_ms = int((time.perf_counter() - t0) * 1000)
    call.cost_usd = _cost(call, cfg)
    return call


async def run_votes(client, audio: bytes, mime: str, facts: dict, cfg: PipelineConfig) -> list[LLMCall]:
    return list(await asyncio.gather(*(_one_vote(client, audio, mime, facts, cfg, i)
                                       for i in range(cfg.llm_votes))))
