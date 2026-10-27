"""One file in → one Result out. Used by the API worker, the CLI and the notebook."""
import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .decode import decode, encode_flac
from .dsp import dsp_features
from .fuse import fuse
from .llm import LLMCall, run_votes
from .schema import Result
from .ser import ser_facts
from .vad import vad_facts

FACT_KEYS = ["longest_gap_s", "speech_ratio", "clip_pct", "spikes_per_min", "snr_proxy_db"]


@dataclass
class Analysis:
    result: Result
    facts: dict
    duration_s: float
    calls: list[LLMCall] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)
    degraded: bool = False          # LLM was configured but every vote failed → baseline used
    ser: dict | None = None         # speech-emotion model output when tone_source=ser

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def latency_ms(self) -> int:
        return sum(self.timings_ms.values())


def local_facts(x: np.ndarray, cfg: PipelineConfig) -> dict:
    return {**dsp_features(x, cfg), **vad_facts(x, cfg)}


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


async def _ser(x: np.ndarray, cfg: PipelineConfig, timings: dict) -> dict:
    t0 = time.perf_counter()
    out = await asyncio.to_thread(ser_facts, x, cfg)
    timings["ser"] = _ms(t0)
    return out


async def analyze_file(path: Path, cfg: PipelineConfig, client=None) -> Analysis:
    timings = {}
    t0 = time.perf_counter()
    x = await asyncio.to_thread(decode, path, cfg.sample_rate)
    timings["decode"] = _ms(t0)

    t0 = time.perf_counter()
    facts = await asyncio.to_thread(local_facts, x, cfg)
    timings["local"] = _ms(t0)

    # runs alongside the LLM votes, so it adds no wall time when the LLM is on
    ser_task = asyncio.create_task(_ser(x, cfg, timings)) if cfg.tone_source == "ser" else None
    calls: list[LLMCall] = []
    if client is not None:
        t0 = time.perf_counter()
        audio = await asyncio.to_thread(encode_flac, x, cfg.sample_rate)
        calls = await run_votes(client, audio, "audio/flac", {k: facts[k] for k in FACT_KEYS}, cfg)
        timings["llm"] = _ms(t0)

    ser = await ser_task if ser_task else None
    votes = [c.output for c in calls if c.output]
    degraded = client is not None and not votes
    t0 = time.perf_counter()
    result = fuse(facts, votes, cfg, degraded=degraded, ser=ser)
    timings["fuse"] = _ms(t0)
    return Analysis(result=result, facts=facts, duration_s=round(len(x) / cfg.sample_rate, 2),
                    calls=calls, timings_ms=timings, degraded=degraded, ser=ser)
