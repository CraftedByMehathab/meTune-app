"""Speech-emotion tone (DR-020, E16): fine-tuned wav2vec2 via ONNX Runtime — PyTorch-free, like the VAD.

Audio LLMs (Gemini, Qwen) hear prosody-only emotion as neutral (E6–E15); this model reads the voice.
Speech windows (Silero VAD, 1–4 s) → per-window tone probabilities → duration-weighted mean over the most
emotional `ser_top_fraction` of windows → neutral unless 1 − p(neutral) ≥ `ser_tau`, else the strongest
non-neutral tone. Parameters tuned on val calls (actors held out from fine-tuning), scored once on test.
The model file is not vendored (379 MB): build it with eval/export_ser_onnx.py.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .schema import TONES
from .vad import speech_probs

MAX_S, MIN_S, BRIDGE_S = 4.0, 1.0, 0.3
EMOTIONS = [t for t in TONES if t != "neutral"]


def speech_windows(x: np.ndarray, threshold: float, sr: int = 16000) -> list[tuple[int, int]]:
    """Contiguous VAD speech runs (bridging pauses < 0.3 s), split into ≤ 4 s windows, ≥ 1 s kept."""
    probs, step = speech_probs(x, sr)
    hop = int(step * sr)
    mask = probs > threshold
    runs, start, quiet = [], None, 0
    for i, s in enumerate(list(mask) + [False] * 20):
        if s:
            start = i if start is None else start
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet * step >= BRIDGE_S:
                runs.append((start * hop, (i - quiet + 1) * hop))
                start, quiet = None, 0
    out = []
    for a, b in runs:
        for s in range(a, b, int(MAX_S * sr)):
            e = min(b, s + int(MAX_S * sr))
            if e - s >= MIN_S * sr:
                out.append((s, e))
    return out


@lru_cache(maxsize=8)
def _session(path: str):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    return ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])


def aggregate(windows: list[tuple[float, np.ndarray]], top: float) -> np.ndarray:
    """Duration-weighted mean probabilities over the most emotional `top` fraction of windows."""
    if not windows:
        return np.eye(len(TONES))[TONES.index("neutral")]
    neu = TONES.index("neutral")
    ranked = sorted(windows, key=lambda w: w[1][neu])[:max(1, round(len(windows) * top))]
    durs = np.array([d for d, _ in ranked])
    return (np.stack([p for _, p in ranked]) * durs[:, None]).sum(0) / durs.sum()


def ser_facts(x: np.ndarray, cfg: PipelineConfig) -> dict:
    """Several models (E20 bag) = mean of their window probabilities, as in eval/e16_calls.py."""
    paths = [Path(m) for m in cfg.ser_models]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        # fail loudly: a silent fallback to the LLM tone is the always-neutral behaviour this replaces
        raise FileNotFoundError(f"tone_source=ser but SER model(s) not found: {missing} (eval/export_ser_onnx.py)")
    sessions = [_session(str(p)) for p in paths]
    sr = cfg.sample_rate
    windows = [((e - s) / sr, np.mean([sess.run(None, {"wav": x[None, s:e].astype(np.float32)})[0][0]
                                       for sess in sessions], axis=0))
               for s, e in speech_windows(x, cfg.vad_threshold, sr)]
    p = aggregate(windows, cfg.ser_top_fraction)
    probs = dict(zip(TONES, (round(float(v), 4) for v in p)))
    score = 1 - probs["neutral"]
    tone = "neutral" if score < cfg.ser_tau else max(EMOTIONS, key=probs.get)
    return {"tone": tone, "emotional_score": round(score, 3), "probs": probs, "windows": len(windows),
            "models": len(sessions)}
