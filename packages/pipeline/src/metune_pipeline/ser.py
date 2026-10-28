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
from scipy.signal import stft

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


def high_band_share(w: np.ndarray, sr: int = 16000) -> float:
    """Energy share 4.2–7.5 kHz of 0.3–7.5 kHz. Phone-line (customer) speech has none; the agent's VoIP leg on
    the trial calls does (DR-005): windows split at ~1e-4 vs 1e-2.5–1e-1.5."""
    f, _, z = stft(w, sr, nperseg=512)
    p = (np.abs(z) ** 2).sum(1)
    return float(p[(f > 4200) & (f < 7500)].sum() / (p[(f > 300) & (f < 7500)].sum() + 1e-12))


@lru_cache(maxsize=8)
def _session(path: str):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    # the arena keeps the largest buffer per model forever; with 5 models × variable-length windows × several
    # files in flight it pushed the API past 2 GB on the first hosted batch. Off: same speed, ~20% less RAM.
    opts.enable_cpu_mem_arena = False
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
    cuts = speech_windows(x, cfg.vad_threshold, sr)
    if cfg.ser_customer_only:          # E21: drop wideband (agent VoIP) windows; keep all if none are narrowband
        narrow = [(s, e) for s, e in cuts if high_band_share(x[s:e], sr) < cfg.ser_narrowband_max]
        cuts = narrow or cuts
    per = [[sess.run(None, {"wav": x[None, s:e].astype(np.float32)})[0][0] for sess in sessions] for s, e in cuts]
    durs = [(e - s) / sr for s, e in cuts]
    p = aggregate([(d, np.mean(w, axis=0)) for d, w in zip(durs, per)], cfg.ser_top_fraction)
    tone = decide(p, cfg.ser_tau)
    # E32 confidence signals: how many models give the same tone on their own, top-2 margin
    own = [decide(aggregate([(d, w[m]) for d, w in zip(durs, per)], cfg.ser_top_fraction), cfg.ser_tau)
           for m in range(len(sessions))]
    probs = dict(zip(TONES, (round(float(v), 4) for v in p)))
    score = 1 - float(p[TONES.index("neutral")])
    top2 = np.sort(p)[-2:]
    return {"tone": tone, "emotional_score": round(score, 3), "probs": probs, "windows": len(cuts),
            "models": len(sessions), "agree": sum(o == tone for o in own) / len(own),
            "margin": round(float(top2[1] - top2[0]), 4),
            "p_pred": round(float(p[TONES.index(tone)]) if tone != "neutral" else 1 - score, 4)}


def decide(p: np.ndarray, tau: float) -> str:
    """Neutral unless 1 − p(neutral) ≥ τ, else the strongest non-neutral tone."""
    score = 1 - p[TONES.index("neutral")]
    return "neutral" if score < tau else max(EMOTIONS, key=lambda t: p[TONES.index(t)])
