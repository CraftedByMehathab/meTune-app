"""Silero VAD v5/v6 via ONNX Runtime (no PyTorch). Model vendored under models/ (MIT)."""
from functools import lru_cache
from importlib import resources

import numpy as np

from .config import PipelineConfig

WINDOW = 512  # samples at 16 kHz = 32 ms


@lru_cache(maxsize=1)
def _session():
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    model = resources.files("metune_pipeline") / "models" / "silero_vad.onnx"
    return ort.InferenceSession(model.read_bytes(), sess_options=opts, providers=["CPUExecutionProvider"])


def speech_probs(x: np.ndarray, sr: int = 16000) -> tuple[np.ndarray, float]:
    """Per-window speech probability, seconds per window."""
    sess = _session()
    state = np.zeros((2, 1, 128), np.float32)
    ctx = np.zeros((1, 64), np.float32)
    out = []
    for i in range(0, len(x) - WINDOW + 1, WINDOW):
        chunk = np.concatenate([ctx, x[None, i:i + WINDOW]], axis=1).astype(np.float32)
        p, state = sess.run(None, {"input": chunk, "state": state, "sr": np.array(sr, np.int64)})
        ctx = chunk[:, -64:]
        out.append(float(p[0, 0]))
    return np.array(out), WINDOW / sr


def silence_gaps(mask: np.ndarray, step: float, edge_s: float) -> list[tuple[float, float]]:
    """(start_s, length_s) for non-speech runs, ignoring call edges."""
    gaps, start = [], None
    for i, s in enumerate(list(mask) + [True]):
        if not s and start is None:
            start = i
        if s and start is not None:
            a, b = start * step, i * step
            if a > edge_s and b < len(mask) * step - edge_s:
                gaps.append((round(a, 1), round(b - a, 1)))
            start = None
    return gaps


def vad_facts(x: np.ndarray, cfg: PipelineConfig) -> dict:
    probs, step = speech_probs(x, cfg.sample_rate)
    mask = probs > cfg.vad_threshold
    gaps = silence_gaps(mask, step, cfg.edge_ignore_s)
    longest = max(gaps, key=lambda g: g[1]) if gaps else None
    return {
        "speech_ratio": round(float(mask.mean()), 2) if len(mask) else 0.0,
        "longest_gap_s": longest[1] if longest else 0.0,
        "gap_at_s": longest[0] if longest else None,
    }
