"""Speaker overlap detector (E31): pyannote segmentation-3.0 via ONNX Runtime. Model vendored under models/ (MIT).

The powerset output per ~17 ms frame is: none, one of 3 speakers, or two at once (classes 4–6). Windows of 10 s,
2.5 s hop, averaged per frame. A call has overlap when p(two speakers) > `overlap_frame_p` for ≥ `overlap_min_s`
seconds (picked on val_big_agent_ovl). The pipeline ORs this with the LLM's answer: they catch different overlaps
(test_big_agent_ovl recall: LLM 0.40, detector 0.47, either 0.68, precision 0.90).
"""
from functools import lru_cache
from importlib import resources

import numpy as np

from .config import PipelineConfig

WIN, HOP, SHIFT = 160000, 40000, 270       # samples at 16 kHz; SHIFT = model frame step


@lru_cache(maxsize=1)
def _session():
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    model = resources.files("metune_pipeline") / "models" / "pyannote_seg3.onnx"
    return ort.InferenceSession(model.read_bytes(), sess_options=opts, providers=["CPUExecutionProvider"])


def overlap_probs(x: np.ndarray) -> tuple[np.ndarray, float]:
    """p(two speakers at once) per model frame, seconds per frame."""
    pad = np.concatenate([x, np.zeros(WIN, np.float32)])
    starts = list(range(0, max(1, len(x) - WIN + HOP), HOP))
    batch = np.stack([pad[s:s + WIN] for s in starts])[:, None, :].astype(np.float32)
    p = np.exp(_session().run(None, {"x": batch})[0])            # (windows, frames, 7) log-probs
    ov = (p / p.sum(-1, keepdims=True))[..., 4:].sum(-1)
    total = len(x) // SHIFT + 1
    acc, cnt = np.zeros(total), np.zeros(total)
    for s, row in zip(starts, ov):
        a = s // SHIFT
        k = min(len(row), total - a)
        acc[a:a + k] += row[:k]
        cnt[a:a + k] += 1
    return acc / np.maximum(cnt, 1), SHIFT / 16000


def overlap_facts(x: np.ndarray, cfg: PipelineConfig) -> dict:
    p, step = overlap_probs(x)
    seconds = float((p > cfg.overlap_frame_p).sum() * step)
    return {"present": seconds >= cfg.overlap_min_s, "seconds": round(seconds, 2)}
