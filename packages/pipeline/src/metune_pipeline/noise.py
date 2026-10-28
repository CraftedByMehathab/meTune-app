"""Background noise (E23, E25): AudioSet sound classifier (AST, 8-bit ONNX) + a small trained head.

Gemini caught 44–68% of noisy proxy calls (E22, E26); AST's class scores separate noisy from clean calls almost
perfectly (val_big AUC 0.98–0.99). Per call: AST on ≤ 10.24 s windows of the whole call and of its non-speech
gaps → strongest class in each noise group (3 sources × 7 groups) → logistic-regression head → severity
(none / low / medium / high). Kinds the proxy set lacks (TV, music) cannot be learned; they get a zero-shot
threshold set above every noise-free tuning call. Head fit on val_big (eval/e25_noise_head.py), scored once on
test_big. The AST model file is not vendored (92 MB): build it with eval/export_ast_onnx.py.
"""
import json
from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .vad import speech_probs

# meTune noise type → AudioSet class ids. Chatter uses babble only: "Conversation" / "Speech" fire on the call itself.
GROUPS = {
    "keyboard typing": [124, 384, 385, 386, 491, 492],
    "road noise": [300, 306, 307, 308, 309, 314, 316, 318, 322, 323, 324, 325, 327, 343, 344, 348, 349, 352, 396],
    "appliance hum": [369, 377, 412, 496, 516],
    "dog barking": [74, 75, 76, 77, 78, 79, 80, 122],
    "baby crying": [23, 24],
    "background chatter": [68, 69, 70],
    "static": [84, 299, 513, 515, 520, 521],
}
EXTRA = {"TV / music": [137, 524, 525]}        # Music, Television, Radio: zero-shot only (no proxy examples)
CLICKY = ["keyboard typing", "static"]         # incl. AudioSet "Clicking": confirms the DSP click-rate static rule (E28)
WIN_S = 10.24                                  # AST input length (1024 frames)


@lru_cache(maxsize=2)
def _session(path: str):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.enable_cpu_mem_arena = False          # see ser._session
    return ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])


@lru_cache(maxsize=1)
def head() -> dict:
    return json.loads((resources.files("metune_pipeline") / "models" / "noise_head.json").read_text())


def gap_audio(x: np.ndarray, probs: np.ndarray, step: float, sr: int = 16000) -> np.ndarray:
    """Non-speech stretches only (VAD < 0.3, one window of margin each side), joined with 10 ms fades."""
    w = int(step * sr)
    speech = probs >= 0.3
    speech = speech | np.roll(speech, 1) | np.roll(speech, -1)
    fade = np.linspace(0, 1, sr // 100, dtype=np.float32)
    parts, i = [], 0
    while i < len(speech):
        if speech[i]:
            i += 1
            continue
        j = i
        while j < len(speech) and not speech[j]:
            j += 1
        seg = x[i * w:j * w].copy()
        if len(seg) > 2 * len(fade):
            seg[:len(fade)] *= fade
            seg[-len(fade):] *= fade[::-1]
            parts.append(seg)
        i = j
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


def window_probs(sess, x: np.ndarray, hop_s: float, sr: int = 16000) -> np.ndarray:
    """AudioSet probabilities per window → (windows, 527). Windows < 1 s are dropped unless there is no other."""
    n, hop = int(WIN_S * sr), int(hop_s * sr)
    chunks = [x[i:i + n] for i in range(0, max(1, len(x) - n + hop), hop)]
    chunks = [c for c in chunks if len(c) >= sr] or [x]
    return np.stack([sess.run(None, {"wav": c[None].astype(np.float32)})[0][0] for c in chunks])


def ast_scores(x: np.ndarray, cfg: PipelineConfig) -> tuple[np.ndarray, float, float]:
    """(3 sources: whole-call max, whole-call mean, gaps max) × 527 AudioSet probabilities; gap seconds and dBFS."""
    path = Path(cfg.noise_model)
    if not path.exists():
        raise FileNotFoundError(f"noise_source=ast but model not found: {path} (eval/export_ast_onnx.py)")
    sess, sr = _session(str(path)), cfg.sample_rate
    full = window_probs(sess, x, 10.0, sr)
    gaps = gap_audio(x, *speech_probs(x, sr), sr)
    gap = window_probs(sess, gaps, 5.0, sr).max(0) if len(gaps) >= sr else np.zeros(full.shape[1], np.float32)
    return np.stack([full.max(0), full.mean(0), gap]), len(gaps) / sr, gap_db(gaps)


def gap_db(gaps: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(gaps ** 2)) + 1e-12)) if len(gaps) else -240.0


def group_scores(s: np.ndarray, groups: dict = GROUPS) -> np.ndarray:
    """Strongest class within each group, per source → (3, groups)."""
    return np.stack([[row[ids].max() for ids in groups.values()] for row in s])


def head_features(g: np.ndarray, gap_s: float) -> np.ndarray:
    """Log-odds of the 21 group scores + log gap seconds. Loud noise leaves few gaps (VAD hears it as speech), so
    quiet gaps alone read as "high" noise; their length tells that apart from long silent pauses."""
    g = np.clip(g.reshape(-1), 1e-4, 1 - 1e-4)
    return np.append(np.log(g / (1 - g)), np.log1p(gap_s))


def classify(s: np.ndarray, gap_s: float, gap_dbfs: float = 0.0) -> dict:
    """AST scores → present / type / severity, using the trained head (+ zero-shot extras)."""
    h = head()
    g = group_scores(s)
    # Proxy gaps always carry some line hiss (≥ -63 dBFS); the head learned "near-zero gap scores = loud noise
    # filled the gaps". Gaps quieter than any tuning call (digital silence) get a typical clean call's scores.
    if gap_s >= 1 and gap_dbfs < h["gap_db_min"]:
        g = np.vstack([g[:2], h["gap_fill"]])
    # clamp to the tuning range: silence scores far below any proxy call; a linear head must not extrapolate
    f = np.clip(head_features(g, gap_s), h["lo"], h["hi"])
    z = (f - np.array(h["mean"])) / np.array(h["scale"])
    logits = np.array(h["coef"]) @ z + np.array(h["intercept"])
    p = np.exp(logits - logits.max())
    p /= p.sum()
    severity = h["classes"][int(p.argmax())]
    best = g[[0, 2]].max(0)                                  # whole-call max or gaps max, per group
    kind = list(GROUPS)[int(best.argmax())]
    extra = group_scores(s, EXTRA)[[0, 2]].max(0)
    for (name, thr), score in zip(h["extra_threshold"].items(), extra):
        if score >= thr and (severity == "none" or score > best.max()):
            kind, severity = name, "medium" if severity == "none" else severity
    present = severity != "none"
    # E28: the DSP click rate also counts word onsets (fired on 15/59 clean val_big calls); trust it only when
    # AST hears clicks or static too
    clicky = float(best[[list(GROUPS).index(k) for k in CLICKY]].max())
    return {"present": present, "type": kind if present else "", "severity": severity,
            "static_confirmed": clicky >= h["static_confirm"],
            "p_none": round(float(p[h["classes"].index("none")]), 3),
            "groups": {k: round(float(v), 3) for k, v in zip([*GROUPS, *EXTRA], [*best, *extra])}}


def noise_facts(x: np.ndarray, cfg: PipelineConfig) -> dict:
    return classify(*ast_scores(x, cfg))
