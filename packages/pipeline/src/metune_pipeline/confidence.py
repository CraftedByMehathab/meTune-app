"""Calibrated confidence (DR-009, E32): P(tone is right), fit on proxy calls with the trial-like wideband agent
(val_big_agent + val_big_agent_ovl) and scored once on unseen actors. The head is a small JSON file written by
eval/e32_confidence.py: a logistic regression on the SER bag's signals, or an isotonic map of the old formula.
"""
import json
from functools import lru_cache
from importlib import resources

import numpy as np


@lru_cache(maxsize=1)
def head() -> dict:
    return json.loads((resources.files("metune_pipeline") / "models" / "confidence_head.json").read_text())


def raw_confidence(ser: dict, speech_ratio: float) -> float:
    """The pre-E32 formula: distance from the neutral/emotional boundary, scaled down for little speech."""
    s = ser["emotional_score"]
    return round(0.35 + 0.6 * max(s, 1 - s) * min(1.0, speech_ratio / 0.3), 2)   # rounded as when the head was fit


def calibrated(ser: dict, speech_ratio: float) -> float:
    h = head()
    if h["kind"] == "isotonic":
        p = float(np.interp(raw_confidence(ser, speech_ratio), h["x"], h["y"]))
    else:
        signals = {**ser, "log_windows": np.log1p(ser["windows"]), "speech_ratio": speech_ratio}
        z = (np.array([signals[f] for f in h["features"]]) - np.array(h["mean"])) / np.array(h["scale"])
        p = float(1 / (1 + np.exp(-(np.array(h["coef"]) @ z + h["intercept"]))))
    return round(min(max(p, 0.0), 1.0), 2)
