"""E33: per-tone bias on the SER bag's call probabilities: tone = argmax_k log p_k + b_k (b_neutral = 0), replacing
"neutral unless 1 − p(neutral) ≥ τ". Target: distressed → upset and frustrated → neutral/upset confusions (E21).

    uv run python eval/e33_tone_bias.py features --split val_big_agent        (+ val_big_agent_ovl, test_*)
    uv run python eval/e33_tone_bias.py fit --tune val_big_agent val_big_agent_ovl --test test_big_agent test_big_agent_ovl

Biases picked on --tune by coordinate ascent on macro F1 (grid −2…2, step 0.1); the procedure's own gain is
checked by 5-fold CV inside --tune. --test (unseen actors) scored once against the shipped τ rule.
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from metune_pipeline.config import load_config
from metune_pipeline.decode import decode
from metune_pipeline.schema import TONES
from metune_pipeline.ser import decide, ser_facts
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

ROOT, CACHE, RUNS = Path(".data/proxy"), Path(".data/proxy/e33_cache"), Path("eval/runs")
GRID = np.round(np.arange(-2, 2.01, 0.1), 2)


def features(split: str) -> None:
    cfg = load_config()
    lab = {r["name"]: json.loads(r["result_json"])["emotional_tone"]
           for r in csv.DictReader((ROOT / split / "labels.csv").open())}
    out = {}
    for k, n in enumerate(sorted(lab), 1):
        s = ser_facts(decode(ROOT / split / n, cfg.sample_rate), cfg)
        out[n] = {"probs": [s["probs"][t] for t in TONES], "label": lab[n]}
        print(f"  {k:>3}/{len(lab)} {n} {s['tone']:<10} label {lab[n]}", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"{split}.json").write_text(json.dumps(out))


def load(splits):
    rows = [r for s in splits for r in json.loads((CACHE / f"{s}.json").read_text()).values()]
    return np.array([r["probs"] for r in rows]), np.array([r["label"] for r in rows])


def predict(P, b):
    return np.array(TONES)[(np.log(np.clip(P, 1e-6, 1)) + b).argmax(1)]


def macro(y, p):
    return f1_score(y, p, labels=TONES, average="macro", zero_division=0)


def fit_bias(P, y, passes=4):
    b = np.zeros(len(TONES))
    for _ in range(passes):
        for k in range(1, len(TONES)):                       # neutral stays 0
            scores = []
            for g in GRID:
                bb = b.copy(); bb[k] = g
                scores.append(macro(y, predict(P, bb)))
            best = max(scores)
            # ties → smallest |bias| (closest to "no change")
            b[k] = min((g for g, s in zip(GRID, scores) if s == best), key=abs)
    return b


def tau_rule(P, tau):
    return np.array([decide(p, tau) for p in P])


def report(y, p):
    return {"macro_f1": round(macro(y, p), 3), "accuracy": round(float((y == p).mean()), 3),
            "recall": {t: round(float((p[y == t] == t).mean()), 3) for t in TONES},
            "confusion (rows=true N S F U D)": confusion_matrix(y, p, labels=TONES).tolist()}


def fit(a):
    cfg = load_config()
    P, y = load(a.tune)
    Pt, yt = load(a.test)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    oof_bias, oof_tau = np.empty(len(y), object), tau_rule(P, cfg.ser_tau)
    for tr, te in cv.split(P, y):
        oof_bias[te] = predict(P[te], fit_bias(P[tr], y[tr]))
    b = fit_bias(P, y)
    res = {"tune": a.tune, "test": a.test, "n_tune": len(y), "n_test": len(yt),
           "bias": dict(zip(TONES, b.round(2).tolist())),
           "tune_cv": {"shipped tau rule": report(y, oof_tau), "bias (5-fold CV)": report(y, oof_bias.astype(str))},
           "test": {"shipped tau rule": report(yt, tau_rule(Pt, cfg.ser_tau)), "bias": report(yt, predict(Pt, b))},
           "when": datetime.now().isoformat(timespec="seconds")}
    out = RUNS / f"{datetime.now():%Y%m%d-%H%M}_E33-tone-bias"
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))
    print("→", out)
    from track import log_run
    log_run(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("features").add_argument("--split", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--tune", nargs="+", required=True)
    f.add_argument("--test", nargs="+", required=True)
    a = ap.parse_args()
    features(a.split) if a.cmd == "features" else fit(a)
