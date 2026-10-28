"""E32: calibrate `confidence` as P(tone is right) (brief: "calibrated"; DR-009: isotonic on the proxy set, ECE).

    uv run python eval/e32_confidence.py features --split val_big_agent      (and val_big_agent_ovl, test_*)
    uv run python eval/e32_confidence.py fit --tune val_big_agent val_big_agent_ovl \
        --test test_big_agent test_big_agent_ovl

Per call, from the shipped SER bag (customer-only windows, E21 rule): the current formula's raw confidence, p of the
predicted tone, top-2 margin, how many of the 5 models give the same tone on their own, windows, speech ratio.
Candidates: current formula · isotonic(current) · logistic regression on all signals. Picked on --tune by 5-fold
CV log-loss; --test (unseen actors) scored once: ECE (10 bins), Brier, wrong answers at ≥ 0.9.
"""
import argparse
import csv
import json
from datetime import datetime
from importlib import resources
from pathlib import Path

import numpy as np
from metune_pipeline.analyze import local_facts
from metune_pipeline.config import load_config
from metune_pipeline.decode import decode
from metune_pipeline.schema import TONES
from metune_pipeline.ser import _session, aggregate, decide, high_band_share, speech_windows

ROOT, CACHE, RUNS = Path(".data/proxy"), Path(".data/proxy/e32_cache"), Path("eval/runs")
HEAD = Path(str(resources.files("metune_pipeline") / "models" / "confidence_head.json"))
FEATS = ["p_pred", "margin", "log_windows", "speech_ratio"]   # E33 refit: "agree" dropped (weight −0.04 in E32)


def call_signals(x: np.ndarray, cfg) -> dict:
    """Same windows and rule as ser.ser_facts, but keeps each model's own answer."""
    sessions = [_session(m) for m in cfg.ser_models]
    cuts = speech_windows(x, cfg.vad_threshold, cfg.sample_rate)
    if cfg.ser_customer_only:
        cuts = [(s, e) for s, e in cuts if high_band_share(x[s:e], cfg.sample_rate) < cfg.ser_narrowband_max] or cuts
    per = np.array([[sess.run(None, {"wav": x[None, s:e].astype(np.float32)})[0][0] for sess in sessions]
                    for s, e in cuts]) if cuts else np.zeros((0, len(sessions), len(TONES)))
    durs = [(e - s) / cfg.sample_rate for s, e in cuts]
    bag = aggregate(list(zip(durs, per.mean(1))) if len(cuts) else [], cfg.ser_top_fraction)
    tone = decide(bag, cfg.ser_tau, cfg.ser_bias)
    own = [decide(aggregate(list(zip(durs, per[:, m])) if len(cuts) else [], cfg.ser_top_fraction), cfg.ser_tau,
                  cfg.ser_bias)
           for m in range(len(sessions))]
    facts = local_facts(x, cfg)
    s = 1 - bag[TONES.index("neutral")]
    srt = np.sort(bag)
    return {"tone": tone, "p_pred": float(bag[TONES.index(tone)] if tone != "neutral" else 1 - s),
            "margin": float(srt[-1] - srt[-2]), "agree": sum(o == tone for o in own) / len(own),
            "log_windows": float(np.log1p(len(cuts))), "speech_ratio": float(facts["speech_ratio"]),
            "raw": round(0.35 + 0.6 * max(s, 1 - s) * min(1.0, facts["speech_ratio"] / 0.3), 2)}


def features(split: str) -> None:
    cfg = load_config()
    lab = {r["name"]: json.loads(r["result_json"])["emotional_tone"]
           for r in csv.DictReader((ROOT / split / "labels.csv").open())}
    out = {}
    for k, n in enumerate(sorted(lab), 1):
        sig = call_signals(decode(ROOT / split / n, cfg.sample_rate), cfg)
        out[n] = {**sig, "label": lab[n], "right": sig["tone"] == lab[n]}
        print(f"  {k:>3}/{len(lab)} {n} {sig['tone']:<10} right={out[n]['right']} raw={sig['raw']}", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"{split}.json").write_text(json.dumps(out))


def oof_rows(splits: list[str]) -> list[dict]:
    """Tune rows with the tone the E33 bias would give WITHOUT having seen the call: biases fit on the other 4 folds
    (the shipped bias was fit on these calls, so in-sample "right" rates are optimistic and the calibrator learned
    to be over-confident). p_pred is the bag probability of that tone; margin does not depend on the bias."""
    from e33_tone_bias import CACHE as BAG, fit_bias, predict
    from sklearn.model_selection import StratifiedKFold

    rows = [r for s in splits for r in json.loads((CACHE / f"{s}.json").read_text()).values()]
    bags = [r for s in splits for r in json.loads((BAG / f"{s}.json").read_text()).values()]
    P, y = np.array([b["probs"] for b in bags]), np.array([b["label"] for b in bags])
    assert all(r["label"] == lab for r, lab in zip(rows, y)), "e32 / e33 caches out of order"
    tone = np.empty(len(y), object)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(P, y):
        tone[te] = predict(P[te], fit_bias(P[tr], y[tr]))
    for r, p, t in zip(rows, P, tone):
        r["p_pred"], r["right"] = float(p[TONES.index(t)]), bool(t == r["label"])
    return rows


def load(splits: list[str], oof: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = oof_rows(splits) if oof else [r for s in splits for r in json.loads((CACHE / f"{s}.json").read_text()).values()]
    return (np.array([[r[f] for f in FEATS] for r in rows]), np.array([r["raw"] for r in rows]),
            np.array([r["right"] for r in rows], float))


def ece(conf: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum(abs(conf[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def report(conf: np.ndarray, y: np.ndarray) -> dict:
    wrong = y == 0
    return {"ece": round(ece(conf, y), 3), "brier": round(float(np.mean((conf - y) ** 2)), 3),
            "mean_conf": round(float(conf.mean()), 3), "accuracy": round(float(y.mean()), 3),
            "conf_when_right": round(float(conf[~wrong].mean()), 3), "conf_when_wrong": round(float(conf[wrong].mean()), 3),
            "wrong_at_0.9+": f"{int(((conf >= 0.9) & wrong).sum())}/{int(wrong.sum())}",
            "flagged_<0.5": f"{int((conf < 0.5).sum())} ({float(y[conf < 0.5].mean()) if (conf < 0.5).any() else 0:.2f} right)"}


def fit(a) -> None:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, raw, y = load(a.tune, oof=bool(load_config().ser_bias))
    Xt, rawt, yt = load(a.test)
    make = {"isotonic": lambda: IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip"),
            "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0))}
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    oof = {"current": np.clip(raw, 0.01, 0.99), "isotonic": np.zeros(len(y)), "logistic": np.zeros(len(y))}
    for tr, te in cv.split(X, y):
        oof["isotonic"][te] = make["isotonic"]().fit(raw[tr], y[tr]).predict(raw[te])
        oof["logistic"][te] = make["logistic"]().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    cv_scores = {k: {"log_loss": round(log_loss(y, np.clip(v, 0.01, 0.99)), 3), **report(v, y)} for k, v in oof.items()}
    pick = min(("isotonic", "logistic"), key=lambda k: cv_scores[k]["log_loss"])

    iso, lr = make["isotonic"]().fit(raw, y), make["logistic"]().fit(X, y)
    test = {"current": report(rawt, yt), "isotonic": report(iso.predict(rawt), yt),
            "logistic": report(lr.predict_proba(Xt)[:, 1], yt)}
    res = {"tune": a.tune, "test": a.test, "n_tune": len(y), "n_test": len(yt), "picked_on_tune_cv": pick,
           "tune_cv": cv_scores, "test": test, "when": datetime.now().isoformat(timespec="seconds")}
    if pick == "logistic":
        sc, m = lr.named_steps["standardscaler"], lr.named_steps["logisticregression"]
        head = {"kind": "logistic", "features": FEATS, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
                "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0])}
    else:
        head = {"kind": "isotonic", "x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
    HEAD.write_text(json.dumps({**head, "experiment": "E32", "fit_on": a.tune, "target": "tone correct"}))
    out = RUNS / f"{datetime.now():%Y%m%d-%H%M}_E32-confidence"
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))
    print("→", out, "·", HEAD)
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
