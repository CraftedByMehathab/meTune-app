"""E25: fit the noise head on the shipped 8-bit AST's scores (int8 shifts probabilities by up to 0.1 vs E23's fp32).

    uv run --group ser python eval/e25_noise_head.py features --split val_big
    uv run --group ser python eval/e25_noise_head.py fit --tune val_big
    uv run --env-file .env python eval/run_eval.py --split test_big --name E25-ast --no-llm --set noise_source=ast

Head = logistic regression, 4-way severity (present = not none), on log-odds of the 21 group scores (E23b recipe)
+ log gap seconds.
Out-of-range guards (silence came out as "medium road noise" without them): inputs clamped to the tuning range;
gaps quieter than any tuning call (− 3 dB) get the median gap scores of the clean tuning calls.
Extras (TV / music) have no proxy examples: threshold = 1.2 × the highest score on any noise-free tuning call.
E28: the DSP static rule needs AST's clicks/static score ≥ 1.2 × its max on noise-free tuning calls.
Leakage rule (DR-008): fit on val_big only (ESC-50 folds 1–3); test_big (folds 4–5) is scored once.
"""
import argparse
import csv
import json
from importlib import resources
from pathlib import Path

import numpy as np
from metune_pipeline.config import load_config
from metune_pipeline.decode import decode
from metune_pipeline.noise import CLICKY, EXTRA, GROUPS, ast_scores, group_scores, head_features

ROOT = Path(".data/proxy")
CACHE = ROOT / "e25_cache"
HEAD = Path(str(resources.files("metune_pipeline") / "models" / "noise_head.json"))


def labels_of(split: str) -> dict:
    return {r["name"]: json.loads(r["result_json"]) for r in csv.DictReader((ROOT / split / "labels.csv").open())}


def features(split: str) -> None:
    cfg = load_config().model_copy(update={"noise_source": "ast"})
    names = sorted(labels_of(split))
    scores, gaps, dbs = [], [], []
    for k, n in enumerate(names, 1):
        s, g, db = ast_scores(decode(ROOT / split / n, cfg.sample_rate), cfg)
        scores.append(s)
        gaps.append(g)
        dbs.append(db)
        print(f"  {k:>3}/{len(names)} {n}", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE / f"{split}.npz", names=np.array(names), scores=np.stack(scores), gap_s=np.array(gaps),
             gap_db=np.array(dbs))
    print("→", CACHE / f"{split}.npz")


def fit(split: str) -> None:
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler

    d = np.load(CACHE / f"{split}.npz")
    lab = labels_of(split)
    sev = np.array([lab[n]["background_noise_severity"] for n in d["names"]])
    X = np.stack([head_features(group_scores(s), g) for s, g in zip(d["scores"], d["gap_s"])])
    sc = StandardScaler().fit(X)
    clf = LogisticRegressionCV(Cs=10, cv=5, max_iter=5000, class_weight="balanced").fit(sc.transform(X), sev)
    pred = clf.predict(sc.transform(X))
    clean = sev == "none"
    extra = np.stack([group_scores(s, EXTRA)[[0, 2]].max(0) for s in d["scores"]])
    thr = {name: round(float(min(0.99, 1.2 * extra[clean, i].max())), 4) for i, name in enumerate(EXTRA)}
    ck = [list(GROUPS).index(k) for k in CLICKY]
    clicky = np.array([group_scores(s)[[0, 2]][:, ck].max() for s in d["scores"]])
    head = {"experiment": "E25", "fit_on": split, "model": load_config().noise_model, "groups": list(GROUPS),
            "classes": clf.classes_.tolist(), "lo": X.min(0).tolist(), "hi": X.max(0).tolist(), "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
            "coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist(), "C": float(clf.C_[0]),
            "extra_threshold": thr, "static_confirm": round(float(1.2 * clicky[clean].max()), 4),
            "gap_db_min": round(float(d["gap_db"].min()) - 3, 1),
            "gap_fill": np.median([group_scores(s)[2] for s in d["scores"][clean]], 0).round(4).tolist()}
    HEAD.write_text(json.dumps(head))
    print(f"C={head['C']:.3g} · tune severity acc {np.mean(pred == sev):.3f} · "
          f"tune present acc {np.mean((pred != 'none') == ~clean):.3f} · extra thresholds {thr}")
    print("→", HEAD)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("features").add_argument("--split", required=True)
    sub.add_parser("fit").add_argument("--tune", default="val_big")
    a = ap.parse_args()
    features(a.split) if a.cmd == "features" else fit(a.tune)
