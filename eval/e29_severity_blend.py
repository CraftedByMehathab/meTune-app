"""E29: blend AST's severity probabilities with Gemini's severity vote. Presence stays AST's (E22–E26: OR/AND gain
nothing); only the low / medium / high choice on calls AST calls noisy is blended:
    severity = argmax over low/medium/high of  w · p_AST + (1 − w) · [Gemini said it]

    uv run python eval/e29_severity_blend.py --tune-run eval/runs/<E29 val_big> --test-run eval/runs/<E26 test_big>

AST probabilities on val_big are 5-fold cross-validated (the head was fit on val_big); on test_big they come from
the shipped head. w is picked on val_big; test_big is scored once. No API calls.
"""
import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
from metune_pipeline.noise import group_scores, head, head_features
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT, CACHE = Path(".data/proxy"), Path(".data/proxy/e25_cache")
LEVELS = ["low", "medium", "high"]


def labels(split: str) -> dict:
    return {r["name"]: json.loads(r["result_json"]) for r in csv.DictReader((ROOT / split / "labels.csv").open())}


def gemini(run: Path) -> dict:
    out = {}
    for r in csv.DictReader((run / "predictions.csv").open()):
        if r["field"] == "vote0":
            v = json.loads(r["label"] or r["pred"])
            out[r["name"]] = v["background_noise_severity"] if v.get("background_noise_present") else "none"
    return out


def feats(split: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(CACHE / f"{split}.npz")
    return d["names"], np.stack([head_features(group_scores(s), g) for s, g in zip(d["scores"], d["gap_s"])])


def blend(p: np.ndarray, classes: list[str], g: list[str], w: float) -> list[str]:
    out = []
    for row, gs in zip(p, g):
        sev = classes[int(row.argmax())]
        if sev != "none":
            score = [w * row[classes.index(s)] + (1 - w) * (gs == s) for s in LEVELS]
            sev = LEVELS[int(np.argmax(score))]
        out.append(sev)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune-run", required=True)
    ap.add_argument("--test-run", required=True)
    a = ap.parse_args()

    names, X = feats("val_big")
    lab = labels("val_big")
    y = np.array([lab[n]["background_noise_severity"] for n in names])
    pipe = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=10, cv=5, max_iter=5000, class_weight="balanced"))
    p_tr = cross_val_predict(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")
    classes = sorted(set(y))
    g_tr = [gemini(Path(a.tune_run))[n] for n in names]
    grid = np.round(np.arange(0, 1.01, 0.05), 2)
    accs = [np.mean(np.array(blend(p_tr, classes, g_tr, w)) == y) for w in grid]
    best = max(accs)
    w = float(max(w for w, s in zip(grid, accs) if s == best))     # ties → more AST weight
    print("val_big severity acc by w (1 = AST only):", dict(zip(grid.tolist(), np.round(accs, 3).tolist())))
    print(f"→ w = {w} (val_big {best:.3f}; AST only {accs[-1]:.3f}; Gemini-led w=0 {accs[0]:.3f})")

    h = head()
    names_te, X_te = feats("test_big")
    lab_te = labels("test_big")
    y_te = np.array([lab_te[n]["background_noise_severity"] for n in names_te])
    z = (np.clip(X_te, h["lo"], h["hi"]) - np.array(h["mean"])) / np.array(h["scale"])
    logits = z @ np.array(h["coef"]).T + np.array(h["intercept"])
    p_te = np.exp(logits - logits.max(1, keepdims=True))
    p_te /= p_te.sum(1, keepdims=True)
    g_te = [gemini(Path(a.test_run))[n] for n in names_te]
    for name, ww in (("AST only", 1.0), (f"blend w={w}", w)):
        pred = np.array(blend(p_te, h["classes"], g_te, ww))
        noisy = y_te != "none"
        print(f"test_big {name:<12} severity acc {np.mean(pred == y_te):.3f} · on noisy calls {np.mean(pred[noisy] == y_te[noisy]):.3f}")


if __name__ == "__main__":
    main()
