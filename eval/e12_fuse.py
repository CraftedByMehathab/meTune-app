"""E12 step 2: tune an SER → tone mapping on dev, apply it unchanged to test, fuse with stored LLM votes.

No API calls: LLM tones come from the stored v2 runs (E7 on dev, T1 on test; same frozen config as the app).

    uv run --group ser python eval/e12_fuse.py --model superb/hubert-large-superb-er
"""
import argparse
import csv
import glob
import itertools
import json
from datetime import datetime
from pathlib import Path

from metune_pipeline.schema import TONES
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(".data/proxy")
LLM_RUNS = {"dev": "*_E7-v2", "test": "*_T1-v2-test"}
TO_TONE = {"hap": "satisfied", "sad": "distressed"}       # "ang" → frustrated | upset by strength


def labels(split: str) -> dict[str, str]:
    return {r["name"]: json.loads(r["result_json"])["emotional_tone"] for r in csv.DictReader((ROOT / split / "labels.csv").open())}


def llm_tones(split: str) -> dict[str, str]:
    path = glob.glob(f"eval/runs/{LLM_RUNS[split]}/predictions.csv")[0]
    return {r["name"]: r["pred"] for r in csv.DictReader(open(path)) if r["field"] == "emotional_tone"}


def aggregate(windows: list[dict], top: float) -> dict[str, float]:
    """Duration-weighted mean probabilities over the most emotional `top` fraction of windows."""
    if not windows:
        return {"neu": 1.0, "hap": 0.0, "ang": 0.0, "sad": 0.0}
    ranked = sorted(windows, key=lambda w: w["p"]["neu"])[:max(1, round(len(windows) * top))]
    total = sum(w["dur"] for w in ranked)
    return {k: sum(w["p"][k] * w["dur"] for w in ranked) / total for k in ranked[0]["p"]}


def ser_tone(p: dict[str, float], tau: float, tau_up: float) -> tuple[str, float]:
    """(tone, emotional score). Neutral unless the non-neutral mass reaches `tau`."""
    score = 1 - p["neu"]
    if score < tau:
        return "neutral", score
    emo = max(("hap", "ang", "sad"), key=p.get)
    if emo == "ang":
        return ("upset" if p["ang"] >= tau_up else "frustrated"), score
    return TO_TONE[emo], score


def metrics(y: list[str], p: list[str]) -> dict:
    return {"accuracy": round(sum(a == b for a, b in zip(y, p)) / len(y), 3),
            "macro_f1": round(float(f1_score(y, p, labels=TONES, average="macro", zero_division=0)), 3),
            "confusion": confusion_matrix(y, p, labels=TONES).tolist()}


def predict(feats: dict, names: list[str], llm: dict, top: float, tau: float, tau_up: float):
    ser, fused = [], []
    for n in names:
        tone, _ = ser_tone(aggregate(feats[n], top), tau, tau_up)
        ser.append(tone)
        fused.append(tone if tone != "neutral" else llm[n])   # SER overrides only when it hears emotion
    return ser, fused


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="superb/hubert-large-superb-er")
    a = ap.parse_args()
    fdir = ROOT / "ser" / a.model.replace("/", "__")
    feats = {s: json.loads((fdir / f"{s}.json").read_text()) for s in ("dev", "test")}

    data = {}
    for s in ("dev", "test"):
        y = labels(s)
        names = sorted(y)
        assert set(names) <= set(feats[s]), f"missing SER features for {s}"
        data[s] = (names, [y[n] for n in names], llm_tones(s))

    # Grid search on dev only; objective = macro F1 of the *fused* tone (what would ship).
    names, y, llm = data["dev"]
    grid = itertools.product([1.0, 0.5, 0.3], [i / 20 for i in range(2, 19)], [i / 20 for i in range(4, 20)])
    best = max(grid, key=lambda g: (f1_score(y, predict(feats["dev"], names, llm, *g)[1], labels=TONES,
                                             average="macro", zero_division=0), -g[1]))
    top, tau, tau_up = best

    out = {"name": f"E12-{a.model.split('/')[-1]}", "model": a.model, "when": datetime.now().isoformat(timespec="seconds"),
           "tuned_on": "dev", "params": {"top_fraction": top, "tau_emotional": tau, "tau_upset": tau_up}, "results": {}}
    rows = []
    for s, (names, y, llm) in data.items():
        ser, fused = predict(feats[s], names, llm, top, tau, tau_up)
        out["results"][s] = {"llm_only": metrics(y, [llm[n] for n in names]),
                             "ser_only": metrics(y, ser), "fused": metrics(y, fused)}
        rows += [[s, n, t, llm[n], a_, b_] for n, t, a_, b_ in zip(names, y, ser, fused)]

    run = Path("eval/runs") / f"{datetime.now():%Y%m%d-%H%M}_{out['name']}"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps(out, indent=2))
    with (run / "predictions.csv").open("w", newline="") as f:
        csv.writer(f).writerows([["split", "name", "label", "llm", "ser", "fused"], *rows])

    print(f"{out['name']}  params (tuned on dev): top={top} tau={tau} tau_upset={tau_up}")
    for s in ("dev", "test"):
        r = out["results"][s]
        print(f"  {s:<5} " + "  ".join(f"{k}: acc {v['accuracy']:.3f} F1 {v['macro_f1']:.3f}" for k, v in r.items()))
    print("  test fused confusion (rows=true):", TONES)
    for t, row in zip(TONES, out["results"]["test"]["fused"]["confusion"]):
        print(f"    {t:<11}{row}")
    print("→", run)


if __name__ == "__main__":
    main()
