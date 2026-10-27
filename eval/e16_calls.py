"""E16 step 2: score the fine-tuned 5-tone SER model on whole calls (compare with E12 on the same test calls).

Window probabilities come from ser_features.py run with the fine-tuned model folder. Call rule (E12, 5 tones):
duration-weighted mean over the most emotional `top` fraction of windows → neutral unless 1 − p(neutral) ≥ τ,
else the strongest non-neutral tone. `top` and τ are tuned on the **val** calls (actors 1036–1045, held out
from fine-tuning) and applied once to **test**. Dev calls share actors with training → not used.

    uv run --group ser python eval/build_proxy.py --val
    uv run --group ser python eval/ser_features.py --model .data/proxy/ser_ft/E16-ft/model --splits val test
    uv run --group ser python eval/e16_calls.py --model .data/proxy/ser_ft/E16-ft/model
"""
import argparse
import csv
import glob
import itertools
import json
from datetime import datetime
from pathlib import Path

from e12_fuse import labels, llm_tones, metrics
from metune_pipeline.schema import TONES
from sklearn.metrics import f1_score

ROOT = Path(".data/proxy")
EMOTIONS = [t for t in TONES if t != "neutral"]


def aggregate(windows: list[dict], top: float) -> dict[str, float]:
    if not windows:
        return {t: float(t == "neutral") for t in TONES}
    ranked = sorted(windows, key=lambda w: w["p"]["neutral"])[:max(1, round(len(windows) * top))]
    total = sum(w["dur"] for w in ranked)
    return {t: sum(w["p"][t] * w["dur"] for w in ranked) / total for t in TONES}


def call_tone(p: dict[str, float], tau: float) -> str:
    return "neutral" if 1 - p["neutral"] < tau else max(EMOTIONS, key=p.get)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="fine-tuned model folder (as passed to ser_features.py)")
    ap.add_argument("--name", default="E16-calls")
    a = ap.parse_args()
    fdir = ROOT / "ser" / a.model.replace("/", "__")
    feats = {s: json.loads((fdir / f"{s}.json").read_text()) for s in ("val", "test")}
    y = {s: labels(s) for s in ("val", "test")}

    def predict(split: str, top: float, tau: float) -> list[str]:
        return [call_tone(aggregate(feats[split][n], top), tau) for n in sorted(y[split])]

    gold = {s: [y[s][n] for n in sorted(y[s])] for s in y}
    grid = itertools.product([1.0, 0.5, 0.3], [i / 20 for i in range(1, 20)])
    # Tie-break toward the higher τ: fewer neutral false alarms on mostly-neutral real traffic (E12 lesson).
    top, tau = max(grid, key=lambda g: (f1_score(gold["val"], predict("val", *g), labels=TONES, average="macro",
                                                 zero_division=0), g[1]))

    test_names = sorted(y["test"])
    llm = llm_tones("test")
    e12 = json.loads(Path(max(glob.glob("eval/runs/*_E12-wav2vec2-base-superb-er/summary.json"))).read_text())
    preds = {s: predict(s, top, tau) for s in ("val", "test")}
    out = {"name": a.name, "model": a.model, "when": datetime.now().isoformat(timespec="seconds"),
           "tuned_on": "val (actors 1036–1045)", "params": {"top_fraction": top, "tau_emotional": tau},
           "results": {"val": {"ser_ft": metrics(gold["val"], preds["val"])},
                       "test": {"ser_ft": metrics(gold["test"], preds["test"]),
                                "llm_only": metrics(gold["test"], [llm[n] for n in test_names]),
                                "e12b_ser_only": e12["results"]["test"]["ser_only"]}}}
    neu = [p for g, p in zip(gold["test"], preds["test"]) if g == "neutral"]
    out["results"]["test"]["neutral_kept"] = f"{neu.count('neutral')}/{len(neu)}"

    run = Path("eval/runs") / f"{datetime.now():%Y%m%d-%H%M}_{a.name}"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps(out, indent=2))
    with (run / "predictions.csv").open("w", newline="") as f:
        csv.writer(f).writerows([["split", "name", "label", "ser_ft", "llm"],
                                 *[["val", n, t, p, ""] for n, t, p in zip(sorted(y["val"]), gold["val"], preds["val"])],
                                 *[["test", n, t, p, llm[n]] for n, t, p in zip(test_names, gold["test"], preds["test"])]])

    print(f"{a.name}  params (tuned on val): top={top} tau={tau}")
    for s, r in out["results"].items():
        print(f"  {s:<5} " + "  ".join(f"{k}: acc {v['accuracy']:.3f} F1 {v['macro_f1']:.3f}"
                                       for k, v in r.items() if isinstance(v, dict)))
    print("  test neutral kept:", out["results"]["test"]["neutral_kept"])
    print("  test confusion (rows=true):", TONES)
    for t, row in zip(TONES, out["results"]["test"]["ser_ft"]["confusion"]):
        print(f"    {t:<11}{row}")
    print("→", run)


if __name__ == "__main__":
    main()
