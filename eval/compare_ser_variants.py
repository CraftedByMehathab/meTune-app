"""Run the shipped pipeline code (metune_pipeline.ser) over a split for several SER model sets and compare them:
call-level tone agreement with the first set, plus macro F1 / neutral kept vs labels. Used to (a) check that the
pipeline reproduces the eval numbers and (b) decide whether int8 exports can replace fp32 (val split only).

    uv run python eval/compare_ser_variants.py --split val_big --tau 0.3 --top 1.0 \\
        --set fp32 .data/models/ser_e20_bag{1..5}.onnx --set int8 .data/models/ser_e20_bag{1..5}.int8.onnx
"""
import argparse
import csv
import json
import time
from pathlib import Path

from metune_pipeline.config import load_config
from metune_pipeline.decode import decode
from metune_pipeline.schema import TONES
from metune_pipeline.ser import ser_facts
from sklearn.metrics import f1_score

ROOT = Path(".data/proxy")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val_big")
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--top", type=float, required=True)
    ap.add_argument("--set", nargs="+", action="append", required=True, metavar="NAME MODEL",
                    help="a name followed by its ONNX model paths; first set is the reference")
    a = ap.parse_args()
    labels = {r["name"]: json.loads(r["result_json"])["emotional_tone"]
              for r in csv.DictReader((ROOT / a.split / "labels.csv").open())}
    names = sorted(labels)
    audio = {n: decode(ROOT / a.split / n, 16000) for n in names}
    gold = [labels[n] for n in names]

    ref = None
    for name, *models in a.set:
        cfg = load_config().model_copy(update={"ser_models": models, "ser_tau": a.tau, "ser_top_fraction": a.top})
        t0 = time.perf_counter()
        pred = [ser_facts(audio[n], cfg)["tone"] for n in names]
        secs = time.perf_counter() - t0
        ref = ref or pred
        neu = [p for g, p in zip(gold, pred) if g == "neutral"]
        size = sum(Path(m).stat().st_size for m in models) / 1e6
        print(f"{name:<6} {len(models)} models {size:>5.0f} MB  macro F1 "
              f"{f1_score(gold, pred, labels=TONES, average='macro', zero_division=0):.3f}  "
              f"acc {sum(g == p for g, p in zip(gold, pred)) / len(gold):.3f}  neutral kept "
              f"{neu.count('neutral')}/{len(neu)}  same tone as {a.set[0][0]}: "
              f"{sum(r == p for r, p in zip(ref, pred))}/{len(pred)}  ({secs / len(names):.2f} s/call)", flush=True)


if __name__ == "__main__":
    main()
