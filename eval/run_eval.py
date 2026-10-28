"""Run the pipeline on a proxy split and log the run for later review (Experiment Log in the vault).

    uv run --env-file .env python eval/run_eval.py --split dev --name E6-v1
    uv run --env-file .env python eval/run_eval.py --split dev --name E7-v2 --set llm_prompt=v2

Writes eval/runs/<timestamp>_<name>/: summary.json (config + hash, metrics, cost, latency) and
predictions.csv (label vs prediction per call, every vote, and the model's `heard` text).
Leakage rule (DR-008): tune on `dev` only; run `test` once per frozen config; trial calls last.
"""
import argparse
import asyncio
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import yaml
from metune_api.scoring import score
from metune_pipeline import config_hash
from metune_pipeline.analyze import analyze_file
from metune_pipeline.config import load_config
from metune_pipeline.llm import make_client
from metune_pipeline.schema import SCORED_FIELDS

ROOT = Path(".data/proxy")
RUNS = Path("eval/runs")
DEFAULTS = {"emotional_tone": "neutral", "background_noise_present": False, "speaker_overlap_present": False}


def binary(pairs: list[tuple[dict, dict]], field: str) -> dict:
    """Precision / recall for a yes/no field: accuracy hides a model that always says "no"."""
    tp = sum(t[field] and p[field] for t, p in pairs)
    fp = sum(not t[field] and p[field] for t, p in pairs)
    fn = sum(t[field] and not p[field] for t, p in pairs)
    return {"precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None, "positives": tp + fn}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", help="dev | test | path to a folder with labels.csv")
    ap.add_argument("--name", required=True, help="experiment id, e.g. E7-v2")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--no-llm", action="store_true", help="no API calls: local models + DSP baseline only")
    a = ap.parse_args()

    overrides = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in a.set)}
    cfg = load_config().model_copy(update=overrides)
    folder = ROOT / a.split if (ROOT / a.split).exists() else Path(a.split)
    labels = {r["name"]: json.loads(r["result_json"]) for r in csv.DictReader((folder / "labels.csv").open())}
    client = None if a.no_llm else make_client()
    assert client or a.no_llm, "GEMINI_API_KEY not set: run with --env-file .env"

    sem = asyncio.Semaphore(a.concurrency)
    finished = 0
    print(f"{a.name}: {len(labels)} calls from {folder}", flush=True)

    async def one(name: str):
        nonlocal finished
        async with sem:
            an = await analyze_file(folder / name, cfg, client)
        finished += 1
        errs = sum(1 for c in an.calls if c.error)
        print(f"  {finished:>3}/{len(labels)} {name} {an.result.emotional_tone}"
              f"{f'  ⚠ {errs} vote errors' if errs else ''}", flush=True)
        return name, an

    t0 = time.perf_counter()
    done = await asyncio.gather(*(one(n) for n in sorted(labels)))
    wall = time.perf_counter() - t0

    pairs = [(labels[n], an.result.model_dump()) for n, an in done]
    metrics = score(pairs).model_dump()
    emotional = [(t, p) for t, p in pairs if t["emotional_tone"] != "neutral"]
    minutes = sum(an.duration_s for _, an in done) / 60
    cost = sum(an.cost_usd for _, an in done)
    conf_right = [p["confidence"] for t, p in pairs if p["emotional_tone"] == t["emotional_tone"]]
    conf_wrong = [p["confidence"] for t, p in pairs if p["emotional_tone"] != t["emotional_tone"]]
    summary = {
        "name": a.name, "split": a.split, "n": len(pairs), "no_llm": a.no_llm, "when": datetime.now().isoformat(timespec="seconds"),
        "config_hash": config_hash(cfg), "overrides": overrides, "config": cfg.model_dump(),
        "accuracy": {f: metrics["per_field"][f]["accuracy"] for f in SCORED_FIELDS},
        "macro_f1": {f: metrics["per_field"][f]["macro_f1"] for f in SCORED_FIELDS},
        "intensity_accuracy_non_neutral": round(sum(t["emotional_intensity"] == p["emotional_intensity"]
                                                    for t, p in emotional) / max(1, len(emotional)), 3),
        "noise_present": binary(pairs, "background_noise_present"),
        "overlap": binary(pairs, "speaker_overlap_present"),
        "all_default_answers": sum(all(p[k] == v for k, v in DEFAULTS.items()) for _, p in pairs),
        "confidence_mean_when_tone_right": round(sum(conf_right) / len(conf_right), 3) if conf_right else None,
        "confidence_mean_when_tone_wrong": round(sum(conf_wrong) / len(conf_wrong), 3) if conf_wrong else None,
        "tone_report": metrics["tone_report"], "tone_confusion": metrics["tone_confusion"],
        "audio_min": round(minutes, 2), "cost_usd": round(cost, 5), "cost_per_min": round(cost / minutes, 5),
        "llm_errors": sum(1 for _, an in done for c in an.calls if c.error),
        "wall_s": round(wall, 1),
    }

    out = RUNS / f"{datetime.now():%Y%m%d-%H%M}_{a.name}"
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out / "predictions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "field", "label", "pred", "match"])
        for (n, an), (t, p) in zip(done, pairs):
            for k in [*SCORED_FIELDS, "background_noise_type", "confidence"]:
                w.writerow([n, k, t.get(k), p[k], t.get(k) == p[k] if k in SCORED_FIELDS else ""])
            for c in an.calls:
                w.writerow([n, f"vote{c.vote}", "", json.dumps(c.output if c.output else {"error": c.error}), ""])

    print(json.dumps({k: summary[k] for k in ["name", "config_hash", "accuracy", "intensity_accuracy_non_neutral",
                                              "noise_present", "overlap", "all_default_answers",
                                              "confidence_mean_when_tone_right", "confidence_mean_when_tone_wrong",
                                              "cost_per_min", "llm_errors"]}, indent=1))
    print("tone confusion (rows=true):", summary["tone_confusion"]["labels"])
    for lbl, row in zip(summary["tone_confusion"]["labels"], summary["tone_confusion"]["matrix"]):
        print(f"  {lbl:<11}{row}")
    print("→", out)


if __name__ == "__main__":
    asyncio.run(main())
