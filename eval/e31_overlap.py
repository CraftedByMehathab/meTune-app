"""E31: speaker overlap on proxy splits with real overlaps (build_proxy.py --overlap-from-speech; labels = both
voices active ≥ 0.5 s). pyannote segmentation-3.0 (ONNX) vs Gemini 3.8 Flash (run_eval.py runs).

    uv run python eval/e31_overlap.py features --split val_big_agent_ovl
    uv run python eval/e31_overlap.py features --split test_big_agent_ovl
    uv run python eval/e31_overlap.py score --tune val_big_agent_ovl --test test_big_agent_ovl \
        --gemini-tune eval/runs/<val run> --gemini-test eval/runs/<test run>

pyannote rule: seconds with p(two speakers) > p ≥ θ; p and θ picked on --tune, applied once to --test.
Combinations (OR / AND with Gemini) are also picked on --tune only.
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from e23_noise_overlap import Segmenter, best_threshold, binary
from metune_pipeline.decode import decode

ROOT, CACHE, RUNS = Path(".data/proxy"), Path(".data/proxy/e31_cache"), Path("eval/runs")
FRAME_P = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def labels_of(split: str) -> dict:
    return {r["name"]: json.loads(r["result_json"])["speaker_overlap_present"]
            for r in csv.DictReader((ROOT / split / "labels.csv").open())}


def features(split: str) -> None:
    seg, names = Segmenter(), sorted(labels_of(split))
    out = {}
    for k, n in enumerate(names, 1):
        ov, step = seg.overlap_probs(decode(ROOT / split / n, 16000))
        out[n] = {f"s{p}": float((ov > p).sum() * step) for p in FRAME_P}
        print(f"  {k:>3}/{len(names)} {n} ov>0.5 {out[n]['s0.5']:.2f}s", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"{split}.json").write_text(json.dumps(out))


def gemini(run: str) -> dict:
    return {r["name"]: r["pred"] == "True" for r in csv.DictReader((Path(run) / "predictions.csv").open())
            if r["field"] == "speaker_overlap_present"}


def score(a) -> None:
    tr, te = json.loads((CACHE / f"{a.tune}.json").read_text()), json.loads((CACHE / f"{a.test}.json").read_text())
    ytr, yte = labels_of(a.tune), labels_of(a.test)
    ntr, nte = sorted(tr), sorted(te)
    y_tr, y_te = np.array([ytr[n] for n in ntr]), [yte[n] for n in nte]

    best = None
    for p in FRAME_P:
        s = np.array([tr[n][f"s{p}"] for n in ntr])
        t = best_threshold(s, y_tr)
        acc = ((s >= t) == y_tr).mean()
        if best is None or acc > best[0]:
            best = (acc, p, t)
    _, p, t = best
    py_tr = [tr[n][f"s{p}"] >= t for n in ntr]
    py_te = [te[n][f"s{p}"] >= t for n in nte]
    g_tr, g_te = gemini(a.gemini_tune), gemini(a.gemini_test)
    gm_tr, gm_te = [g_tr[n] for n in ntr], [g_te[n] for n in nte]

    combos = {"pyannote": (py_tr, py_te), "gemini": (gm_tr, gm_te),
              "pyannote OR gemini": ([x or y for x, y in zip(py_tr, gm_tr)], [x or y for x, y in zip(py_te, gm_te)]),
              "pyannote AND gemini": ([x and y for x, y in zip(py_tr, gm_tr)], [x and y for x, y in zip(py_te, gm_te)])}
    res = {"tune": a.tune, "test": a.test, "when": datetime.now().isoformat(timespec="seconds"),
           "pyannote_rule": {"frame_p": p, "min_seconds": round(t, 2)},
           "tune_scores": {k: binary(list(y_tr), v[0]) for k, v in combos.items()}}
    pick = max(combos, key=lambda k: (res["tune_scores"][k]["acc"], k == "pyannote"))   # ties → no API needed
    res["picked_on_tune"] = pick
    res["test_scores"] = {k: binary(y_te, v[1]) for k, v in combos.items()}
    out = RUNS / f"{datetime.now():%Y%m%d-%H%M}_E31-overlap-{a.test}"
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float))
    print("→", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("features").add_argument("--split", required=True)
    s = sub.add_parser("score")
    for k in ("--tune", "--test", "--gemini-tune", "--gemini-test"):
        s.add_argument(k, required=True)
    a = ap.parse_args()
    features(a.split) if a.cmd == "features" else score(a)
