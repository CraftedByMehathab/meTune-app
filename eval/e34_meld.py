"""E34: does the tone model transfer to conversational speech? MELD (Friends dialogue, declare-lab, GPL-3.0; cite
Poria et al. ACL 2019) as an outside check: CREMA-D is acted read sentences, the trial calls were 0/3 on tone.

    uv run python eval/e34_meld.py prep --split dev          # mp4 → phone-channel ogg + labels.csv (also: test)
    uv run python eval/e34_meld.py features --split dev      # SER bag probabilities + confidence signals per clip
    uv run python eval/e34_meld.py score                     # rules on MELD test; bias refit on MELD dev

Fixed before looking at results: each utterance is one "call" through the same phone channel as the proxy set
(300–3400 Hz, 8 kHz, Opus); mapping neutral→neutral, joy→satisfied, anger→upset, fear→distressed,
disgust→frustrated (sadness, surprise dropped, as CREMA-D SAD was); clips < 1.5 s dropped. MELD dev = tune,
MELD test = scored once. Data stays in .data/meld (gitignored, research use only, never redistributed).
"""
import argparse
import csv
import json
import subprocess
import tarfile
from datetime import datetime
from pathlib import Path

import numpy as np

RAW, OUT, RUNS = Path(".data/meld/raw"), Path(".data/meld"), Path("eval/runs")
TONE_OF = {"neutral": "neutral", "joy": "satisfied", "anger": "upset", "fear": "distressed", "disgust": "frustrated"}
ARCHIVE = {"dev": ("dev.tar.gz", "dev_sent_emo.csv", "dev_splits_complete"),
           "test": ("test.tar.gz", "test_sent_emo.csv", "output_repeated_splits_test")}
MIN_S = 1.5


def secs(t: str) -> float:
    h, m, s = t.replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def prep(split: str) -> None:
    inner, csv_name, _ = ARCHIVE[split]
    root = RAW / "MELD.Raw"
    if not (root / inner).exists():
        with tarfile.open(RAW / "MELD.Raw.tar.gz") as t:
            members = [m for m in t.getmembers() if m.name.endswith((inner, csv_name)) and m.isfile()]
            t.extractall(RAW, members=members, filter="data")
    vids = RAW / f"{split}_videos"
    if not vids.exists():
        with tarfile.open(root / inner) as t:
            t.extractall(vids, filter="data")
    mp4 = {p.name: p for p in vids.rglob("*.mp4") if not p.name.startswith("._")}
    dst = OUT / split
    dst.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], {"unmapped": 0, "short": 0, "missing": 0, "decode": 0}
    for r in csv.DictReader((root / csv_name).open(encoding="utf-8", errors="replace")):
        tone = TONE_OF.get(r["Emotion"].strip().lower())
        if not tone:
            skipped["unmapped"] += 1
            continue
        if secs(r["EndTime"]) - secs(r["StartTime"]) < MIN_S:
            skipped["short"] += 1
            continue
        src = mp4.get(f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}.mp4")
        if not src:
            skipped["missing"] += 1
            continue
        name = f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}.ogg"
        if not (dst / name).exists():
            p = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1",
                                "-af", "highpass=f=300,lowpass=f=3400,aresample=8000", "-c:a", "libopus", "-b:a", "24k",
                                str(dst / name)], capture_output=True)
            if p.returncode:
                skipped["decode"] += 1
                continue
        rows.append([name, tone, r["Speaker"], r["Emotion"]])
    with (dst / "labels.csv").open("w", newline="") as f:
        csv.writer(f).writerows([["name", "tone", "speaker", "meld_emotion"], *rows])
    counts = {t: sum(r[1] == t for r in rows) for t in TONE_OF.values()}
    print(f"{split}: {len(rows)} clips {counts} · skipped {skipped}")


def features(split: str) -> None:
    from metune_pipeline.analyze import local_facts
    from metune_pipeline.confidence import calibrated
    from metune_pipeline.config import load_config
    from metune_pipeline.decode import decode
    from metune_pipeline.schema import TONES
    from metune_pipeline.ser import ser_facts

    cfg = load_config()
    rows = list(csv.DictReader((OUT / split / "labels.csv").open()))
    out = {}
    for k, r in enumerate(rows, 1):
        x = decode(OUT / split / r["name"], cfg.sample_rate)
        s, sr = ser_facts(x, cfg), local_facts(x, cfg)["speech_ratio"]
        out[r["name"]] = {"probs": [s["probs"][t] for t in TONES], "label": r["tone"], "tone": s["tone"],
                          "windows": s["windows"], "confidence": calibrated(s, sr)}
        if k % 100 == 0:
            print(f"  {k}/{len(rows)}", flush=True)
    (OUT / f"{split}_features.json").write_text(json.dumps(out))


def score(_) -> None:
    import sys
    sys.path.insert(0, "eval")
    from e33_tone_bias import fit_bias, predict, report, tau_rule
    from metune_pipeline.config import load_config
    from metune_pipeline.schema import TONES

    cfg = load_config()
    dev, test = (json.loads((OUT / f"{s}_features.json").read_text()) for s in ("dev", "test"))
    P, y = np.array([r["probs"] for r in dev.values()]), np.array([r["label"] for r in dev.values()])
    Pt, yt = np.array([r["probs"] for r in test.values()]), np.array([r["label"] for r in test.values()])
    b_meld = fit_bias(P, y)
    shipped = np.array([cfg.ser_bias[t] for t in TONES])
    conf = np.array([r["confidence"] for r in test.values()])
    right = np.array([r["tone"] == r["label"] for r in test.values()], float)
    idx = np.minimum((conf * 10).astype(int), 9)
    ece = float(sum(abs(conf[idx == i].mean() - right[idx == i].mean()) * (idx == i).mean()
                    for i in range(10) if (idx == i).any()))
    majority = np.array(["neutral"] * len(yt))
    res = {"n_dev": len(y), "n_test": len(yt), "test_class_counts": {t: int((yt == t).sum()) for t in TONES},
           "bias_refit_on_meld_dev": dict(zip(TONES, b_meld.round(2).tolist())),
           "test": {"always neutral": report(yt, majority), "tau rule (v0.4.0)": report(yt, tau_rule(Pt, cfg.ser_tau)),
                    "shipped bias (v0.5.0, CREMA-D tuned)": report(yt, predict(Pt, shipped)),
                    "bias refit on MELD dev": report(yt, predict(Pt, b_meld))},
           "confidence_v0.5.0_on_test": {"ece": round(ece, 3), "mean_conf": round(float(conf.mean()), 3),
                                         "accuracy": round(float(right.mean()), 3),
                                         "wrong_at_0.9+": f"{int(((conf >= 0.9) & (right == 0)).sum())}/{int((right == 0).sum())}"},
           "when": datetime.now().isoformat(timespec="seconds")}
    out = RUNS / f"{datetime.now():%Y%m%d-%H%M}_E34-meld"
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "test"}, indent=1))
    for k, v in res["test"].items():
        print(f"{k:<40} macro F1 {v['macro_f1']:.3f} · acc {v['accuracy']:.3f} · recall {v['recall']}")
    print("→", out)
    from track import log_run
    log_run(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("prep", "features"):
        sub.add_parser(c).add_argument("--split", choices=["dev", "test"], required=True)
    sub.add_parser("score")
    a = ap.parse_args()
    {"prep": lambda: prep(a.split), "features": lambda: features(a.split), "score": lambda: score(a)}[a.cmd]()
