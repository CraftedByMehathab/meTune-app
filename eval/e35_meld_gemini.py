"""E35: Gemini's tone on MELD, alone and fused with the SER bag. CREMA-D sentences are neutral in wording by design,
which made audio LLMs look useless for tone (E6–E15); MELD's emotion is in words and voice, like real calls.

    uv run --env-file .env python eval/e35_meld_gemini.py votes --split dev     (then --split test)
    uv run python eval/e35_meld_gemini.py score

Same prompt / model / thinking as the app (config.yaml), 1 vote per clip; SER probabilities from E34's cache.
Fusion: tone = argmax log p_SER + shipped bias + g · [Gemini said k], g picked on MELD dev; test scored once.
"""
import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

import numpy as np

OUT, RUNS = Path(".data/meld"), Path("eval/runs")
FACT_KEYS = ["longest_gap_s", "speech_ratio", "clip_pct", "spikes_per_min", "snr_proxy_db"]


async def votes(split: str, concurrency: int = 8) -> None:
    from metune_pipeline.analyze import local_facts
    from metune_pipeline.config import load_config
    from metune_pipeline.decode import decode, encode_flac
    from metune_pipeline.llm import make_client, run_votes

    cfg = load_config().model_copy(update={"llm_votes": 1, "llm_temperatures": [0.0]})
    client = make_client()
    assert client, "GEMINI_API_KEY not set: run with --env-file .env"
    feats = json.loads((OUT / f"{split}_features.json").read_text())
    path = OUT / f"{split}_gemini.json"
    done = json.loads(path.read_text()) if path.exists() else {}         # resumable
    todo = [n for n in feats if n not in done]
    sem, cost = asyncio.Semaphore(concurrency), [0.0]

    async def one(name: str):
        async with sem:
            x = decode(OUT / split / name, cfg.sample_rate)
            facts = local_facts(x, cfg)
            calls = await run_votes(client, encode_flac(x, cfg.sample_rate), "audio/flac",
                                    {k: facts[k] for k in FACT_KEYS}, cfg)
        c = calls[0]
        cost[0] += c.cost_usd
        done[name] = {"tone": c.output["emotional_tone"] if c.output else None, "error": c.error,
                      "heard": (c.output or {}).get("heard", "")}
        if len(done) % 100 == 0:
            path.write_text(json.dumps(done))
            print(f"  {len(done)}/{len(feats)} · ${cost[0]:.3f}", flush=True)

    await asyncio.gather(*(one(n) for n in todo))
    path.write_text(json.dumps(done))
    print(f"{split}: {len(done)} clips · errors {sum(1 for v in done.values() if v['error'])} · this run ${cost[0]:.3f}")


def score(_) -> None:
    import sys
    sys.path.insert(0, "eval")
    from e33_tone_bias import report
    from metune_pipeline.config import load_config
    from metune_pipeline.schema import TONES

    b = np.array([load_config().ser_bias[t] for t in TONES])

    def load(split):
        f = json.loads((OUT / f"{split}_features.json").read_text())
        g = json.loads((OUT / f"{split}_gemini.json").read_text())
        names = [n for n in f if g.get(n, {}).get("tone")]
        return (np.array([f[n]["probs"] for n in names]), np.array([f[n]["label"] for n in names]),
                np.array([g[n]["tone"] for n in names]), len(f) - len(names))

    def fuse(P, G, g):
        onehot = np.array([[t == x for t in TONES] for x in G], float)
        return np.array(TONES)[(np.log(np.clip(P, 1e-6, 1)) + b + g * onehot).argmax(1)]

    P, y, G, miss = load("dev")
    Pt, yt, Gt, miss_t = load("test")
    from sklearn.metrics import f1_score
    grid = [0.0, 0.5, 1, 1.5, 2, 3, 4, 6, 10]
    dev_f1 = {g: round(f1_score(y, fuse(P, G, g), labels=TONES, average="macro", zero_division=0), 3) for g in grid}
    best = max(dev_f1.values())
    g = min(k for k, v in dev_f1.items() if v == best)
    res = {"n_dev": len(y), "n_test": len(yt), "dropped_no_gemini_answer": {"dev": miss, "test": miss_t},
           "fusion_weight_dev_macro_f1": dev_f1, "picked_g": g,
           "test": {"always neutral": report(yt, np.array(["neutral"] * len(yt))),
                    "SER v0.5.0": report(yt, fuse(Pt, Gt, 0.0)),
                    "Gemini 3.8 Flash": report(yt, Gt),
                    f"fusion g={g}": report(yt, fuse(Pt, Gt, g))},
           "when": datetime.now().isoformat(timespec="seconds")}
    out = RUNS / f"{datetime.now():%Y%m%d-%H%M}_E35-meld-gemini"
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "test"}, indent=1))
    for k, v in res["test"].items():
        print(f"{k:<20} macro F1 {v['macro_f1']:.3f} · acc {v['accuracy']:.3f} · recall {v['recall']}")
    print("→", out)
    from track import log_run
    log_run(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("votes").add_argument("--split", choices=["dev", "test"], required=True)
    sub.add_parser("score")
    a = ap.parse_args()
    asyncio.run(votes(a.split)) if a.cmd == "votes" else score(a)
