"""metune run <folder> --out results.csv"""
import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from .analyze import analyze_file
from .config import load_config
from .decode import AUDIO_EXTS
from .llm import make_client


async def _run(folder: Path, out: Path, config: str | None, concurrency: int, use_llm: bool) -> int:
    cfg = load_config(config)
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    if not files:
        print(f"No audio files in {folder}", file=sys.stderr)
        return 1
    client = make_client() if use_llm else None
    print(f"{len(files)} files · model={cfg.llm_model if client else 'none (DSP baseline)'} · k={cfg.llm_votes}")
    sem = asyncio.Semaphore(concurrency)

    async def one(p: Path):
        async with sem:
            try:
                a = await analyze_file(p, cfg, client)
                print(f"  ✓ {p.name}: {a.result.emotional_tone}, noise={a.result.background_noise_present}"
                      f"{' (degraded)' if a.degraded else ''}")
                return p, a, None
            except Exception as e:  # one bad file never stops the batch
                print(f"  ✗ {p.name}: {e}", file=sys.stderr)
                return p, None, str(e)

    rows = await asyncio.gather(*(one(p) for p in files))
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "result_json"])
        for p, a, _ in rows:
            if a:
                w.writerow([p.name, json.dumps(a.result.model_dump())])

    ok = [a for _, a, _ in rows if a]
    minutes = sum(a.duration_s for a in ok) / 60 or 1e-9
    cost = sum(a.cost_usd for a in ok)
    llm_s = sum(a.timings_ms.get("llm", 0) for a in ok) / 1000
    print(f"\n{len(ok)}/{len(files)} ok → {out}")
    print(f"audio {minutes:.2f} min · cost ${cost:.5f} (${cost / minutes:.5f}/min) · LLM {llm_s / minutes:.1f} s/min")
    return 0 if len(ok) == len(files) else 2


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="metune")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="analyze every audio file in a folder")
    run.add_argument("folder", type=Path)
    run.add_argument("--out", type=Path, default=Path("results.csv"))
    run.add_argument("--config", default=None)
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--no-llm", action="store_true", help="DSP baseline only")
    args = ap.parse_args(argv)
    sys.exit(asyncio.run(_run(args.folder, args.out, args.config, args.concurrency, not args.no_llm)))


if __name__ == "__main__":
    main()
