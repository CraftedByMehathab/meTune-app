"""Window cache for the shipped ONNX SER models (ensemble = mean of probs), same format as ser_features.py
({file: [{"t", "dur", "hi", "p": {tone: prob}}]}), so e16_calls.py can tune/score it offline — e.g. when only
the ONNX exports are at hand (E21). Uses the pipeline's own windows, high-band share and sessions.

    uv run python eval/onnx_features.py --name bag5int8 --splits val_big_agent test_big_agent \\
        --models .data/models/ser_e20_bag1.int8.onnx ... .data/models/ser_e20_bag5.int8.onnx
    uv run python eval/e16_calls.py --model bag5int8 --val-split val_big_agent --test-split none --customer-only 3e-4
"""
import argparse
import json
from pathlib import Path

import numpy as np
from metune_pipeline.config import load_config
from metune_pipeline.decode import decode
from metune_pipeline.schema import TONES
from metune_pipeline.ser import _session, high_band_share, speech_windows

ROOT = Path(".data/proxy")
SR = 16000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="cache name; pass it to e16_calls.py --model")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    a = ap.parse_args()
    cfg = load_config()
    sessions = [_session(m) for m in a.models]
    for split in a.splits:
        out_path = ROOT / "ser" / a.name / f"{split}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cache = json.loads(out_path.read_text()) if out_path.exists() else {}
        for f in sorted((ROOT / split).glob("*.ogg")):
            if f.name in cache:
                continue
            x = decode(f, SR)
            rows = []
            for s, e in speech_windows(x, cfg.vad_threshold, SR):
                w = x[None, s:e].astype(np.float32)
                p = np.mean([sess.run(None, {"wav": w})[0][0] for sess in sessions], axis=0)
                rows.append({"t": round(s / SR, 2), "dur": round((e - s) / SR, 2),
                             "hi": round(high_band_share(x[s:e], SR), 6),
                             "p": dict(zip(TONES, (round(float(v), 4) for v in p)))})
            cache[f.name] = rows
        out_path.write_text(json.dumps(cache))
        print(f"→ {out_path} ({len(cache)} calls)", flush=True)


if __name__ == "__main__":
    main()
