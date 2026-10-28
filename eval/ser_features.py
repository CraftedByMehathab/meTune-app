"""E12 step 1: per-window speech-emotion probabilities for every proxy call (DR-020).

Speech is found with the pipeline's own Silero VAD, cut into windows of 1-4 s, and scored by a
SUPERB emotion model (IEMOCAP-trained: neutral / happy / angry / sad — no CREMA-D in training).
Output is cached so fusion rules can be tuned offline without re-running the model:
.data/proxy/ser/<model>/<split>.json = {file: [{"t": start_s, "dur": s, "p": {label: prob}}]}

    uv run --group ser python eval/ser_features.py --model superb/hubert-large-superb-er
"""
import argparse
import json
import time
from pathlib import Path

import torch
from metune_pipeline.config import load_config
from metune_pipeline.decode import decode
from metune_pipeline.ser import high_band_share, speech_windows  # shared with the shipped pipeline
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

ROOT = Path(".data/proxy")
SR = 16000



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="superb/hubert-large-superb-er")
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    a = ap.parse_args()
    cfg = load_config()
    fe = AutoFeatureExtractor.from_pretrained(a.model)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForAudioClassification.from_pretrained(a.model).eval().to(dev)
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    print(a.model, "labels:", labels)
    torch.set_num_threads(4)

    for split in a.splits:
        folder = ROOT / split if (ROOT / split).exists() else Path(split)
        out_path = ROOT / "ser" / a.model.replace("/", "__") / f"{folder.name}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cache = json.loads(out_path.read_text()) if out_path.exists() else {}
        for f in sorted(p for p in folder.iterdir() if p.suffix in {".ogg", ".flac"}):
            if f.name in cache:
                continue
            t0 = time.perf_counter()
            x = decode(f, SR)
            wins = speech_windows(x, cfg.vad_threshold)
            rows = []
            with torch.inference_mode():
                for s, e in wins:
                    inp = fe(x[s:e], sampling_rate=SR, return_tensors="pt")
                    p = torch.softmax(model(**inp.to(dev)).logits[0], -1).tolist()
                    rows.append({"t": round(s / SR, 2), "dur": round((e - s) / SR, 2),
                                 "hi": round(high_band_share(x[s:e], SR), 6),
                                 "p": dict(zip(labels, (round(v, 4) for v in p)))})
            cache[f.name] = rows
            print(f"  {folder.name}/{f.name}: {len(rows)} windows · {len(x) / SR:.0f} s audio · "
                  f"{time.perf_counter() - t0:.1f} s ", flush=True)
            out_path.write_text(json.dumps(cache))
        print("→", out_path)


if __name__ == "__main__":
    main()
