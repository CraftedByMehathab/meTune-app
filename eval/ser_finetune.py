"""E16: fine-tune the E12 speech-emotion model on CREMA-D clips → meTune's 5 tones directly (DR-020 follow-up).

Start: superb/wav2vec2-base-superb-er (IEMOCAP, 4 classes). Its 4-way head is replaced by a fresh 5-way head
(neutral / satisfied / frustrated / upset / distressed) and the model is trained further on single clips.

Speaker-grouped, never mixed (DR-008 leakage rule):
  train  actors 1001–1035   → weights are learned here
  val    actors 1036–1045   → checked after every epoch; best epoch kept
  test   actors 1046–1091   → untouched here; scored once later on the test calls
Dev calls use actors 1001–1045 → they are NOT a fair score for this model.

Every clip goes through the proxy phone channel (300–3400 Hz, 8 kHz, Opus 24k) so training audio sounds like
the calls. Random ±6 dB gain teaches the model that loudness alone is not emotion (brief warning).

    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --group ser python eval/ser_finetune.py --name E16-ft
Output: .data/proxy/ser_ft/<name>/model (best epoch) + eval/runs/<ts>_<name>/summary.json (curves).
"""
import argparse
import json
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from build_proxy import ESC_URL, SEVERITY_SNR, SRC, TONE_OF, Builder, fetch, loop_to, rms, synth_static
from metune_pipeline.decode import decode
from metune_pipeline.schema import TONES
from sklearn.metrics import confusion_matrix, f1_score
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    get_linear_schedule_with_warmup,
)

ROOT = Path(".data/proxy/ser_ft")
RUNS = Path("eval/runs")
SR = 16000
MAX_S = 4.0
SPLIT = {"train": range(1001, 1036), "val": range(1036, 1046)}
NOISE_FOLDS = {1, 2, 3}            # ESC-50 folds of dev/val calls; test calls use 4–5


def phone(name: str, src: Path | None = None) -> Path:
    """Same channel as build_proxy.write_call (mono here: the model sees the mixed-down signal anyway)."""
    dst = ROOT / "phone" / f"{name}.ogg"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src or SRC / "crema" / f"{name}.wav"),
                        "-af", "highpass=f=300,lowpass=f=3400,aresample=8000", "-ac", "1",
                        "-c:a", "libopus", "-b:a", "24k", str(dst)], check=True)
    return dst


def load_noise(b: Builder) -> list[np.ndarray]:
    """E17: ESC-50 background clips from folds 1–3 only (test calls use folds 4–5), phone channel applied
    separately and mixed after it — close to (not identical with) the proxy, which mixes before the channel."""
    rows = [r for r in b.esc if int(r["fold"]) in NOISE_FOLDS]
    def one(r):
        src = fetch(ESC_URL.format(r["filename"]), SRC / "esc50" / r["filename"])
        return decode(phone(f"esc50_{Path(r['filename']).stem}", src), SR)
    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(one, rows))


def add_noise(x: np.ndarray, noise: list[np.ndarray], rng: random.Random) -> np.ndarray:
    """Background at one of the proxy severities (20 / 10 / 3 dB SNR); ~1 in 8 is synthetic line static."""
    if rng.random() < 0.125:
        n = synth_static(len(x), np.random.default_rng(rng.randrange(2**32)))
    else:
        clip = rng.choice(noise)
        n = np.roll(loop_to(clip, len(x)), rng.randrange(len(x)))
    snr = rng.choice(list(SEVERITY_SNR.values()))
    return (x + n * rms(x) / (rms(n) * 10 ** (snr / 20))).astype(np.float32)


def load_split(b: Builder, actors: range) -> list[tuple[np.ndarray, int]]:
    items = [(f, TONES.index(TONE_OF[emo])) for (a, emo), clips in b.clips.items() if a in actors for f, _ in clips]
    with ThreadPoolExecutor(8) as ex:
        audio = list(ex.map(lambda it: decode(phone(it[0]), SR), items))
    return [(x[: int(MAX_S * SR)], y) for x, (_, y) in zip(audio, items)]


def batches(data, size: int, train: bool, fe, rng: random.Random, noise=None, noise_p: float = 0.0,
            mask: bool = False):
    """Yields (model inputs, labels). `mask`: pass the attention mask (layer-norm models, e.g. hubert-large)."""
    order = list(range(len(data)))
    if train:
        rng.shuffle(order)
    for i in range(0, len(order), size):
        xs, ys = zip(*(data[j] for j in order[i:i + size]))
        if train:
            if noise_p:
                xs = [add_noise(x, noise, rng) if rng.random() < noise_p else x for x in xs]
            xs = [x * 10 ** (rng.uniform(-6, 6) / 20) for x in xs]
        # group-norm models (wav2vec2-base) take zero padding without a mask, as their model card does;
        # layer-norm models (hubert-large) need the mask or padding is treated as real silence
        feats = fe(list(xs), sampling_rate=SR, padding=True, return_tensors="pt")
        keys = ["input_values", "attention_mask"] if mask else ["input_values"]
        yield {k: feats[k] for k in keys}, torch.tensor(ys)


def evaluate(model, data, fe, device, mask: bool = False) -> dict:
    model.eval()
    preds, gold, loss = [], [], 0.0
    with torch.inference_mode():
        for x, y in batches(data, 16, False, fe, None, mask=mask):
            out = model(**{k: v.to(device) for k, v in x.items()}, labels=y.to(device))
            loss += out.loss.item() * len(y)
            preds += out.logits.argmax(-1).cpu().tolist()
            gold += y.tolist()
    neu = TONES.index("neutral")
    said_neutral = [g for p, g in zip(preds, gold) if p == neu]
    return {
        "loss": round(loss / len(gold), 4),
        "accuracy": round(float(np.mean(np.array(preds) == np.array(gold))), 3),
        "macro_f1": round(f1_score(gold, preds, average="macro"), 3),
        "recall": {t: round(float(np.mean([p == i for p, g in zip(preds, gold) if g == i])), 3)
                   for i, t in enumerate(TONES)},
        # the E12 blocker: when the model says neutral, is it? / how many neutral clips does it keep?
        "neutral_precision": round(float(np.mean([g == neu for g in said_neutral])), 3) if said_neutral else None,
        "confusion": confusion_matrix(gold, preds, labels=range(len(TONES))).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", default="superb/wav2vec2-base-superb-er")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--noise-p", type=float, default=0.0, help="E17: share of train clips with background noise")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)

    b = Builder(a.seed)
    t0 = time.perf_counter()
    train, val = load_split(b, SPLIT["train"]), load_split(b, SPLIT["val"])
    noise = load_noise(b) if a.noise_p else None
    print(f"clips: train {len(train)}, val {len(val)} (phone channel, ≤{MAX_S:.0f}s) in {time.perf_counter() - t0:.0f}s",
          flush=True)

    fe = AutoFeatureExtractor.from_pretrained(a.base)
    model = AutoModelForAudioClassification.from_pretrained(
        a.base, id2label=dict(enumerate(TONES)), label2id={t: i for i, t in enumerate(TONES)},
        ignore_mismatched_sizes=True,                         # 4-way IEMOCAP head → fresh 5-way head
        layerdrop=0.0)      # skipped layers break the head's weighted sum over all 13 layer outputs
    model.freeze_feature_encoder()                            # CNN front end stays as pre-trained
    mask = model.config.feat_extract_norm == "layer"
    model.to(a.device)

    # Upset/neutral clips outnumber the rest ~2:1; weight the loss so rare tones count as much (macro F1).
    counts = np.bincount([y for _, y in train], minlength=len(TONES))
    weights = torch.tensor(counts.sum() / (len(TONES) * counts), dtype=torch.float32, device=a.device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01)
    steps = a.epochs * -(-len(train) // a.batch)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    best, history = -1.0, []
    out_dir = ROOT / a.name / "model"
    before = evaluate(model, val, fe, a.device, mask)
    print(f"epoch 0 (fresh head, untrained)  val acc {before['accuracy']}  macro F1 {before['macro_f1']}", flush=True)
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0, total, n = time.perf_counter(), 0.0, 0
        for x, y in batches(train, a.batch, True, fe, rng, noise, a.noise_p, mask):
            logits = model(**{k: v.to(a.device) for k, v in x.items()}).logits
            loss = loss_fn(logits, y.to(a.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            total, n = total + loss.item() * len(y), n + len(y)
        v = evaluate(model, val, fe, a.device, mask)
        history.append({"epoch": epoch, "train_loss": round(total / n, 4), "val": v, "s": round(time.perf_counter() - t0)})
        mark = ""
        if v["macro_f1"] > best:
            best, mark = v["macro_f1"], "  ← best, saved"
            model.save_pretrained(out_dir); fe.save_pretrained(out_dir)
        print(f"epoch {epoch}  train loss {total / n:.3f}  val loss {v['loss']:.3f}  acc {v['accuracy']}  "
              f"macro F1 {v['macro_f1']}  neutral prec {v['neutral_precision']}  ({history[-1]['s']}s){mark}", flush=True)

    run = RUNS / f"{datetime.now():%Y%m%d-%H%M}_{a.name}"
    run.mkdir(parents=True)
    best_epoch = max(history, key=lambda h: h["val"]["macro_f1"])
    (run / "summary.json").write_text(json.dumps({
        "name": a.name, "when": datetime.now().isoformat(timespec="seconds"), "base": a.base, "args": vars(a),
        "clips": {"train": len(train), "val": len(val)}, "class_weights": weights.cpu().round(decimals=3).tolist(),
        "val_before_training": before, "best_epoch": best_epoch, "history": history, "model_dir": str(out_dir),
    }, indent=2))
    print("best epoch", best_epoch["epoch"], "val recall per tone:", best_epoch["val"]["recall"])
    print("val confusion (rows=true):", TONES)
    for t, row in zip(TONES, best_epoch["val"]["confusion"]):
        print(f"  {t:<11}{row}")
    print("→", run)


if __name__ == "__main__":
    main()
