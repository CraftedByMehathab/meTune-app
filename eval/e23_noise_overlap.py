"""E23 / E24: dedicated local detectors for background noise (AST, AudioSet classes) and speaker overlap
(pyannote segmentation-3.0, ONNX). Compared with Gemini Flash-Lite (E22 runs from run_eval.py).

    uv run --group ser python eval/e23_noise_overlap.py features --split val_big
    uv run --group ser python eval/e23_noise_overlap.py features --split test_big
    uv run --group ser python eval/e23_noise_overlap.py score --tune val_big --test test_big \
        --gemini eval/runs/<E22-k3-ref-test_big> eval/runs/<E22-k1-test_big>

Leakage rule (DR-008): every threshold / classifier is fit on --tune only (ESC-50 folds 1–3, actors 1036–1045)
and applied once to --test (folds 4–5, unseen actors). No API calls.
"""
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(".data/proxy")
CACHE = ROOT / "e23_cache"
RUNS = Path("eval/runs")
SEG_MODEL = Path(".data/models/pyannote_seg3/sherpa-onnx-pyannote-segmentation-3-0/model.onnx")
AST_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
SR = 16000

# meTune noise type → AudioSet class ids (AST label space). Chatter uses babble only: "Conversation"/"Speech"
# fire on the call itself.
GROUPS = {
    "keyboard typing": [124, 384, 385, 386, 491, 492],
    "road noise": [300, 306, 307, 308, 309, 314, 316, 318, 322, 323, 324, 325, 327, 343, 344, 348, 349, 352, 396],
    "appliance hum": [369, 377, 412, 496, 516],
    "dog barking": [74, 75, 76, 77, 78, 79, 80, 122],
    "baby crying": [23, 24],
    "background chatter": [68, 69, 70],
    "static": [84, 299, 513, 515, 520, 521],
}
NAMES = list(GROUPS)
SEVERITIES = ["none", "low", "medium", "high"]


def labels_of(split: str) -> dict:
    rows = csv.DictReader((ROOT / split / "labels.csv").open())
    return {r["name"]: json.loads(r["result_json"]) for r in rows}


# ---------- features ----------

def gap_audio(x: np.ndarray, probs: np.ndarray, step: float) -> np.ndarray:
    """Non-speech stretches only (VAD < 0.3, one window of margin each side), joined with 10 ms fades."""
    w = int(step * SR)
    speech = probs >= 0.3
    speech = speech | np.roll(speech, 1) | np.roll(speech, -1)
    fade = np.linspace(0, 1, 160, dtype=np.float32)
    parts, i = [], 0
    while i < len(speech):
        if speech[i]:
            i += 1
            continue
        j = i
        while j < len(speech) and not speech[j]:
            j += 1
        seg = x[i * w:j * w].copy()
        if len(seg) > 2 * len(fade):
            seg[:160] *= fade
            seg[-160:] *= fade[::-1]
            parts.append(seg)
        i = j
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


class AST:
    def __init__(self):
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        self.torch = torch
        self.fe = ASTFeatureExtractor.from_pretrained(AST_ID)
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.m = ASTForAudioClassification.from_pretrained(AST_ID).eval().to(self.dev)

    def probs(self, x: np.ndarray, hop_s: float = 10.0) -> np.ndarray:
        """Sigmoid AudioSet probabilities per ~10 s window → (windows, 527)."""
        n, hop = int(10.24 * SR), int(hop_s * SR)
        chunks = [x[i:i + n] for i in range(0, max(1, len(x) - n + hop), hop)]
        chunks = [c for c in chunks if len(c) >= SR] or [x]
        inp = self.fe(chunks, sampling_rate=SR, return_tensors="pt")
        with self.torch.no_grad():
            logits = self.m(inp["input_values"].to(self.dev)).logits
        return self.torch.sigmoid(logits).cpu().numpy()


class Segmenter:
    """pyannote powerset output: 0 = none, 1–3 = one speaker, 4–6 = two speakers at once."""

    def __init__(self):
        import onnxruntime as ort
        self.s = ort.InferenceSession(str(SEG_MODEL), providers=["CPUExecutionProvider"])

    def overlap_probs(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        win, hop, shift = 160000, 40000, 270
        pad = np.concatenate([x, np.zeros(win, np.float32)])
        starts = list(range(0, max(1, len(x) - win + hop), hop))
        batch = np.stack([pad[s:s + win] for s in starts])[:, None, :].astype(np.float32)
        y = self.s.run(None, {"x": batch})[0]                  # (N, frames, 7) log-probs
        p = np.exp(y)
        p = p / p.sum(-1, keepdims=True)
        ov = p[..., 4:].sum(-1)
        total = int(len(x) / shift) + 1
        acc, cnt = np.zeros(total), np.zeros(total)
        for s, row in zip(starts, ov):
            a = s // shift
            k = min(len(row), total - a)
            acc[a:a + k] += row[:k]
            cnt[a:a + k] += 1
        return acc / np.maximum(cnt, 1), shift / SR


def features(split: str) -> None:
    from metune_pipeline.analyze import local_facts
    from metune_pipeline.config import load_config
    from metune_pipeline.decode import decode
    from metune_pipeline.vad import speech_probs

    cfg = load_config()
    ast, seg = AST(), Segmenter()
    names = sorted(labels_of(split))
    rows = []
    for k, name in enumerate(names, 1):
        x = decode(ROOT / split / name, SR)
        facts = local_facts(x, cfg)
        vp, step = speech_probs(x, SR)
        full = ast.probs(x)
        gaps = gap_audio(x, vp, step)
        gap = ast.probs(gaps, hop_s=5.0) if len(gaps) >= SR else np.zeros((1, full.shape[1]), np.float32)
        ov, ostep = seg.overlap_probs(x)
        rows.append({"name": name, "full_max": full.max(0), "full_mean": full.mean(0), "gap_max": gap.max(0),
                     "gap_s": len(gaps) / SR, "ov": ov, "ov_step": ostep, "facts": facts})
        print(f"  {k:>3}/{len(names)} {name} gap {len(gaps) / SR:4.1f}s  ov>0.5 {(ov > 0.5).sum() * ostep:4.1f}s",
              flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / f"{split}.npy", np.array(rows, dtype=object), allow_pickle=True)
    print("→", CACHE / f"{split}.npy")


# ---------- scoring ----------

def load(split: str) -> list[dict]:
    return list(np.load(CACHE / f"{split}.npy", allow_pickle=True))


def group_scores(r: dict) -> np.ndarray:
    """(3 sources: full max, full mean, gap max) × 7 groups → max class prob within each group."""
    return np.array([[r[src][ids].max() for ids in GROUPS.values()] for src in ("full_max", "full_mean", "gap_max")])


def static_rule(r: dict, cfg) -> bool:
    return r["facts"]["spikes_per_min"] >= cfg.spike_rate_static


def ovl_seconds(r: dict, p: float) -> float:
    return float((r["ov"] > p).sum() * r["ov_step"])


def binary(y: list[bool], p: list[bool]) -> dict:
    tp = sum(a and b for a, b in zip(y, p))
    fp = sum(b and not a for a, b in zip(y, p))
    fn = sum(a and not b for a, b in zip(y, p))
    return {"acc": round(sum(a == b for a, b in zip(y, p)) / len(y), 3),
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
            "f1": round(2 * tp / (2 * tp + fp + fn), 3) if tp else 0.0}


def best_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    """Threshold maximising accuracy on the tuning split (ties → the higher, i.e. more conservative)."""
    cands = np.unique(scores)
    accs = [((scores >= t) == y).mean() for t in cands]
    best = max(accs)
    return float(max(t for t, a in zip(cands, accs) if a == best))


def gemini_preds(run: Path) -> dict:
    out: dict = {}
    for r in csv.DictReader((run / "predictions.csv").open()):
        if r["field"] in ("background_noise_present", "background_noise_severity", "speaker_overlap_present",
                          "background_noise_type"):
            v = r["pred"]
            out.setdefault(r["name"], {})[r["field"]] = v == "True" if v in ("True", "False") else v
    s = json.loads((run / "summary.json").read_text())
    return {"preds": out, "cost_per_min": s["cost_per_min"], "name": s["name"]}


def score(a) -> None:
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from metune_pipeline.config import load_config

    cfg = load_config()
    tr, te = load(a.tune), load(a.test)
    ltr, lte = labels_of(a.tune), labels_of(a.test)
    y_tr = np.array([ltr[r["name"]]["background_noise_present"] for r in tr])
    y_te = [lte[r["name"]]["background_noise_present"] for r in te]
    sev_tr = np.array([ltr[r["name"]]["background_noise_severity"] for r in tr])
    sev_te = [lte[r["name"]]["background_noise_severity"] for r in te]
    typ_te = [lte[r["name"]]["background_noise_type"] for r in te]
    ov_tr = np.array([ltr[r["name"]]["speaker_overlap_present"] for r in tr])
    ov_te = [lte[r["name"]]["speaker_overlap_present"] for r in te]
    G_tr, G_te = np.stack([group_scores(r) for r in tr]), np.stack([group_scores(r) for r in te])
    st_tr = np.array([static_rule(r, cfg) for r in tr])
    st_te = np.array([static_rule(r, cfg) for r in te])
    res: dict = {"tune": a.tune, "test": a.test, "n_test": len(te), "when": datetime.now().isoformat(timespec="seconds")}

    # E23a zero-shot: one threshold on the strongest noise-group probability (full call or gaps), OR static rule
    z_tr, z_te = G_tr[:, [0, 2]].max((1, 2)), G_te[:, [0, 2]].max((1, 2))
    th = best_threshold(z_tr, y_tr)
    pa_te = list((z_te >= th) | st_te)
    res["E23a_zero_shot"] = {"threshold": round(th, 4), "tune": binary(list(y_tr), list((z_tr >= th) | st_tr)),
                             "noise_present": binary(y_te, pa_te)}

    # E23b trained: logistic regression on the 21 group scores (log-odds), severity 4-way → present = not none
    def X(G):
        G = np.clip(G.reshape(len(G), -1), 1e-4, 1 - 1e-4)
        return np.log(G / (1 - G))
    clf = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=10, cv=5, max_iter=5000, class_weight="balanced"))
    clf.fit(X(G_tr), sev_tr)
    sev_pred = clf.predict(X(G_te))
    pb_te = [s != "none" or bool(st) for s, st in zip(sev_pred, st_te)]
    sev_b = [("medium" if st else "none") if s == "none" else s for s, st in zip(sev_pred, st_te)]
    res["E23b_trained"] = {"noise_present": binary(y_te, pb_te),
                           "severity_acc": round(np.mean([p == t for p, t in zip(sev_b, sev_te)]), 3)}

    # noise type among calls where noise is present and detected: strongest group
    def typ(r):
        return NAMES[int(group_scores(r)[[0, 2]].max(0).argmax())]
    for key, pred in (("E23a_zero_shot", pa_te), ("E23b_trained", pb_te)):
        hits = [(typ(r), t) for r, t, p in zip(te, typ_te, pred) if t and p]
        res[key]["type_acc_on_detected"] = round(np.mean([a_ == b for a_, b in hits]), 3) if hits else None

    # E24 overlap: seconds with p(two speakers) > p ≥ θ; p and θ tuned on the tuning split
    best = None
    for p in (0.3, 0.4, 0.5, 0.6, 0.7):
        s_tr = np.array([ovl_seconds(r, p) for r in tr])
        t = best_threshold(s_tr, ov_tr)
        acc = ((s_tr >= t) == ov_tr).mean()
        if best is None or acc > best[0]:
            best = (acc, p, t)
    _, p_ov, t_ov = best
    po_te = [ovl_seconds(r, p_ov) >= t_ov for r in te]
    res["E24_overlap"] = {"frame_p": p_ov, "min_seconds": round(t_ov, 2),
                          "tune": binary(list(ov_tr), list(np.array([ovl_seconds(r, p_ov) for r in tr]) >= t_ov)),
                          "overlap": binary(ov_te, po_te)}
    # where do false alarms come from? background chatter is several voices at once
    res["E24_overlap"]["false_alarms_by_noise"] = dict(sorted(
        {k: sum(1 for p, t, ty in zip(po_te, ov_te, typ_te) if p and not t and (ty or "-") == k)
         for k in set(ty or "-" for ty in typ_te)}.items()))

    # baselines on the same test calls
    res["no_llm_baseline"] = {"noise_present": binary(y_te, list(st_te)),
                              "severity_acc": round(np.mean([("medium" if s else "none") == t
                                                             for s, t in zip(st_te, sev_te)]), 3),
                              "overlap": binary(ov_te, [False] * len(te)), "cost_per_min": 0.0}
    for run in a.gemini:
        g = gemini_preds(Path(run))
        pr = [g["preds"][r["name"]] for r in te]
        res[g["name"]] = {
            "noise_present": binary(y_te, [p["background_noise_present"] for p in pr]),
            "severity_acc": round(np.mean([p["background_noise_severity"] == t for p, t in zip(pr, sev_te)]), 3),
            "overlap": binary(ov_te, [p["speaker_overlap_present"] for p in pr]),
            "cost_per_min": g["cost_per_min"],
        }
        # hybrid: local detector OR Gemini says noise (Gemini as a second opinion)
        res[f"{g['name']} OR E23b"] = {"noise_present": binary(
            y_te, [p["background_noise_present"] or b for p, b in zip(pr, pb_te)])}

    out = RUNS / f"{datetime.now():%Y%m%d-%H%M}_E23-E24-{a.test}"
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(res, indent=2, default=float))
    with (out / "predictions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "noise_label", "noise_type", "severity", "E23a", "E23b", "E23b_severity", "E23_type",
                    "overlap_label", "overlap_s", "E24"])
        for r, y, ty, sv, pa, pb, sb, yo, po in zip(te, y_te, typ_te, sev_te, pa_te, pb_te, sev_b, ov_te, po_te):
            w.writerow([r["name"], y, ty, sv, pa, pb, sb, typ(r), yo, round(ovl_seconds(r, p_ov), 2), po])
    print(json.dumps(res, indent=1, default=float))
    print("→", out)
    from track import log_run
    log_run(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features")
    f.add_argument("--split", required=True)
    s = sub.add_parser("score")
    s.add_argument("--tune", default="val_big")
    s.add_argument("--test", default="test_big")
    s.add_argument("--gemini", nargs="*", default=[], help="run_eval.py run dirs on the --test split")
    args = ap.parse_args()
    features(args.split) if args.cmd == "features" else score(args)
    sys.exit(0)
