"""E15: a local open audio LLM (Qwen2.5-Omni) in place of Gemini, same prompt, facts, fusion and scoring.

Question: is "always neutral" a Gemini limitation, or do audio LLMs in general miss prosody-only emotion?
Runs locally (MPS / CPU), so no API cost and no data egress. Only the thinker (text output) is loaded.
Raw outputs are cached so an interrupted run resumes and re-scoring needs no model:
.data/proxy/qwen/<model>/<split>_<prompt>.json = {file: [vote dict | {"error": ..., "raw": ...}]}

    uv run --group ser python eval/qwen_eval.py --split dev --name E15-qwen7b-v2 --limit 3   # smoke
    uv run --group ser python eval/qwen_eval.py --split dev --name E15-qwen7b-v2

Leakage caveat: Qwen's training mix is not fully published (MELD is used for SER); CREMA-D / ESC-50
exposure cannot be ruled out, so a strong proxy score is an upper bound until checked on other data.
"""
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import get_args

import numpy as np
import torch
from metune_api.scoring import score
from metune_pipeline.analyze import FACT_KEYS, local_facts
from metune_pipeline.config import load_config
from metune_pipeline.decode import decode
from metune_pipeline.fuse import fuse
from metune_pipeline.llm import PROMPTS
from metune_pipeline.schema import SCORED_FIELDS
from pydantic import ValidationError
from run_eval import DEFAULTS, binary
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

ROOT = Path(".data/proxy")
RUNS = Path("eval/runs")
# Without the stock system prompt the model keeps generating fake "Human:" turns after the answer.
SYSTEM = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving "
          "auditory and visual inputs, as well as generating text and speech.")


def json_hint(schema) -> str:
    """No schema-enforced JSON outside Gemini: spell out every key with its allowed values (smoke run wrote
    "none" for a bool and "good" for audio_quality when given key names only)."""
    def allowed(t) -> str:
        if t is bool:
            return "true | false"
        if t is str:
            return "<string>"
        return " | ".join(f'"{v}"' for v in get_args(t))
    lines = ",\n".join(f'  "{k}": {allowed(f.annotation)}' for k, f in schema.model_fields.items())
    return f"\nReturn exactly one JSON object, nothing after it:\n{{\n{lines}\n}}"


def parse(text: str, schema) -> dict:
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in output")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])         # first object only; ignore trailing text
    return schema.model_validate(obj).model_dump()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--name", required=True, help="experiment id, e.g. E15-qwen7b-v2")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Omni-7B")
    ap.add_argument("--prompt", default="v2", choices=sorted(PROMPTS))
    ap.add_argument("--votes", type=int, default=1, help="k; temperatures cycle through config llm_temperatures")
    ap.add_argument("--limit", type=int, default=0, help="first N calls only (smoke test)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()

    cfg = load_config()
    folder = ROOT / a.split if (ROOT / a.split).exists() else Path(a.split)
    labels = {r["name"]: json.loads(r["result_json"]) for r in csv.DictReader((folder / "labels.csv").open())}
    names = sorted(labels)[: a.limit or None]
    prompt, schema = PROMPTS[a.prompt]
    hint = json_hint(schema)

    cache_path = ROOT / "qwen" / a.model.split("/")[-1] / f"{a.split}_{a.prompt}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    todo = [n for n in names if len(cache.get(n, [])) < a.votes]
    model = processor = None
    if todo:
        print(f"loading {a.model} on {a.device} …", flush=True)
        processor = Qwen2_5OmniProcessor.from_pretrained(a.model)
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            a.model, dtype=torch.bfloat16 if a.device != "cpu" else torch.float32).to(a.device).eval()

    facts_by, dur_by, secs = {}, {}, []
    print(f"{a.name}: {len(names)} calls from {folder}, {len(todo)} to run", flush=True)
    for i, n in enumerate(names, 1):
        x = decode(folder / n, cfg.sample_rate)
        facts_by[n], dur_by[n] = local_facts(x, cfg), len(x) / cfg.sample_rate
        votes = cache.setdefault(n, [])
        while len(votes) < a.votes:
            text = prompt.format(facts=json.dumps({k: facts_by[n][k] for k in FACT_KEYS})) + hint
            conv = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                    {"role": "user", "content": [{"type": "audio", "path": n}, {"type": "text", "text": text}]}]
            chat = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=chat, audio=[x.astype(np.float32)], sampling_rate=cfg.sample_rate,
                               return_tensors="pt", padding=True).to(a.device)
            if "input_features" in inputs:
                inputs["input_features"] = inputs["input_features"].to(model.dtype)
            temp = cfg.llm_temperatures[len(votes) % len(cfg.llm_temperatures)]
            t0 = time.perf_counter()
            with torch.inference_mode():
                ids = model.generate(**inputs, max_new_tokens=400, do_sample=temp > 0,
                                     temperature=temp if temp > 0 else None,
                                     stop_strings=["\n}"], tokenizer=processor.tokenizer)
            secs.append(time.perf_counter() - t0)
            raw = processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            try:
                votes.append(parse(raw, schema))
            except (ValueError, ValidationError) as err:
                votes.append({"error": str(err)[:200], "raw": raw[:2000]})
            cache_path.write_text(json.dumps(cache, indent=1))      # resumable
        ok = [v for v in votes[: a.votes] if "error" not in v]
        print(f"  {i:>3}/{len(names)} {n} {[v['emotional_tone'] for v in ok]}"
              f"{f'  ⚠ {a.votes - len(ok)} parse errors' if len(ok) < a.votes else ''}", flush=True)

    results = {n: fuse(facts_by[n], [v for v in cache[n][: a.votes] if "error" not in v], cfg,
                       degraded=not any("error" not in v for v in cache[n][: a.votes])).model_dump()
               for n in names}
    pairs = [(labels[n], results[n]) for n in names]
    metrics = score(pairs).model_dump()
    emotional = [(t, p) for t, p in pairs if t["emotional_tone"] != "neutral"]
    minutes = sum(dur_by.values()) / 60
    summary = {
        "name": a.name, "split": a.split, "n": len(pairs), "when": datetime.now().isoformat(timespec="seconds"),
        "model": a.model, "prompt": a.prompt, "votes": a.votes, "device": a.device,
        "accuracy": {f: metrics["per_field"][f]["accuracy"] for f in SCORED_FIELDS},
        "macro_f1": {f: metrics["per_field"][f]["macro_f1"] for f in SCORED_FIELDS},
        "intensity_accuracy_non_neutral": round(sum(t["emotional_intensity"] == p["emotional_intensity"]
                                                    for t, p in emotional) / max(1, len(emotional)), 3),
        "noise_present": binary(pairs, "background_noise_present"),
        "overlap": binary(pairs, "speaker_overlap_present"),
        "all_default_answers": sum(all(p[k] == v for k, v in DEFAULTS.items()) for _, p in pairs),
        "tone_report": metrics["tone_report"], "tone_confusion": metrics["tone_confusion"],
        "audio_min": round(minutes, 2),
        "parse_errors": sum(1 for n in names for v in cache[n][: a.votes] if "error" in v),
        # only the votes generated in this invocation (cached ones cost nothing)
        "gen_s_per_vote": round(sum(secs) / len(secs), 1) if secs else None,
        "gen_s_per_audio_min": round(sum(secs) / minutes / a.votes, 1) if secs and len(secs) == len(names) * a.votes
        else None,
    }

    out = RUNS / f"{datetime.now():%Y%m%d-%H%M}_{a.name}"
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out / "predictions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "field", "label", "pred", "match"])
        for n, (t, p) in zip(names, pairs):
            for k in [*SCORED_FIELDS, "background_noise_type", "confidence"]:
                w.writerow([n, k, t.get(k), p[k], t.get(k) == p[k] if k in SCORED_FIELDS else ""])
            for j, v in enumerate(cache[n][: a.votes]):
                w.writerow([n, f"vote{j}", "", json.dumps(v), ""])

    print(json.dumps({k: summary[k] for k in ["name", "accuracy", "macro_f1", "intensity_accuracy_non_neutral",
                                              "noise_present", "overlap", "all_default_answers", "parse_errors",
                                              "gen_s_per_vote"]}, indent=1))
    print("tone confusion (rows=true):", summary["tone_confusion"]["labels"])
    for lbl, row in zip(summary["tone_confusion"]["labels"], summary["tone_confusion"]["matrix"]):
        print(f"  {lbl:<11}{row}")
    print("→", out)


if __name__ == "__main__":
    main()
