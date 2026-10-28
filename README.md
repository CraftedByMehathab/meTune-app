# meTune
> **Heard. Understood. Attuned.**

meTune analyzes call audio and returns, for each clip:
- the customer's **emotional tone** and **intensity**
- **background noise**: whether it's present, its type and its severity
- **audio quality**, **speaker overlap** and **long silence**

It uses a hybrid pipeline: DSP rules + VAD for the measurable fields, an audio LLM (Gemini 3.8 Flash, one vote) for tone, intensity and overlap, an AudioSet sound classifier (AST, ONNX, local) for background noise, and a speaker-overlap detector (pyannote, ONNX, local) ORed with the LLM. Target cost is ≤ $0.003 per audio minute; measured cost is **$0.0028** on the proxy test calls (E26).

Design notes and decisions live in the meTune Obsidian vault (DR-001 → DR-020; experiments in the Experiment Log).

**Speech-emotion model (optional):** tone comes from Gemini (`tone_source: llm`, E35). `tone_source: ser` switches to the speech-emotion bag, which needs the 5 models listed under `ser_models` in `.data/models/` (8-bit ONNX, 5 × 122 MB, not in git). They are a bag: 5 fine-tunes of wav2vec2 on bootstrap samples of training speakers (`eval/ser_finetune.py --bootstrap i`), exported with `eval/export_ser_onnx.py`; their probabilities are averaged. It wins on acted speech (CREMA-D proxy) but not on natural conversation (MELD, E34–E35).

## Repo layout
```
apps/
  client/            Vite + React + TS · React Router · shadcn/ui · TanStack Query
  api/               FastAPI · auth · batch upload + manifest validation · Postgres-queue worker · scoring
packages/
  pipeline/          decode → VAD (Silero ONNX) → DSP → Gemini votes → fuse → Result   (+ `metune` CLI)
deploy/              compose.yaml (caddy, client, api, db) · Caddyfile · scripts · DEPLOY.md
config.yaml          every threshold, model id, vote count, price (hash stored per batch)
```

## Quick start (local)
Requirements: [uv](https://docs.astral.sh/uv/), pnpm 9, Node 22, Docker, ffmpeg (`brew install ffmpeg`).
```bash
make setup
cp .env.example .env
make hash-password          # paste into ADMIN_PASSWORD_HASH (single quotes); add GEMINI_API_KEY
make dev                    # Postgres in Docker + api :8000 + client :5173
```
Open http://localhost:5173, log in, and upload a ZIP or folder. Without `GEMINI_API_KEY`, tone falls back to neutral (or the speech-emotion model with `tone_source: ser`); noise and overlap still come from the local detectors.

## CLI (no web app)
```bash
make run DIR=../meTune/data/trial      # → results.csv  (name,result_json)
uv run metune run <folder> --no-llm    # no API calls: speech-emotion tone + DSP baseline
```

## Commands
| | |
|---|---|
| `make test` | pytest (pipeline + api) and client typecheck |
| `make types` | regenerate client types from FastAPI OpenAPI (CI fails if stale) |
| `make mlflow-ui` | compare eval runs in MLflow (local only); `uv run --group track python eval/track.py backfill` imports `eval/runs/` |
| `make up` / `make down` / `make logs` | full production stack locally (needs `deploy/.env`) |

## Batch input
Upload a **ZIP** or a **folder** of audio files: wav, mp3, ogg/opus, m4a, flac or webm.
- **Manifest:** an optional CSV at the shallowest level, with a `name` column and an optional `result_json` column (labels). When labels are present, the batch is auto-scored: accuracy, macro F1, per-class P/R/F1 and a tone confusion matrix.
- **Validation first:** a report is shown before processing starts (matched, audio without row, row without audio, unsupported, bad JSON).
- **Failure isolation:** each file is a separate job, retried up to 3 times. Decode errors are not retried. One bad file never stops the batch.

## Deploy
One VM (2 vCPU / 4 GB) running Docker Compose behind Caddy (auto TLS). See **[deploy/DEPLOY.md](deploy/DEPLOY.md)**. Releases: push a `v*` tag, see **[deploy/RELEASE.md](deploy/RELEASE.md)**.

## Known limitations (trial)
- **Tone:** from Gemini, which hears words and voice. On natural conversation (MELD test, 1,617 lines) macro F1 **0.38**, accuracy 0.56; the speech-emotion bag, trained on acted CREMA-D, scores 0.19 there although it wins on the acted proxy set (E34–E35). Weakest: frustrated (Gemini's pick right ~10%) and distressed. MELD is TV dialogue, not phone calls. Confidence = how often Gemini's pick of that tone was right on MELD dev (test ECE 0.044, E36).
- **Overlap:** pyannote segmentation-3.0 (vendored ONNX) OR the LLM (E31): recall 0.68, precision 0.90 on proxy calls with real overlaps; 1 in 3 overlaps still missed.
- **Audio quality:** the DSP rule uses clipping only.
- **Confidence:** vote agreement only; not calibrated (DR-009).
- **Database migrations:** tables are created on boot. Add Alembic before changing the schema in production.
