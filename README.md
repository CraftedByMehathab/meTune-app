# meTune
> **Heard. Understood. Attuned.**

meTune analyzes call audio and returns, for each clip:
- the customer's **emotional tone** and **intensity**
- **background noise**: whether it's present, its type and its severity
- **audio quality**, **speaker overlap** and **long silence**

It uses a hybrid pipeline: DSP rules + VAD for the measurable fields, and an audio LLM (Gemini Flash-Lite) for tone and noise. Target cost is ≤ $0.003 per audio minute; measured cost is **$0.00135**.

Design notes and decisions live in the meTune Obsidian vault (DR-001 → DR-012).

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
Open http://localhost:5173, log in, and upload a ZIP or folder. Without `GEMINI_API_KEY`, results come from the DSP baseline.

## CLI (no web app)
```bash
make run DIR=../meTune/data/trial      # → results.csv  (name,result_json)
uv run metune run <folder> --no-llm    # DSP baseline only
```

## Commands
| | |
|---|---|
| `make test` | pytest (pipeline + api) and client typecheck |
| `make types` | regenerate client types from FastAPI OpenAPI (CI fails if stale) |
| `make up` / `make down` / `make logs` | full production stack locally (needs `deploy/.env`) |

## Batch input
Upload a **ZIP** or a **folder** of audio files: wav, mp3, ogg/opus, m4a, flac or webm.
- **Manifest:** an optional CSV at the shallowest level, with a `name` column and an optional `result_json` column (labels). When labels are present, the batch is auto-scored: accuracy, macro F1, per-class P/R/F1 and a tone confusion matrix.
- **Validation first:** a report is shown before processing starts (matched, audio without row, row without audio, unsupported, bad JSON).
- **Failure isolation:** each file is a separate job, retried up to 3 times. Decode errors are not retried. One bad file never stops the batch.

## Deploy
One VM (2 vCPU / 4 GB) running Docker Compose behind Caddy (auto TLS). See **[deploy/DEPLOY.md](deploy/DEPLOY.md)**.

## Known limitations (trial)
- **Tone:** the current prompt predicts `neutral` for all 3 trial calls. Prompt work must be validated on proxy sets, not on the scored calls.
- **Overlap:** comes from the LLM only; there is no dedicated overlap detector yet.
- **Audio quality:** the DSP rule uses clipping only.
- **Confidence:** vote agreement only; not calibrated (DR-009).
- **Database migrations:** tables are created on boot. Add Alembic before changing the schema in production.
