# meTune
> **Heard. Understood. Attuned.**

[![ci](https://github.com/CraftedByMehathab/meTune-app/actions/workflows/ci.yml/badge.svg)](https://github.com/CraftedByMehathab/meTune-app/actions/workflows/ci.yml)

meTune analyzes call recordings in batches and returns, for each call:
- the customer's **emotional tone** and **intensity**, with a **calibrated confidence**
- **background noise**: present, type and severity
- **audio quality**, **speaker overlap** and **long silence**

**Dashboard:** https://metune-dev.mehathab.dev · **Technical memo** (approaches, validation, cost, latency, privacy, limits): **[docs/TECHNICAL_MEMO.md](docs/TECHNICAL_MEMO.md)**

## How it works
Each field comes from the cheapest component that measured best for it on held-out data:

| Field | Component | Held-out result |
|---|---|---|
| Tone, intensity | Gemini 3.8 Flash, one request per file | Macro F1 0.38, accuracy 0.56 on natural dialogue (MELD test, n = 1,617) |
| Background noise | AST AudioSet classifier (8-bit ONNX, local) + trained head | Accuracy 0.95, precision 0.94, recall 0.97 |
| Speaker overlap | pyannote segmentation (ONNX, local) OR the LLM | Accuracy 0.83, precision 0.90, recall 0.68 |
| Silence, audio quality | DSP + Silero VAD (ONNX, local) | Silence accuracy 0.98 |
| Confidence | P(tone is right), calibrated per predicted tone | ECE 0.044 |

- **Cost:** about $0.0014 per audio minute on a 3-minute call (ceiling: $0.003).
- **Latency:** 15–40 s per call on a 4-vCPU VM.
- **External service:** only the Gemini API (paid tier, not used for training).

Details and the approaches that were tried and rejected are in the [memo](docs/TECHNICAL_MEMO.md).

## Repo layout
```
apps/
  client/            React + TypeScript dashboard (Vite, shadcn/ui, TanStack Query)
  api/               FastAPI: auth, batch upload + manifest validation, Postgres job queue, worker, scoring
packages/
  pipeline/          decode → DSP + VAD → AST noise · pyannote overlap · Gemini → fuse → Result  (+ `metune` CLI)
deploy/              Docker Compose (Caddy, client, API, Postgres), deploy and release guides
eval/                proxy-set builder, experiments E6–E36, run outputs in eval/runs/
docs/                technical memo
config.yaml          every threshold, model id and price (its hash is stored on each batch)
```

## Quick start (local)
Requirements: [uv](https://docs.astral.sh/uv/), pnpm 9, Node 22, Docker, ffmpeg.
```bash
make setup
cp .env.example .env
make hash-password          # paste into ADMIN_PASSWORD_HASH (single quotes); add GEMINI_API_KEY
make dev                    # Postgres in Docker + API on :8000 + dashboard on :5173
```
Open http://localhost:5173, log in, and upload a folder or ZIP.

Noise needs the AST model in `.data/models/`: build it with `uv run --group ser python eval/export_ast_onnx.py`. The overlap and voice-activity models are included in the package.

## CLI (no web app)
```bash
uv run --env-file .env metune run <folder> --out results.csv    # one row per file: name,result_json
uv run metune run <folder> --no-llm                             # no API calls: local fields only, tone neutral
```

## Batch input
Upload a **ZIP** or a **folder** of audio files: wav, mp3, ogg/opus, m4a, flac or webm.
- **Manifest (optional):** a CSV with a `name` column and an optional `result_json` column of labels. With labels, the batch is scored automatically: accuracy, macro F1, per-class precision/recall/F1 and a tone confusion matrix.
- **Validation first:** a report appears before processing starts, listing matched files, audio without a manifest row, rows without audio, unsupported files and bad JSON.
- **Failure isolation:** each file is its own job, retried up to 3 times; decode errors are not retried. One bad file never stops a batch.
- **Downloads:** results as CSV (`name,result_json`) or JSON.

## Commands
| | |
|---|---|
| `make test` | Python tests (pipeline + API) and client typecheck |
| `make types` | regenerate client API types from FastAPI's OpenAPI schema (CI fails if stale) |
| `make mlflow-ui` | compare experiment runs in MLflow (local only) |
| `make up` / `make down` / `make logs` | full production stack locally (needs `deploy/.env`) |

## Deploy and release
One VM (4 vCPU / 8 GB) running Docker Compose behind Caddy with automatic TLS. See **[deploy/DEPLOY.md](deploy/DEPLOY.md)**.

Releases are tag-driven: merge a pull request, then push a `v*` tag. GitHub Actions builds the images, pushes them to GHCR and deploys over SSH. See **[deploy/RELEASE.md](deploy/RELEASE.md)**.

## Configuration
Everything that changes results lives in [`config.yaml`](config.yaml):
- model ids and the tone source (`tone_source: llm` or `ser`)
- vote count and thinking level
- every threshold, the noise and overlap detectors, and the confidence calibration
- prices used for cost reporting

## Limits
Summarized from the memo:
- **Tone on real calls is unproven.** The 3 trial calls score 0/3 on tone for every approach tried, and 3 calls cannot rank them.
- **Weak classes:** "frustrated" and "distressed" are the weakest tones.
- **Noise severity** is right about 6 times in 10.
- **Cost:** calls shorter than about 30 s exceed $0.003/min.

Next steps are listed in [memo §9](docs/TECHNICAL_MEMO.md#9-failure-modes-limits-and-next-steps).
