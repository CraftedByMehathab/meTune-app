.DEFAULT_GOAL := help
COMPOSE_DEV  := docker compose -f deploy/compose.dev.yaml
COMPOSE_PROD := docker compose -f deploy/compose.yaml --env-file deploy/.env

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

setup:           ## install python + node deps
	uv sync
	cd apps/client && pnpm install

db:              ## start local Postgres
	$(COMPOSE_DEV) up -d --wait

api:             ## run FastAPI with reload (needs .env)
	uv run --env-file .env uvicorn metune_api.main:app --reload --port 8000

client:          ## run Vite dev server (proxies /api → :8000)
	cd apps/client && pnpm dev

dev: db          ## db + api + client
	$(MAKE) -j2 api client

test:            ## python tests + client typecheck
	uv run pytest -q
	cd apps/client && pnpm typecheck

lint:            ## ruff
	uv run ruff check packages apps/api

types:           ## regenerate client API types from FastAPI's OpenAPI
	uv run python -m metune_api.export_openapi > apps/client/src/lib/openapi.json
	cd apps/client && pnpm types

hash-password:   ## print an argon2 hash for ADMIN_PASSWORD_HASH
	@uv run python -c "import getpass, sys; from pwdlib import PasswordHash; p = getpass.getpass('Password: '); \
	  p == getpass.getpass('Repeat:   ') or sys.exit('Passwords differ, nothing printed.'); print(PasswordHash.recommended().hash(p))"

run:             ## CLI on a folder: make run DIR=path/to/audio
	uv run --env-file .env metune run $(DIR) --out results.csv

mlflow-ui:       ## compare eval runs in MLflow (local; backfill: uv run --group track python eval/track.py backfill)
	MLFLOW_DISABLE_AGENT_HINT=1 uv run --group track mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

up:              ## build + run the full production stack locally (needs deploy/.env)
	$(COMPOSE_PROD) up -d --build

down:            ## stop the production stack
	$(COMPOSE_PROD) down

logs:            ## follow production stack logs
	$(COMPOSE_PROD) logs -f --tail=100

.PHONY: help setup db api client dev test lint types hash-password run mlflow-ui up down logs
