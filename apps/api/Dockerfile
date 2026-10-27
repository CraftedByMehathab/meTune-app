# Build context: repo root (needs the uv workspace + packages/pipeline)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/app/.venv

# deps layer (cached until a lockfile / pyproject changes)
COPY pyproject.toml uv.lock ./
COPY packages/pipeline/pyproject.toml packages/pipeline/
COPY apps/api/pyproject.toml apps/api/
RUN uv sync --frozen --no-dev --package metune-api --no-install-workspace

COPY packages packages
COPY apps/api apps/api
COPY config.yaml ./
RUN uv sync --frozen --no-dev --package metune-api

ARG GIT_SHA=dev
ENV PATH=/app/.venv/bin:$PATH METUNE_CONFIG=/app/config.yaml UPLOAD_DIR=/data/uploads GIT_SHA=$GIT_SHA

RUN useradd --create-home --uid 1000 app && mkdir -p /data/uploads && chown -R app /data
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/healthz')"
CMD ["uvicorn", "metune_api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
