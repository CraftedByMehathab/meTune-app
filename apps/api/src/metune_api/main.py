import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from metune_pipeline import load_config
from sqlalchemy import text
from starlette.middleware.sessions import SessionMiddleware

from . import auth
from .db import engine, init_db
from .routes import batches
from .settings import get_settings
from .worker import Worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("metune")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.cookie_secure and settings.session_secret.startswith("dev-"):
        raise RuntimeError("Set SESSION_SECRET for production (COOKIE_SECURE=true)")
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    await init_db()
    app.state.cfg = load_config(settings.metune_config)
    app.state.worker = Worker(settings, app.state.cfg)
    task = asyncio.create_task(app.state.worker.run())
    log.info("meTune API up · model=%s · votes=%s · sha=%s", app.state.cfg.llm_model,
             app.state.cfg.llm_votes, settings.git_sha)
    yield
    task.cancel()


app = FastAPI(title="meTune API", version="0.1.0", lifespan=lifespan,
              docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(SessionMiddleware, secret_key=settings.session_secret, session_cookie="metune_session",
                   https_only=settings.cookie_secure, same_site="lax", max_age=settings.session_max_age_s)
app.include_router(auth.router)
app.include_router(batches.router)


@app.get("/api/healthz", tags=["ops"])
async def healthz() -> dict:
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    return {"ok": True, "sha": settings.git_sha}
