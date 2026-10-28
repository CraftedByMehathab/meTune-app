"""In-process worker. Postgres is the queue (FOR UPDATE SKIP LOCKED); one failed file never stops a batch."""
import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path

from metune_pipeline import PipelineConfig, analyze_file, make_client
from metune_pipeline.decode import DecodeError
from sqlalchemy import func, select, text, update

from .db import SessionLocal
from .models import Batch, File, LLMCallRow, ResultRow
from .settings import Settings

log = logging.getLogger("metune.worker")

CLAIM_SQL = text("""
UPDATE files SET status = 'running', attempts = attempts + 1, updated_at = now()
WHERE id = (SELECT id FROM files WHERE status = 'queued' ORDER BY id FOR UPDATE SKIP LOCKED LIMIT 1)
RETURNING id
""")


class Worker:
    def __init__(self, settings: Settings, cfg: PipelineConfig):
        self.settings, self.cfg = settings, cfg
        self.wake = asyncio.Event()
        self.client = make_client(settings.gemini_api_key or None)
        if self.client is None:
            log.warning("GEMINI_API_KEY not set → DSP baseline only")

    async def run(self) -> None:
        async with SessionLocal() as s:  # files interrupted by a restart go back to the queue
            await s.execute(update(File).where(File.status == "running").values(status="queued"))
            await s.commit()
        tasks: set[asyncio.Task] = set()
        while True:
            try:
                while len(tasks) < self.settings.worker_concurrency and (fid := await self._claim()):
                    t = asyncio.create_task(self._process(fid))
                    tasks.add(t)
                    t.add_done_callback(lambda t: (tasks.discard(t), self.wake.set()))
            except Exception:
                log.exception("claim failed")
            try:
                await asyncio.wait_for(self.wake.wait(), timeout=2.0)
            except TimeoutError:
                pass
            self.wake.clear()

    async def _claim(self) -> int | None:
        async with SessionLocal() as s:
            fid = (await s.execute(CLAIM_SQL)).scalar_one_or_none()
            await s.commit()
            return fid

    async def _process(self, fid: int) -> None:
        async with SessionLocal() as s:
            f = await s.get(File, fid)
            try:
                a = await asyncio.wait_for(analyze_file(Path(f.path), self.cfg, self.client),
                                           self.settings.file_timeout_s)
                s.add(ResultRow(file_id=f.id, **a.result.model_dump(),
                                raw={"facts": a.facts, "timings_ms": a.timings_ms,
                                     "votes": [c.output for c in a.calls], "ser": a.ser, "noise": a.noise, "overlap": a.overlap}))
                for c in a.calls:
                    s.add(LLMCallRow(file_id=f.id, **{k: v for k, v in c.to_dict().items() if k != "output"}))
                f.status, f.error = "done", None
                f.duration_s, f.latency_ms, f.cost_usd, f.degraded = a.duration_s, a.latency_ms, a.cost_usd, a.degraded
            except Exception as e:
                retry = not isinstance(e, DecodeError) and f.attempts < self.settings.max_attempts
                f.status = "queued" if retry else "error"
                f.error = (str(e) or type(e).__name__)[:1000]
                log.warning("file %s (%s) failed attempt %s: %s %s", f.id, f.name, f.attempts, f.error,
                            getattr(e, "detail", ""))
            await s.commit()
            await self._maybe_finish(s, f.batch_id)

    async def _maybe_finish(self, s, batch_id: str) -> None:
        open_ = await s.scalar(select(func.count()).select_from(File).where(
            File.batch_id == batch_id, File.status.in_(("queued", "running"))))
        if open_:
            return
        errors = await s.scalar(select(func.count()).select_from(File).where(
            File.batch_id == batch_id, File.status == "error"))
        await s.execute(update(Batch).where(Batch.id == batch_id, Batch.status == "running").values(
            status="done_with_errors" if errors else "done", finished_at=datetime.now(UTC)))
        await s.commit()
