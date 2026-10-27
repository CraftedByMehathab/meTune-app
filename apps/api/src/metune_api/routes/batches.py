import asyncio
import csv
import io
import json
import shutil
from datetime import UTC, datetime
from typing import Literal

import numpy as np
from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile
from fastapi import File as Upload
from fastapi.responses import Response
from metune_pipeline import config_hash
from metune_pipeline.schema import FIELDS, Result
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..auth import require_user
from ..db import get_session
from ..manifest import ManifestReport, UnsafePath, build_manifest, extract_zip, safe_rel
from ..models import Batch, File
from ..schemas import BatchDetail, BatchSummary, Counts, FileOut, Totals
from ..scoring import score
from ..settings import Settings, get_settings

router = APIRouter(prefix="/api/batches", tags=["batches"], dependencies=[Depends(require_user)])


def _result_dict(f: File) -> dict | None:
    if not f.result:
        return None
    return {k: getattr(f.result, k) for k in FIELDS}


async def _counts(s: AsyncSession, batch_ids: list[str]) -> dict[str, Counts]:
    rows = await s.execute(select(File.batch_id, File.status, func.count())
                           .where(File.batch_id.in_(batch_ids)).group_by(File.batch_id, File.status))
    out = {b: Counts() for b in batch_ids}
    for bid, status, n in rows:
        setattr(out[bid], status, n)
    return out


async def _get_batch(s: AsyncSession, batch_id: str) -> Batch:
    b = await s.get(Batch, batch_id)
    if not b:
        raise HTTPException(404, "Batch not found")
    return b


def _default_name(files: list[UploadFile]) -> str:
    """Folder upload → top folder ("trial"); ZIP → its stem; single loose file → its name."""
    first = (files[0].filename or "batch").replace("\\", "/").strip("/")
    if "/" in first:
        return first.split("/", 1)[0]
    return first.removesuffix(".zip").removesuffix(".ZIP") if len(files) == 1 else first


def _save_uploads(files: list[UploadFile], root, max_bytes: int) -> None:
    """ZIP → extract; otherwise files keep their (folder-relative) names."""
    src = root / "_upload"
    src.mkdir(parents=True)
    total = 0
    for up in files:
        rel = safe_rel(up.filename or "unnamed")
        target = src / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as out:
            shutil.copyfileobj(up.file, out, 1 << 20)
        total += target.stat().st_size
        if total > max_bytes:
            raise ValueError(f"Upload exceeds {max_bytes // 1_000_000} MB")
    zips = [p for p in src.rglob("*.zip")]
    files_dir = root / "files"
    if len(files) == 1 and len(zips) == 1:
        extract_zip(zips[0], files_dir, max_bytes * 4)
        shutil.rmtree(src)
    else:
        src.rename(files_dir)


@router.post("", response_model=BatchDetail, status_code=201)
async def create_batch(
    request: Request,
    files: list[UploadFile] = Upload(..., description="One ZIP, or every file of a folder"),
    name: str | None = Form(None),
    s: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
    user: str = Depends(require_user),
) -> BatchDetail:
    batch = Batch(name=(name or _default_name(files))[:200], created_by=user)
    s.add(batch)
    await s.flush()
    root = settings.upload_dir / batch.id
    try:
        await asyncio.to_thread(_save_uploads, files, root, settings.max_upload_mb * 1_000_000)
        report, planned = await asyncio.to_thread(build_manifest, root / "files")
    except (UnsafePath, ValueError, OSError) as e:
        shutil.rmtree(root, ignore_errors=True)
        await s.rollback()
        raise HTTPException(400, f"Upload rejected: {e}")
    batch.manifest = report.model_dump()
    batch.has_labels = report.has_labels
    batch.status = "invalid" if report.fatal else "validated"
    for p in planned:
        s.add(File(batch_id=batch.id, name=p.name, path=str(p.path), label=p.label))
    await s.commit()
    return await get_batch(batch.id, s)


@router.get("", response_model=list[BatchSummary])
async def list_batches(s: AsyncSession = Depends(get_session)) -> list[BatchSummary]:
    batches = (await s.scalars(select(Batch).order_by(Batch.created_at.desc()).limit(100))).all()
    counts = await _counts(s, [b.id for b in batches])
    return [BatchSummary(id=b.id, name=b.name, status=b.status, created_at=b.created_at,
                         finished_at=b.finished_at, has_labels=b.has_labels, counts=counts[b.id])
            for b in batches]


@router.post("/{batch_id}/start", response_model=BatchDetail)
async def start_batch(batch_id: str, request: Request, s: AsyncSession = Depends(get_session),
                      settings: Settings = Depends(get_settings)) -> BatchDetail:
    b = await _get_batch(s, batch_id)
    if b.status != "validated":
        raise HTTPException(409, f"Batch is {b.status}; only validated batches can start")
    cfg = request.app.state.cfg
    b.status, b.started_at = "running", datetime.now(UTC)
    b.model_id = cfg.llm_model if request.app.state.worker.client else "dsp-baseline"
    b.config_hash, b.git_sha = config_hash(cfg), settings.git_sha
    await s.execute(update(File).where(File.batch_id == b.id, File.status == "pending").values(status="queued"))
    await s.commit()
    request.app.state.worker.wake.set()
    return await get_batch(batch_id, s)


@router.get("/{batch_id}", response_model=BatchDetail)
async def get_batch(batch_id: str, s: AsyncSession = Depends(get_session)) -> BatchDetail:
    b = await _get_batch(s, batch_id)
    files = (await s.scalars(select(File).where(File.batch_id == batch_id)
                             .options(selectinload(File.result)))).all()
    done = [f for f in files if f.status == "done"]
    totals = None
    if done:
        minutes = sum(f.duration_s or 0 for f in done) / 60
        cost = sum(f.cost_usd or 0 for f in done)
        lat = np.array([f.latency_ms or 0 for f in done]) / 1000
        totals = Totals(audio_min=round(minutes, 2), cost_usd=round(cost, 5),
                        cost_per_min=round(cost / minutes, 5) if minutes else None,
                        latency_s_per_min=round(float(lat.sum()) / minutes, 2) if minutes else None,
                        p50_latency_s=round(float(np.percentile(lat, 50)), 2),
                        p95_latency_s=round(float(np.percentile(lat, 95)), 2))
    metrics = score([(f.label, _result_dict(f)) for f in done if f.label]) if b.has_labels else None
    return BatchDetail(
        id=b.id, name=b.name, status=b.status, created_at=b.created_at, started_at=b.started_at,
        finished_at=b.finished_at, has_labels=b.has_labels, counts=(await _counts(s, [b.id]))[b.id],
        model_id=b.model_id, config_hash=b.config_hash, git_sha=b.git_sha,
        manifest=ManifestReport(**b.manifest), totals=totals, metrics=metrics,
    )


@router.get("/{batch_id}/files", response_model=list[FileOut])
async def list_files(batch_id: str, s: AsyncSession = Depends(get_session)) -> list[FileOut]:
    await _get_batch(s, batch_id)
    files = (await s.scalars(select(File).where(File.batch_id == batch_id).order_by(File.name)
                             .options(selectinload(File.result)))).all()
    return [FileOut(id=f.id, name=f.name, status=f.status, error=f.error, attempts=f.attempts,
                    duration_s=f.duration_s, latency_ms=f.latency_ms, cost_usd=f.cost_usd,
                    degraded=f.degraded, label=f.label, result=_result_dict(f)) for f in files]


@router.get("/{batch_id}/download")
async def download(batch_id: str, format: Literal["csv", "json"] = "csv",
                   s: AsyncSession = Depends(get_session)) -> Response:
    b = await _get_batch(s, batch_id)
    files = (await s.scalars(select(File).where(File.batch_id == batch_id).order_by(File.name)
                             .options(selectinload(File.result)))).all()
    stem = f"metune_{b.name.rsplit('.', 1)[0]}_{b.id[:8]}"
    if format == "json":
        body = [{"name": f.name, "status": f.status, "error": f.error,
                 "result": Result(**_result_dict(f)).model_dump() if f.result else None} for f in files]
        return Response(json.dumps(body, indent=2), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="{stem}.json"'})
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["name", "result_json"])            # same shape as the input manifest
    for f in files:
        if f.result:
            w.writerow([f.name, json.dumps(_result_dict(f))])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{stem}.csv"'})
