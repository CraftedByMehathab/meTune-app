from datetime import datetime

from metune_pipeline.schema import Result
from pydantic import BaseModel

from .manifest import ManifestReport
from .scoring import Metrics


class Counts(BaseModel):
    pending: int = 0
    queued: int = 0
    running: int = 0
    done: int = 0
    error: int = 0


class Totals(BaseModel):
    audio_min: float
    cost_usd: float
    cost_per_min: float | None
    latency_s_per_min: float | None
    p50_latency_s: float | None
    p95_latency_s: float | None


class BatchSummary(BaseModel):
    id: str
    name: str
    status: str
    created_at: datetime
    finished_at: datetime | None
    has_labels: bool
    counts: Counts


class BatchDetail(BatchSummary):
    started_at: datetime | None
    model_id: str | None
    config_hash: str | None
    git_sha: str | None
    manifest: ManifestReport
    totals: Totals | None
    metrics: Metrics | None


class FileOut(BaseModel):
    id: int
    name: str
    status: str
    error: str | None
    attempts: int
    duration_s: float | None
    latency_ms: int | None
    cost_usd: float | None
    degraded: bool
    label: Result | None
    result: Result | None
