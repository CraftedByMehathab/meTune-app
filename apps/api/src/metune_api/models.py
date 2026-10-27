import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base

Json = JSON().with_variant(JSONB(), "postgresql")


def now() -> datetime:
    return datetime.now(UTC)


class Batch(Base):
    __tablename__ = "batches"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(200))
    created_by: Mapped[str] = mapped_column(String(100))
    # validated → running → done | done_with_errors ; invalid when the manifest is unusable
    status: Mapped[str] = mapped_column(String(20), default="validated", index=True)
    manifest: Mapped[dict] = mapped_column(Json, default=dict)
    has_labels: Mapped[bool] = mapped_column(Boolean, default=False)
    model_id: Mapped[str | None] = mapped_column(String(100))
    config_hash: Mapped[str | None] = mapped_column(String(20))
    git_sha: Mapped[str | None] = mapped_column(String(40))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    files: Mapped[list["File"]] = relationship(back_populates="batch", cascade="all, delete-orphan")


class File(Base):
    """One audio file. Also the job queue: workers claim rows with status='queued'."""
    __tablename__ = "files"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[str] = mapped_column(ForeignKey("batches.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(500))
    path: Mapped[str] = mapped_column(Text)
    # pending → queued → running → done | error
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    error: Mapped[str | None] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    label: Mapped[dict | None] = mapped_column(Json)
    duration_s: Mapped[float | None] = mapped_column(Float)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    cost_usd: Mapped[float | None] = mapped_column(Float)
    degraded: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)
    batch: Mapped[Batch] = relationship(back_populates="files")
    result: Mapped["ResultRow | None"] = relationship(back_populates="file", cascade="all, delete-orphan",
                                                      uselist=False)


class ResultRow(Base):
    __tablename__ = "results"
    file_id: Mapped[int] = mapped_column(ForeignKey("files.id", ondelete="CASCADE"), primary_key=True)
    emotional_tone: Mapped[str] = mapped_column(String(20))
    emotional_intensity: Mapped[str] = mapped_column(String(10))
    background_noise_present: Mapped[bool] = mapped_column(Boolean)
    background_noise_type: Mapped[str] = mapped_column(String(200))
    background_noise_severity: Mapped[str] = mapped_column(String(10))
    audio_quality: Mapped[str] = mapped_column(String(20))
    speaker_overlap_present: Mapped[bool] = mapped_column(Boolean)
    long_silence_present: Mapped[bool] = mapped_column(Boolean)
    confidence: Mapped[float] = mapped_column(Float)
    raw: Mapped[dict] = mapped_column(Json, default=dict)     # DSP facts, votes, timings
    file: Mapped[File] = relationship(back_populates="result")


class LLMCallRow(Base):
    __tablename__ = "llm_calls"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    file_id: Mapped[int] = mapped_column(ForeignKey("files.id", ondelete="CASCADE"), index=True)
    vote: Mapped[int] = mapped_column(Integer)
    model_id: Mapped[str] = mapped_column(String(100))
    audio_in: Mapped[int] = mapped_column(Integer, default=0)
    text_in: Mapped[int] = mapped_column(Integer, default=0)
    text_out: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
