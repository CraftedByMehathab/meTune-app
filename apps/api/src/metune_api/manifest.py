"""Upload handling + manifest validation (DR-011). Pure functions, no DB."""
import csv
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from metune_pipeline.decode import AUDIO_EXTS
from metune_pipeline.schema import Result
from pydantic import BaseModel, ValidationError


class ManifestIssue(BaseModel):
    level: Literal["error", "warning"]
    name: str
    message: str


class ManifestReport(BaseModel):
    csv_name: str | None = None
    has_labels: bool = False
    matched: int = 0                # audio + CSV row
    audio_without_row: int = 0      # processed, no label
    row_without_audio: int = 0      # cannot be processed
    unsupported: int = 0            # non-audio files ignored
    bad_json: int = 0               # label ignored
    to_process: int = 0
    fatal: str | None = None        # batch cannot run at all
    issues: list[ManifestIssue] = []


@dataclass
class PlannedFile:
    name: str
    path: Path
    label: dict | None


class UnsafePath(ValueError):
    pass


def safe_rel(name: str) -> Path:
    """Client-supplied path → safe relative path (no absolute, no '..')."""
    parts = PurePosixPath(name.replace("\\", "/")).parts
    if any(p == ".." for p in parts):
        raise UnsafePath(name)
    parts = [p for p in parts if p not in ("", ".", "/")]
    if not parts:
        raise UnsafePath(name)
    return Path(*parts)


def _ignored(rel: Path) -> bool:
    return any(p.startswith(".") or p == "__MACOSX" for p in rel.parts)


def extract_zip(zip_path: Path, dest: Path, max_bytes: int) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        total = sum(i.file_size for i in zf.infolist())
        if total > max_bytes:
            raise ValueError(f"ZIP expands to {total / 1e6:.0f} MB, over the {max_bytes / 1e6:.0f} MB limit")
        for info in zf.infolist():
            if info.is_dir():
                continue
            rel = safe_rel(info.filename)
            if _ignored(rel):
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as out:
                while chunk := src.read(1 << 20):
                    out.write(chunk)


def _parse_label(raw: str) -> tuple[dict | None, str | None]:
    if not raw or not raw.strip():
        return None, None
    try:
        return Result.model_validate(json.loads(raw)).model_dump(), None
    except (json.JSONDecodeError, ValidationError) as e:
        return None, str(e).splitlines()[0][:200]


def build_manifest(root: Path) -> tuple[ManifestReport, list[PlannedFile]]:
    rep = ManifestReport()
    all_files = [p for p in root.rglob("*") if p.is_file() and not _ignored(p.relative_to(root))]
    audio: dict[str, Path] = {}
    csvs: list[Path] = []
    for p in sorted(all_files):
        ext = p.suffix.lower()
        if ext == ".csv":
            csvs.append(p)
        elif ext in AUDIO_EXTS:
            if p.name in audio:
                rep.issues.append(ManifestIssue(level="error", name=p.name,
                                                message="duplicate file name; second copy ignored"))
                continue
            audio[p.name] = p
        else:
            rep.unsupported += 1
            rep.issues.append(ManifestIssue(level="error", name=p.name, message=f"unsupported format '{ext}'"))

    if not audio:
        rep.fatal = "No supported audio files found (wav, mp3, ogg, opus, m4a, flac, webm)."
        return rep, []

    rows: dict[str, str] = {}
    if csvs:
        csvs.sort(key=lambda p: (len(p.relative_to(root).parts), p.name))
        manifest = csvs[0]
        rep.csv_name = manifest.relative_to(root).as_posix()
        if len(csvs) > 1:
            rep.issues.append(ManifestIssue(level="warning", name=manifest.name,
                                            message=f"{len(csvs)} CSV files found; using {rep.csv_name}"))
        with manifest.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or "name" not in reader.fieldnames:
                rep.fatal = f"{rep.csv_name} has no 'name' column."
                return rep, []
            for row in reader:
                if name := (row.get("name") or "").strip():
                    rows[PurePosixPath(name).name] = row.get("result_json") or ""
    else:
        rep.issues.append(ManifestIssue(level="warning", name="(manifest)",
                                        message="No CSV manifest; processing all audio without labels"))

    planned: list[PlannedFile] = []
    for name, path in audio.items():
        label = None
        if name in rows:
            rep.matched += 1
            label, err = _parse_label(rows[name])
            if err:
                rep.bad_json += 1
                rep.issues.append(ManifestIssue(level="warning", name=name, message=f"label ignored: {err}"))
        elif csvs:
            rep.audio_without_row += 1
            rep.issues.append(ManifestIssue(level="warning", name=name, message="no CSV row; processed unlabeled"))
        planned.append(PlannedFile(name=name, path=path, label=label))

    for name in rows.keys() - audio.keys():
        rep.row_without_audio += 1
        rep.issues.append(ManifestIssue(level="error", name=name, message="listed in CSV but no audio file"))

    rep.has_labels = any(p.label for p in planned)
    rep.to_process = len(planned)
    return rep, planned
