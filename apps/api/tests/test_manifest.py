import json
import zipfile

import pytest
from metune_api.manifest import UnsafePath, build_manifest, extract_zip, safe_rel

LABEL = {"emotional_tone": "upset", "emotional_intensity": "high", "background_noise_present": False,
         "background_noise_type": "", "background_noise_severity": "none", "audio_quality": "clear",
         "speaker_overlap_present": False, "long_silence_present": False, "confidence": 0.82}


def make_batch(root, csv_text: str | None, audio=("a.ogg", "b.ogg", "c.ogg"), extra=("notes.txt",)):
    (root / "batch").mkdir()
    for n in (*audio, *extra):
        (root / "batch" / n).write_bytes(b"x")
    if csv_text is not None:
        (root / "batch" / "labels.csv").write_text(csv_text)
    return root


def test_report_counts_every_case(tmp_path):
    label = json.dumps(LABEL).replace('"', '""')
    csv_text = f'name,result_json\na.ogg,"{label}"\nb.ogg,{{bad json\nmissing.ogg,\n'
    rep, planned = build_manifest(make_batch(tmp_path, csv_text))
    assert rep.fatal is None
    assert (rep.matched, rep.audio_without_row, rep.row_without_audio, rep.bad_json, rep.unsupported) == (2, 1, 1, 1, 1)
    assert rep.has_labels and rep.to_process == 3
    assert {p.name: bool(p.label) for p in planned} == {"a.ogg": True, "b.ogg": False, "c.ogg": False}


def test_no_csv_is_warning_not_fatal(tmp_path):
    rep, planned = build_manifest(make_batch(tmp_path, None))
    assert rep.fatal is None and len(planned) == 3 and not rep.has_labels


def test_csv_without_name_column_is_fatal(tmp_path):
    rep, planned = build_manifest(make_batch(tmp_path, "file,result_json\na.ogg,\n"))
    assert rep.fatal and planned == []


def test_no_audio_is_fatal(tmp_path):
    rep, _ = build_manifest(make_batch(tmp_path, "name\n", audio=()))
    assert rep.fatal


def test_safe_rel_blocks_traversal():
    assert str(safe_rel("folder/a.ogg")) == "folder/a.ogg"
    assert str(safe_rel("/abs/a.ogg")) == "abs/a.ogg"
    with pytest.raises(UnsafePath):
        safe_rel("../../etc/passwd")


def test_zip_slip_rejected(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../evil.ogg", b"x")
    with pytest.raises(UnsafePath):
        extract_zip(z, tmp_path / "out", max_bytes=10_000)


def test_zip_skips_macos_junk(tmp_path):
    z = tmp_path / "ok.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("batch/a.ogg", b"x")
        zf.writestr("__MACOSX/batch/._a.ogg", b"x")
    extract_zip(z, tmp_path / "out", max_bytes=10_000)
    assert [p.name for p in (tmp_path / "out").rglob("*") if p.is_file()] == ["a.ogg"]
