"""Calendar retention over real small immutable backup archives."""

from pathlib import Path
import importlib.util
import io
import tarfile

SPEC = importlib.util.spec_from_file_location(
    "backup_retention", Path(__file__).resolve().parents[1] / "scripts/backup_retention.py"
)


def module():
    result = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(result)
    return result


def backup(root, stamp, content=b"fixture"):
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"index-{stamp}.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo("registry")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    return path


def test_first_successful_backup_after_first_week_is_monthly(tmp_path):
    m = module()
    latest = backup(tmp_path, "20260915-010000")
    m.rotate(tmp_path, latest)
    assert (tmp_path / "monthly" / latest.name).is_file()
    assert latest.stat().st_ino == (tmp_path / "monthly" / latest.name).stat().st_ino


def test_monthly_coverage_uses_distinct_calendar_months(tmp_path):
    m = module()
    for stamp in ("20260501-000000", "20260601-000000", "20260701-000000", "20260801-000000", "20260803-000000"):
        backup(tmp_path / "monthly", stamp)
    latest = backup(tmp_path, "20260915-010000")
    m.rotate(tmp_path, latest)
    assert sorted(p.name[6:12] for p in (tmp_path / "monthly").glob("*.tar.gz")) == ["202607", "202608", "202609"]
    assert (tmp_path / "monthly/index-20260801-000000.tar.gz").exists()


def test_weekly_and_daily_generations_are_bounded(tmp_path):
    m = module()
    for stamp in ("20260802-000000", "20260809-000000", "20260816-000000", "20260823-000000", "20260830-000000"):
        path = backup(tmp_path, stamp)
        m.rotate(tmp_path, path)
    assert len(list(tmp_path.glob("*.tar.gz"))) == 3
    assert len(list((tmp_path / "weekly").glob("*.tar.gz"))) == 4
    latest = tmp_path / "index-20260830-000000.tar.gz"
    assert latest.stat().st_ino == (tmp_path / "weekly" / latest.name).stat().st_ino


def test_failed_archive_never_prunes_existing_generations(tmp_path):
    m = module()
    old = backup(tmp_path / "monthly", "20250101-000000")
    bad = tmp_path / "index-20260915-010000.tar.gz"
    bad.write_bytes(b"broken")
    import pytest

    with pytest.raises((tarfile.TarError, EOFError, OSError)):
        m.rotate(tmp_path, bad)
    assert old.exists()


def test_unknown_and_partial_files_preserved(tmp_path):
    m = module()
    unknown = tmp_path / "notes.txt"
    unknown.write_text("keep")
    partial = tmp_path / "index-20250101-000000.tar.gz.partial"
    partial.write_text("partial")
    m.rotate(tmp_path, backup(tmp_path, "20260915-010000"))
    assert unknown.exists() and partial.exists()


def test_identical_existing_copies_share_storage(tmp_path):
    import shutil

    m = module()
    latest = backup(tmp_path, "20260915-010000")
    monthly = tmp_path / "monthly"
    monthly.mkdir()
    copy = monthly / latest.name
    shutil.copy2(latest, copy)
    m.rotate(tmp_path, latest)
    assert copy.stat().st_ino == latest.stat().st_ino


def test_symlinks_are_not_retention_targets(tmp_path):
    m = module()
    outside = backup(tmp_path / "outside", "20200101-000000")
    link = tmp_path / outside.name
    link.symlink_to(outside)
    m.rotate(tmp_path, backup(tmp_path, "20260915-010000"))
    assert link.is_symlink() and outside.exists()
