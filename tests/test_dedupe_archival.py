import json
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.dedupe as dedupe
from core.dedupe import archive_duplicate_file, archive_duplicate_record


def _assert_within_archive_root(archive_root: Path, archived_path: Path) -> None:
    resolved_root = archive_root.resolve()
    resolved_path = archived_path.resolve()
    assert resolved_path == resolved_root or resolved_root in resolved_path.parents


def test_archive_duplicate_record_writes_json_snapshot(tmp_path):
    archive_root = tmp_path / "duplicates"
    snapshot_path = archive_duplicate_record(
        archive_root=archive_root,
        source_name="comm_messages",
        canonical_doc_id="documents::00001",
        natural_key="message-123",
        record={
            "doc_id": "comm_messages::message-123",
            "source_name": "comm_messages",
            "body": "hello",
        },
    )

    assert snapshot_path.as_posix().endswith(
        "/postgres/comm_messages/documents%3A%3A00001/message-123.json"
    )
    payload = json.loads(snapshot_path.read_text())
    assert payload["source_name"] == "comm_messages"
    assert payload["natural_key"] == "message-123"


def test_reserve_unique_record_snapshot_path_reserves_name_immediately(tmp_path, monkeypatch):
    archive_root = tmp_path / "duplicates"
    archive_dir = archive_root / "postgres" / "comm_messages" / "documents%3A%3A00001"
    archive_dir.mkdir(parents=True)
    monkeypatch.setattr(dedupe, "_timestamp_token", lambda: "20260423T120000123456Z")

    first_path, first_fd = dedupe._reserve_unique_record_snapshot_path(
        archive_root,
        archive_dir,
        "message-123",
    )
    second_path, second_fd = dedupe._reserve_unique_record_snapshot_path(
        archive_root,
        archive_dir,
        "message-123",
    )
    try:
        assert first_path != second_path
        assert first_path.name == "message-123.json"
        assert second_path.name == "message-123__20260423T120000123456Z.json"
        assert first_path.exists()
        assert second_path.exists()
    finally:
        os.close(first_fd)
        os.close(second_fd)
        first_path.unlink(missing_ok=True)
        second_path.unlink(missing_ok=True)


def test_archive_duplicate_record_keeps_prior_snapshot_for_same_natural_key(tmp_path, monkeypatch):
    archive_root = tmp_path / "duplicates"
    monkeypatch.setattr(dedupe, "_timestamp_token", lambda: "20260423T120000123456Z")

    first_path = archive_duplicate_record(
        archive_root=archive_root,
        source_name="comm_messages",
        canonical_doc_id="documents::00001",
        natural_key="message-123",
        record={
            "doc_id": "comm_messages::message-123",
            "source_name": "wrong-source",
            "body": "first",
        },
    )
    second_path = archive_duplicate_record(
        archive_root=archive_root,
        source_name="comm_messages",
        canonical_doc_id="documents::00001",
        natural_key="message-123",
        record={
            "doc_id": "comm_messages::message-123",
            "source_name": "still-wrong",
            "body": "second",
        },
    )

    assert first_path != second_path
    assert first_path.name == "message-123.json"
    assert second_path.name == "message-123__20260423T120000123456Z.json"

    first_payload = json.loads(first_path.read_text())
    second_payload = json.loads(second_path.read_text())
    assert first_payload["body"] == "first"
    assert second_payload["body"] == "second"
    assert first_payload["natural_key"] == "message-123"
    assert second_payload["natural_key"] == "message-123"
    assert first_payload["source_name"] == "comm_messages"
    assert second_payload["source_name"] == "comm_messages"


def test_archive_duplicate_file_copies_without_removing_source(tmp_path):
    archive_root = tmp_path / "duplicates"
    source_root = tmp_path / "vault"
    source_root.mkdir()
    source_path = source_root / "Projects" / "a.pdf"
    source_path.parent.mkdir()
    source_path.write_bytes(b"pdf-bytes")

    archive_path = archive_duplicate_file(
        archive_root=archive_root,
        source_name="documents",
        canonical_doc_id="documents::00001",
        source_path=source_path,
        rel_path="Projects/a.pdf",
    )

    posix_path = archive_path.as_posix()
    assert "/filesystem/documents%3A%3A00001/" in posix_path
    assert posix_path.endswith("__Projects%2Fa.pdf")
    assert re.match(r"^\d{8}T\d{6}\d+Z__Projects%2Fa\.pdf$", archive_path.name)
    assert archive_path.read_bytes() == b"pdf-bytes"
    assert source_path.exists()
    metadata_path = archive_path.parent / f"{archive_path.name}.metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["source_name"] == "documents"
    assert metadata["original_rel_path"] == "Projects/a.pdf"


def test_archive_duplicate_file_uses_separate_metadata_sidecar_for_json_payload(tmp_path):
    archive_root = tmp_path / "duplicates"
    source_root = tmp_path / "vault"
    source_root.mkdir()
    source_path = source_root / "folder" / "src.json"
    source_path.parent.mkdir()
    source_path.write_bytes(b'{\"payload\":true}\n')

    archive_path = archive_duplicate_file(
        archive_root=archive_root,
        source_name="documents",
        canonical_doc_id="documents::00001",
        source_path=source_path,
        rel_path="folder/src.json",
    )

    metadata_path = archive_path.parent / f"{archive_path.name}.metadata.json"

    assert archive_path.read_bytes() == b'{\"payload\":true}\n'
    assert metadata_path.exists()
    assert metadata_path != archive_path
    metadata = json.loads(metadata_path.read_text())
    assert metadata["archived_path"] == str(archive_path)
    assert metadata["original_rel_path"] == "folder/src.json"


def test_archive_duplicate_file_blocks_canonical_doc_id_traversal(tmp_path):
    archive_root = tmp_path / "duplicates"
    source_path = tmp_path / "vault" / "secret.txt"
    source_path.parent.mkdir()
    source_path.write_text("top-secret")

    archive_path = archive_duplicate_file(
        archive_root=archive_root,
        source_name="documents",
        canonical_doc_id="../escape-target",
        source_path=source_path,
        rel_path="secret.txt",
    )

    _assert_within_archive_root(archive_root, archive_path)
    assert "/filesystem/%2E%2E%2Fescape-target/" in archive_path.as_posix()
    assert archive_path.read_text() == "top-secret"


def test_archive_duplicate_record_blocks_source_and_doc_id_traversal(tmp_path):
    archive_root = tmp_path / "duplicates"

    snapshot_path = archive_duplicate_record(
        archive_root=archive_root,
        source_name="../outside:source",
        canonical_doc_id="../../escape-target",
        natural_key="message-123",
        record={"body": "hello"},
    )

    _assert_within_archive_root(archive_root, snapshot_path)
    assert "/postgres/%2E%2E%2Foutside%3Asource/" in snapshot_path.as_posix()
    assert "/%2E%2E%2F%2E%2E%2Fescape-target/" in snapshot_path.as_posix()
    payload = json.loads(snapshot_path.read_text())
    assert payload["source_name"] == "../outside:source"
    assert payload["canonical_doc_id"] == "../../escape-target"


def test_archive_duplicate_file_bounds_long_rel_path_filename(tmp_path):
    archive_root = tmp_path / "duplicates"
    source_path = tmp_path / "vault" / "deep.txt"
    source_path.parent.mkdir()
    source_path.write_text("payload")
    rel_path = f"nested/{'a' * 320}.txt"

    archive_path = archive_duplicate_file(
        archive_root=archive_root,
        source_name="documents",
        canonical_doc_id="documents::00001",
        source_path=source_path,
        rel_path=rel_path,
    )

    assert archive_path.exists()
    assert len(archive_path.name) <= 180
    assert re.match(r"^\d{8}T\d{6}\d+Z__.+__[0-9a-f]{16}\.txt$", archive_path.name)
    metadata = json.loads((archive_path.parent / f"{archive_path.name}.metadata.json").read_text())
    assert metadata["original_rel_path"] == rel_path


def test_archive_duplicate_record_bounds_long_natural_key_filename(tmp_path):
    archive_root = tmp_path / "duplicates"
    natural_key = "message-" + ("x" * 320)

    snapshot_path = archive_duplicate_record(
        archive_root=archive_root,
        source_name="comm_messages",
        canonical_doc_id="documents::00001",
        natural_key=natural_key,
        record={"body": "hello"},
    )

    assert snapshot_path.exists()
    assert len(snapshot_path.name) <= 180
    assert re.match(r"^.+__[0-9a-f]{16}\.json$", snapshot_path.name)
    payload = json.loads(snapshot_path.read_text())
    assert payload["natural_key"] == natural_key


@pytest.mark.parametrize(
    ("reserved_name", "expected_name"),
    [
        ("CON", "seg__CON.json"),
        ("PRN", "seg__PRN.json"),
        ("AUX", "seg__AUX.json"),
        ("NUL", "seg__NUL.json"),
        ("COM1", "seg__COM1.json"),
        ("NUL.", "seg__NUL..json"),
    ],
)
def test_archive_duplicate_record_prefixes_windows_reserved_snapshot_stem(
    tmp_path,
    reserved_name,
    expected_name,
):
    archive_root = tmp_path / "duplicates"

    snapshot_path = archive_duplicate_record(
        archive_root=archive_root,
        source_name="comm_messages",
        canonical_doc_id="documents::00001",
        natural_key=reserved_name,
        record={"body": "hello"},
    )

    assert snapshot_path.name == expected_name
    payload = json.loads(snapshot_path.read_text())
    assert payload["natural_key"] == reserved_name


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("NUL.", "seg__NUL."),
        ("COM1 ", "seg__COM1 "),
        ("LPT9...", "seg__LPT9..."),
    ],
)
def test_prefix_windows_reserved_basename_strips_trailing_dots_and_spaces(value, expected):
    assert dedupe._prefix_windows_reserved_basename(value) == expected


def test_archive_duplicate_file_removes_copied_payload_when_metadata_write_fails(tmp_path, monkeypatch):
    archive_root = tmp_path / "duplicates"
    source_path = tmp_path / "vault" / "deep.txt"
    source_path.parent.mkdir()
    source_path.write_text("payload")

    def fail_metadata_write(target_path: Path, content: str) -> None:
        raise OSError(f"metadata write failed for {target_path}")

    monkeypatch.setattr(dedupe, "_write_text_atomic", fail_metadata_write)

    with pytest.raises(OSError, match="metadata write failed"):
        archive_duplicate_file(
            archive_root=archive_root,
            source_name="documents",
            canonical_doc_id="documents::00001",
            source_path=source_path,
            rel_path="nested/deep.txt",
        )

    assert source_path.exists()
    assert not any(path.is_file() for path in archive_root.rglob("*"))


def test_archive_duplicate_file_rejects_empty_rel_path(tmp_path):
    archive_root = tmp_path / "duplicates"
    source_path = tmp_path / "vault" / "deep.txt"
    source_path.parent.mkdir()
    source_path.write_text("payload")

    with pytest.raises(ValueError, match="rel_path"):
        archive_duplicate_file(
            archive_root=archive_root,
            source_name="documents",
            canonical_doc_id="documents::00001",
            source_path=source_path,
            rel_path="",
        )


def test_archive_duplicate_record_rejects_empty_natural_key(tmp_path):
    archive_root = tmp_path / "duplicates"

    with pytest.raises(ValueError, match="natural_key"):
        archive_duplicate_record(
            archive_root=archive_root,
            source_name="comm_messages",
            canonical_doc_id="documents::00001",
            natural_key="",
            record={"body": "hello"},
        )
