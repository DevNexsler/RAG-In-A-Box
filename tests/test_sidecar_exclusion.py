"""Communication sidecars must not be indexed as standalone documents.

Attachment *.json sidecars carry message/media/channel context for a sibling
binary and are consumed by the communication-context provider. Indexing them
as docs adds no search value (the context is already in the attachment's
enrichment) and re-processing thousands every run is waste.
"""

import json

from flow_index_vault import _is_communication_sidecar, scan_filesystem_records
from doc_id_store import DocIDStore


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def test_detects_communication_sidecar(tmp_path):
    p = tmp_path / "msg1__mm0.json"
    _write(p, {"schema_version": 2, "source": "zoho_mail",
              "message": {"source_message_id": "<a@x>"},
              "media": {"storage_path": "x.pdf"}})
    assert _is_communication_sidecar(p) is True


def test_detects_sidecar_without_schema_version(tmp_path):
    p = tmp_path / "s.json"
    _write(p, {"message": {"id": 1}, "media": {"media_index": 0}})
    assert _is_communication_sidecar(p) is True


def test_detects_legacy_flat_attachment_review_sidecar(tmp_path):
    """Pre-schema review manifests flatten message/media fields at top level."""
    p = tmp_path / "photo.jpg.json"
    _write(
        p,
        {
            "message_id": "msg-1",
            "provider_file_id": "file-1",
            "filename": "photo.jpg",
            "mime": "image/jpeg",
            "retrieval_status": "saved",
            "local_path": "/documents/photo.jpg",
            "classification": "routine progress photo",
            "visual_review": "Kitchen cabinet installation is complete.",
        },
    )
    assert _is_communication_sidecar(p) is True


def test_detects_legacy_flat_metadata_only_sidecar(tmp_path):
    p = tmp_path / "attachment.metadata-only.json"
    _write(
        p,
        {
            "source_message_id": "msg-2",
            "provider_file_id": "file-2",
            "filename": "attachment.heic",
            "mime_type": "image/heic",
            "retrieval_status": "metadata_only",
        },
    )
    assert _is_communication_sidecar(p) is True


def test_flat_document_metadata_is_not_a_legacy_sidecar(tmp_path):
    p = tmp_path / "export.json"
    _write(
        p,
        {
            "message_id": "business-record-1",
            "filename": "quarterly-report.pdf",
            "mime_type": "application/pdf",
            "classification": "confidential",
        },
    )
    assert _is_communication_sidecar(p) is False


def test_legit_json_data_is_not_a_sidecar(tmp_path):
    p = tmp_path / "config.json"
    _write(p, {"setting": "value", "items": [1, 2, 3]})
    assert _is_communication_sidecar(p) is False


def test_json_with_media_word_but_no_object_not_sidecar(tmp_path):
    p = tmp_path / "notes.json"
    _write(p, {"title": "media plan", "body": "discuss social media"})
    assert _is_communication_sidecar(p) is False


def test_scan_skips_sidecars_keeps_real_docs(tmp_path):
    # modern + token-bearing legacy sidecars, one text doc, one real JSON file
    _write(tmp_path / "email-attachments/dan/msg1__mm0.json",
           {"schema_version": 2, "message": {"id": 1}, "media": {"storage_path": "a.pdf"}})
    _write(
        tmp_path / "quo-attachments/photo.jpg@00old@.json",
        {
            "message_id": "msg-1",
            "provider_file_id": "file-1",
            "filename": "photo.jpg",
            "mime": "image/jpeg",
            "retrieval_status": "saved",
            "local_path": "/documents/photo.jpg",
            "visual_review": "Kitchen cabinet installation is complete.",
        },
    )
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "real.txt").write_text("genuine document content")
    _write(tmp_path / "data" / "config.json", {"k": "v"})

    registry = DocIDStore(tmp_path / "reg.db")
    records = scan_filesystem_records(
        tmp_path, ["**/*.txt", "**/*.json"], ["reg.db*"], doc_id_store=registry,
    )
    paths = {r["rel_path"] for r in records}
    # scan injects @ID@ into filenames, so match on stem, not full name.
    assert not any("msg1__mm0" in p for p in paths), "sidecar should be skipped"
    assert not any("photo.jpg" in p for p in paths), "legacy sidecar should be skipped"
    assert any("real" in p and p.endswith(".txt") for p in paths), "real text doc should index"
    assert any("config" in p and p.endswith(".json") for p in paths), "legit json data should index"
    assert len(records) == 2, f"only the 2 real docs, got {len(records)}: {paths}"
    registry.close()


def test_sidecar_detected_via_channel_counterparty_only(tmp_path):
    # Real sidecars sometimes surface channel/counterparty without an early
    # message/schema_version key — must still be detected.
    p = tmp_path / "s.json"
    _write(p, {"channel": {"id": "c1"}, "counterparty": {"name": "x"},
               "media": {"media_index": 0}, "context": {"nearest": []}})
    assert _is_communication_sidecar(p) is True


def test_sidecar_with_large_context_block_still_detected(tmp_path):
    # A big context array (nearby messages) must not hide the schema — the
    # detector parses the whole file, not just a head window.
    p = tmp_path / "big.json"
    big_ctx = {"nearest_nonempty_before": [{"text": "x" * 200} for _ in range(50)]}
    _write(p, {"context": big_ctx, "media": {"storage_path": "a.pdf"},
               "schema_version": 2, "message": {"id": 1}})
    assert _is_communication_sidecar(p) is True


def test_huge_non_sidecar_json_not_parsed_as_sidecar(tmp_path):
    p = tmp_path / "data.json"
    _write(p, {"rows": [{"v": i} for i in range(1000)]})  # no media key
    assert _is_communication_sidecar(p) is False
