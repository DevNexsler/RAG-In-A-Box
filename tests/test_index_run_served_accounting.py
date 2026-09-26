"""Mid-sweep served-lane writes must appear on Index run completion counters.

Regression guard for #2692: a sweep that serves durable queue requests between
document batches was writing real chunks (Inserted/Upserted lines, enrichment,
hook delivery) while ``Index stats:`` / ``Index run completion:`` reported only
the scan lane. Root cause: ``_build_single_doc_runtime`` cleared ``_RUNTIME``
without preserving the sweep's ``run_progress``, so ``_record_index_write`` at
the store write seam was a no-op for every served document; restoring the
saved sweep runtime then brought back the pre-checkpoint counters.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import flow_index_vault as fiv
from core.index_request_queue import IndexRequestQueue


def _scan_and_serve_config(tmp_path: Path) -> dict:
    return {
        "index_root": str(tmp_path),
        "lancedb": {"table": "chunks"},
        "index_queue": {},
        "chunking": {"max_chars": 1800, "overlap": 200},
        "enrichment": {"enabled": False},
        "communication_context": {},
        "pdf": {},
        "sources": [
            {
                "type": "filesystem",
                "name": "documents",
                "root": str(tmp_path / "docs"),
                "scan": {"include": ["**/*"], "exclude": []},
            }
        ],
    }


@pytest.fixture
def sweep_with_progress(tmp_path):
    """A live sweep runtime that already counted one scan-lane write."""
    store = MagicMock()
    store.set_memory_observer = MagicMock()
    fiv._RUNTIME.clear()
    fiv._RUNTIME.update(
        {
            "store": store,
            "doc_id_store": MagicMock(),
            "index_root": tmp_path,
            "config": {"index_root": str(tmp_path)},
        }
    )
    fiv._initialize_run_progress()
    fiv._update_run_progress(phase="process", queued=2, processed=1, skipped=0)
    fiv._record_index_write(2)
    yield fiv._RUNTIME
    fiv._RUNTIME.clear()


def test_single_doc_runtime_preserves_sweep_run_progress(tmp_path, sweep_with_progress):
    """The write-seam counters must survive the single-doc runtime rebuild."""
    progress = fiv._RUNTIME["run_progress"]
    lock = fiv._RUNTIME["run_progress_lock"]
    config = _scan_and_serve_config(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    target = docs / "served.txt"
    target.write_text("served body\n")
    record = {
        "doc_id": "documents::served",
        "rel_path": "served.txt",
        "abs_path": str(target),
        "mtime": 1.0,
        "size": target.stat().st_size,
        "ext": "txt",
        "source_type": "filesystem",
        "source_name": "documents",
    }
    fake_source = SimpleNamespace(name="documents")

    with patch("flow_index_vault.build_embed_provider", return_value=MagicMock()), \
         patch("flow_index_vault.build_ocr_provider", return_value=None), \
         patch("flow_index_vault.build_media_provider", return_value=None), \
         patch("flow_index_vault.SentenceSplitter", return_value=MagicMock()), \
         patch("sources.build_source", return_value=fake_source), \
         patch(
             "sources.filesystem._communication_sidecar_metadata",
             return_value={},
         ), \
         patch(
             "flow_index_vault._targeted_communication_context_provider",
             return_value=(None, False),
         ), \
         patch("flow_index_vault._get_logger", return_value=logging.getLogger("test")):
        fiv._build_single_doc_runtime(
            config,
            sweep_with_progress["store"],
            sweep_with_progress["doc_id_store"],
            "documents",
            record,
        )

    assert fiv._RUNTIME.get("run_progress") is progress
    assert fiv._RUNTIME.get("run_progress_lock") is lock
    fiv._record_index_write(5)
    assert progress["indexed_docs"] == 2
    assert progress["indexed_chunks"] == 7


def test_mid_sweep_served_writes_appear_on_completion_line(
    tmp_path, sweep_with_progress, caplog
):
    """Acceptance: Index run completion accounts for scan work and served requests.

    Drives both lanes at the checkpoint seam: the sweep already recorded one
    scan write, then serves a queued request that writes five chunks through
    the real single-doc runtime rebuild + write-seam counter.
    """
    config = _scan_and_serve_config(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    target = docs / "served.txt"
    target.write_text("served body\n")
    queue = IndexRequestQueue(tmp_path)
    queue.enqueue("chunks", "documents", "served.txt")
    fake_source = SimpleNamespace(name="documents")

    def serve_through_real_runtime(passed_config, request, store, registry):
        record = {
            "doc_id": "documents::served",
            "rel_path": request.target,
            "abs_path": str(target),
            "mtime": 1.0,
            "size": target.stat().st_size,
            "ext": "txt",
            "source_type": "filesystem",
            "source_name": "documents",
        }
        fiv._build_single_doc_runtime(
            passed_config, store, registry, request.source_name, record
        )
        fiv._record_index_write(5)
        fiv._note_indexed_incomplete(record["doc_id"])
        return {
            "status": "indexed",
            "doc_id": record["doc_id"],
            "rel_path": request.target,
        }

    with patch.object(
        fiv, "_index_document_unlocked", side_effect=serve_through_real_runtime
    ), patch("flow_index_vault.build_embed_provider", return_value=MagicMock()), \
         patch("flow_index_vault.build_ocr_provider", return_value=None), \
         patch("flow_index_vault.build_media_provider", return_value=None), \
         patch("flow_index_vault.SentenceSplitter", return_value=MagicMock()), \
         patch("sources.build_source", return_value=fake_source), \
         patch(
             "sources.filesystem._communication_sidecar_metadata",
             return_value={},
         ), \
         patch(
             "flow_index_vault._targeted_communication_context_provider",
             return_value=(None, False),
         ), \
         patch("flow_index_vault._get_logger", return_value=logging.getLogger("test")):
        assert fiv._service_index_queue(config, "chunks") == 1

    # Sweep runtime restored, with served write folded into the same counters.
    assert fiv._RUNTIME["store"] is sweep_with_progress["store"]
    assert fiv._RUNTIME["indexed_incomplete"] == {"documents::served"}
    snap = fiv._run_progress_snapshot()
    assert snap["indexed_docs"] == 2, snap
    assert snap["indexed_chunks"] == 7, snap
    assert snap["queued"] == 3, snap  # scan 2 + served 1
    assert snap["processed"] == 2, snap  # scan 1 + served 1

    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test.completion")
    with caplog.at_level(logging.INFO, logger=logger.name):
        fiv._log_run_completion(
            logger,
            run_id=str(snap["run_id"]),
            queued=int(snap["queued"]),
            processed=int(snap["processed"]),
            skipped=int(snap["skipped"] or 0),
            indexed_docs=int(snap["indexed_docs"]),
            indexed_chunks=int(snap["indexed_chunks"]),
            elapsed_seconds=12.0,
        )
    line = next(
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("Index run completion:")
    )
    assert "queued=3" in line, line
    assert "processed=2" in line, line
    assert "indexed_docs=2" in line, line
    assert "indexed_chunks=7" in line, line
