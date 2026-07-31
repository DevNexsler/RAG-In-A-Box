"""The indexer heartbeat must track flow progress, not just per-doc progress.

Regression guard for #0127: the /health probe reports the indexer as frozen
(HTTP 503) when the heartbeat ages past INDEXER_HEARTBEAT_MAX_AGE while a live
indexer pid exists. The heartbeat used to be stamped only at flow start and once
per processed doc, leaving the two long, per-doc-heartbeat-free windows — the
source scan (before any doc is processed) and finalization/FTS (after the last
doc) — unstamped. On a large corpus either window can exceed the max age while
the indexer is progressing normally, producing a *false* 503 that ops "fixes"
by restarting (aborting the sweep). The scan must re-stamp as it goes.
"""

import json
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import flow_index_vault


class _FakeSource:
    """Minimal Source stand-in yielding `count` scannable records."""

    def __init__(self, name: str, count: int):
        self.name = name
        self._count = count

    def scan(self):
        for i in range(self._count):
            yield SimpleNamespace(
                doc_id=f"{i:06d}",
                natural_key=f"{self.name}/doc-{i}.txt",
                mtime=float(i),
                change_hash=f"h{i}",
                size=10,
                source_type="filesystem",
                metadata={"abs_path": f"/data/{self.name}/doc-{i}.txt", "ext": ".txt"},
            )


class _FakeDocIDStore:
    def __init__(self):
        self.registered: list[tuple] = []

    def register(self, doc_id, natural_key, source_name=None):
        self.registered.append((doc_id, natural_key, source_name))


def test_scan_stamps_heartbeat_periodically(tmp_path, monkeypatch):
    """A scan longer than one heartbeat interval must re-stamp several times.

    With per-doc heartbeats absent during the scan, the only stamps a healthy
    long scan can emit are these progress stamps — so their absence is exactly
    the false-503 bug.
    """
    stamps: list[int] = []
    real_write = flow_index_vault._write_heartbeat

    def _spy(index_root):
        stamps.append(1)
        real_write(index_root)

    monkeypatch.setattr(flow_index_vault, "_write_heartbeat", _spy)
    monkeypatch.setattr(flow_index_vault, "_SCAN_HEARTBEAT_EVERY", 100)

    src = _FakeSource("documents", count=350)
    store = _FakeDocIDStore()

    records, record_map = flow_index_vault._scan_and_register_sources(
        [src], store, tmp_path
    )

    # 350 records at every-100 => stamps at 100/200/300 plus a start and end
    # stamp: strictly more than the 2 boundary stamps a non-progressing scan
    # would emit. This is what keeps the heartbeat fresh through a long scan.
    assert len(stamps) >= 5
    assert flow_index_vault._heartbeat_path(tmp_path).exists()

    # Extraction correctness: every scanned record is namespaced, mapped, and
    # registered exactly once.
    assert len(records) == 350
    assert len(record_map) == 350
    assert records[0]["doc_id"] == "documents::000000"
    assert records[0]["rel_path"] == "documents/doc-0.txt"
    assert store.registered[0] == ("documents::000000", "documents/doc-0.txt", "documents")
    assert len(store.registered) == 350


def test_scan_stamps_heartbeat_even_when_empty(tmp_path, monkeypatch):
    """An empty scan still brackets the phase with a start+end stamp."""
    stamps: list[int] = []
    monkeypatch.setattr(
        flow_index_vault, "_write_heartbeat", lambda index_root: stamps.append(1)
    )

    records, record_map = flow_index_vault._scan_and_register_sources(
        [_FakeSource("documents", count=0)], _FakeDocIDStore(), tmp_path
    )

    assert records == []
    assert record_map == {}
    assert len(stamps) == 2  # start + complete, no per-record stamps


def test_heartbeat_persists_run_identity_and_queue_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEX_RUN_ID", "run-progress")
    flow_index_vault._RUNTIME.clear()
    flow_index_vault._initialize_run_progress()
    flow_index_vault._update_run_progress(
        queued=10,
        processed=3,
        skipped=1,
        phase="process",
    )

    flow_index_vault._write_heartbeat(tmp_path)

    payload = json.loads(flow_index_vault._heartbeat_path(tmp_path).read_text())
    assert payload["run_id"] == "run-progress"
    assert payload["phase"] == "process"
    assert payload["queued"] == 10
    assert payload["processed"] == 3
    assert payload["skipped"] == 1
    assert payload["updated_at"]


def test_completion_summary_names_queue_progress_and_elapsed():
    logger = Mock(spec=logging.Logger)

    flow_index_vault._log_run_completion(
        logger,
        run_id="run-complete",
        queued=10,
        processed=10,
        skipped=2,
        elapsed_seconds=12.5,
    )

    logger.info.assert_called_once_with(
        "Index run completion: run_id=%s queued=%d processed=%d skipped=%d "
        "elapsed=%.1fs completion=%.1f%%",
        "run-complete",
        10,
        10,
        2,
        12.5,
        100.0,
    )
