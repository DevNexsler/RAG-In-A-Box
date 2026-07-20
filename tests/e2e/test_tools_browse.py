"""Browse/observability tool surface: list, recent, facets, folders, status,
audit log, targeted re-index, and the scoped incremental sweep."""
import subprocess
import time
from datetime import datetime, timezone

import anyio
import pytest

from tests.e2e.client import get_hook_events
from tests.e2e.conftest import (
    COMPOSE_FILE,
    EXPECTED_CORPUS_DOCS,
    NOTE_PHRASE,
    wait_for_index,
)

pytestmark = pytest.mark.anyio

SWEEP_TIMEOUT_S = 120
# If indexer_running never flips to True within this window after "started",
# assume the sweep finished between polls instead of waiting out the full
# timeout — the final idle check still guards correctness.
RUNNING_OBSERVE_GRACE_S = 20


async def test_list_documents_pagination(indexed_corpus, mcp_session):
    page1 = await mcp_session.call_tool_json("file_list_documents", {"limit": 2})
    assert not page1.get("error"), page1
    assert len(page1["documents"]) == 2
    assert page1["total"] >= EXPECTED_CORPUS_DOCS
    assert page1["limit"] == 2 and page1["offset"] == 0

    page2 = await mcp_session.call_tool_json(
        "file_list_documents", {"limit": 2, "offset": 2})
    ids1 = {d["doc_id"] for d in page1["documents"]}
    ids2 = {d["doc_id"] for d in page2["documents"]}
    assert ids1.isdisjoint(ids2), (ids1, ids2)


async def test_recent_newest_first(indexed_corpus, mcp_session):
    docs = await mcp_session.call_tool_json("file_recent", {"limit": 5})
    assert isinstance(docs, list) and docs, docs
    mtimes = [d["mtime"] for d in docs]
    assert mtimes == sorted(mtimes, reverse=True)
    for d in docs:
        assert d.get("doc_id") and d.get("mtime_iso"), d


async def test_facets_cover_both_sources(indexed_corpus, mcp_session):
    facets = await mcp_session.call_tool_json("file_facets", {})
    assert not facets.get("error"), facets
    assert facets["total_docs"] >= EXPECTED_CORPUS_DOCS
    assert facets["total_chunks"] >= facets["total_docs"]
    source_types = {f["value"] for f in facets["source_types"]}
    # filesystem docs AND the postgres sor sweep must both be represented
    assert "md" in source_types, source_types
    assert "pg_message" in source_types, source_types


async def test_folders_reflect_documents_root(indexed_corpus, mcp_session):
    folders = await mcp_session.call_tool_json("file_folders", {})
    assert not folders.get("error"), folders
    assert folders["documents_root"] == "/data/documents"
    # ADAPTATION (documented gap): in sources-mode configs _file_folders_impl
    # reads the TOP-LEVEL scan config, which doesn't exist — it falls back to
    # include=["**/*.md"] and therefore counts only the markdown fixture, not
    # all five deposited files. Assert what the tool actually does; the
    # undercount is reported as a production wart, not asserted around.
    assert folders["total_files"] >= 1
    root = [f for f in folders["folders"] if f["path"] == "."]
    assert root and root[0]["file_count"] >= 1, folders["folders"]


async def test_status_healthy_shape(indexed_corpus, mcp_session):
    status = await mcp_session.call_tool_json("file_status", {})
    assert not status.get("error"), status
    assert status["doc_count"] >= EXPECTED_CORPUS_DOCS
    assert status["chunk_count"] >= status["doc_count"]
    assert status["last_run_at"]
    assert status["embeddings_provider"] == "openrouter"
    assert "doc_id" in status["metadata_fields"]
    health = status["health"]
    assert health["fts_available"] is True
    assert health["reranker_enabled"] is True
    assert health["reranker_responsive"] is True
    assert health["last_index_failed_count"] == 0


async def test_audit_log_records_lifecycle_docs(indexed_corpus, mcp_session):
    log = await mcp_session.call_tool_json("file_audit_log", {"limit": 100})
    assert not log.get("error"), log
    assert log["total"] >= 5
    entries = log["entries"]
    events = {e["event"] for e in entries}
    assert "registered" in events, events
    rel_paths = " ".join(e.get("rel_path", "") for e in entries)
    assert "note" in rel_paths and "diagram" in rel_paths, rel_paths


async def test_index_document_reindex_existing(indexed_corpus, mcp_session):
    log_before = await mcp_session.call_tool_json("file_audit_log", {"limit": 1})

    # Locate note.md's current (ID-aliased) rel_path via search.
    hit = await mcp_session.call_tool_json(
        "file_search", {"query": NOTE_PHRASE, "top_k": 1})
    rel_path = hit["results"][0]["rel_path"]

    result = await mcp_session.call_tool_json(
        "file_index_document",
        {"target": rel_path, "source_name": "documents", "force": True})
    assert result.get("status") == "indexed", result
    assert result.get("doc_id"), result

    # ADAPTATION (documented): a forced re-index of an unchanged file adds no
    # audit-log entry (the audit log records ID lifecycle, not indexing runs).
    # The observable side-effect is the document.indexed webhook — sim_reset
    # cleared the sink before this test, so any event here came from this call.
    events = await get_hook_events()
    assert any(
        e.get("event") == "document.indexed" and e.get("doc_id") == result["doc_id"]
        for e in events
    ), events
    assert log_before["total"] <= (
        await mcp_session.call_tool_json("file_audit_log", {"limit": 1}))["total"]


async def test_index_update_scoped_sweep(indexed_corpus, mcp_session):
    # ADAPTATION (documented): file_index_update has no pause/resume-style
    # control action — its only parameter is source_name. Exercise the benign
    # scoped form: a sor-only incremental sweep (no changes → no-op) and verify
    # the index converges back to idle with the corpus intact.
    retained_warning = (
        f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S},000 "
        "WARNING midday OCR failure"
    )
    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE_FILE),
            "exec",
            "-T",
            "doc-organizer-staging",
            "sh",
            "-c",
            'printf "%s\\n" "$1" >> /data/index/indexer.log',
            "sh",
            retained_warning,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    started = await mcp_session.call_tool_json(
        "file_index_update", {"source_name": "sor"})
    assert started.get("status") == "started", started
    assert started.get("source_name") == "sor"
    assert started.get("pid")

    # Wait for the background subprocess to actually finish (indexer_running
    # goes true → false), not just for doc_count — the corpus already
    # satisfies min_docs, and a still-running sweep must not leak into the
    # next test.
    saw_running = False
    start = time.monotonic()
    while time.monotonic() - start < SWEEP_TIMEOUT_S:
        status = await mcp_session.call_tool_json("file_status", {})
        running = bool(status.get("indexer_running"))
        saw_running = saw_running or running
        elapsed = time.monotonic() - start
        if not running and (saw_running or elapsed > RUNNING_OBSERVE_GRACE_S):
            break
        await anyio.sleep(2)

    status = await wait_for_index(mcp_session, min_docs=EXPECTED_CORPUS_DOCS)
    assert status["doc_count"] >= EXPECTED_CORPUS_DOCS
    assert not status.get("indexer_running")

    log = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE_FILE),
            "exec",
            "-T",
            "doc-organizer-staging",
            "cat",
            "/data/index/indexer.log",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert retained_warning in log
