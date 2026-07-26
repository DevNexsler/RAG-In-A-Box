"""Webhook delivery + provider-fault recovery, driven from outside.

Fault notes:
  - provider-sim faults auto-exhaust (times counts down and the armed entry is
    removed); the autouse sim_reset fixture ALSO resets before and after every
    test, so no fault can leak across tests even on assertion failure.
  - hook-delivery evidence for the initial sweep is the snapshot captured by
    indexed_corpus itself (the per-test /admin/reset wipes the live sink, so
    asserting on a later GET /hooks/received would race with reset ordering).
"""
import json
import os
import tempfile
import uuid
from pathlib import Path

import httpx
import pytest

from tests.e2e.client import get_hook_events, search_hits
from tests.e2e.conftest import E2E_SIM_URL, _compose_cp_into_documents

pytestmark = pytest.mark.anyio
E2E_REAL = os.environ.get("E2E_REAL") == "1"


async def _arm_fault(route_prefix: str, fault: str, times: int):
    async with httpx.AsyncClient(timeout=10) as sim:
        resp = await sim.post(f"{E2E_SIM_URL}/admin/fault", json={
            "route_prefix": route_prefix, "fault": fault, "times": times})
        assert resp.status_code == 200, resp.text
        assert resp.json()["ok"] is True


async def _upload_and_index(api, mcp_session, name: str, content: bytes) -> dict:
    # Content salting for re-run safety: a repeat run against a live stack
    # would otherwise hit the idempotent unchanged-skip path (the previous
    # run already indexed identical bytes) and fail the "indexed" assertion.
    content = content + f"\nrun-salt: {uuid.uuid4().hex}\n".encode()
    resp = await api.post("/api/upload", files={"file": (name, content)})
    assert resp.status_code == 201, resp.text
    result = await mcp_session.call_tool_json("file_index_document", {
        "target": name, "source_name": "documents"})
    assert result.get("status") == "indexed", result
    return result


async def test_hooks_delivered_for_initial_sweep(indexed_corpus):
    events = [e for e in indexed_corpus["hook_events"]
              if e.get("event") == "document.indexed"]
    # all five fixture files must have produced a document.indexed webhook
    assert len(events) >= 5, f"only {len(events)} document.indexed events"
    for e in events:
        assert e.get("doc_id"), e
        assert isinstance(e.get("chunks"), list) and e["chunks"], e["doc_id"]
    rel_paths = " ".join(e.get("rel_path", "") for e in events)
    for stem in ("note", "report", "diagram", "clip"):
        assert stem in rel_paths, f"{stem} missing from hook events: {rel_paths}"


async def test_recovery_from_embeddings_429(indexed_corpus, api, mcp_session):
    # Two 429s on embeddings: core/resilience.call_with_retry must absorb them.
    await _arm_fault("/api/v1/embeddings", "429", times=2)

    content = b"# Permit\n\nThe xylophone glacier permit was approved yesterday.\n"
    result = await _upload_and_index(api, mcp_session, "fault-note.md", content)

    # Retry-layer proof independent of search: the full pipeline (extract →
    # enrich → embed → upsert) completed and emitted document.indexed despite
    # the two 429s (sink was reset before this test).
    events = await get_hook_events()
    delivered = [e for e in events
                 if e.get("event") == "document.indexed" and e.get("doc_id") == result["doc_id"]]
    assert delivered and delivered[0]["chunks"], events

    # Documented contract of file_index_document: the doc "must become ...
    # searchable within seconds" — single-doc indexing must invalidate the
    # serving cache (regression test for commit 66ce76e).
    payload = await mcp_session.call_tool_json(
        "file_search", {"query": "xylophone glacier permit", "top_k": 5})
    assert search_hits(payload, "fault-note"), payload["results"]

    # no failure-shaped audit events for this doc (audit records ID lifecycle;
    # rename_failed / collision are the failure signals it can carry)
    log = await mcp_session.call_tool_json(
        "file_audit_log", {"doc_id": result["doc_id"], "limit": 50})
    audit_events = {e["event"] for e in log["entries"]}
    assert audit_events, log
    assert not audit_events & {"rename_failed", "collision"}, log["entries"]


@pytest.mark.skipif(
    E2E_REAL,
    reason="enrichment is live in real mode; the sim's chat fault can't be injected into real OpenRouter",
)
async def test_degraded_enrichment_still_indexes(indexed_corpus, api, mcp_session):
    # Three garbage responses on chat/completions: enrichment cannot succeed,
    # but indexing must degrade gracefully — the document still gets chunked,
    # embedded, and searchable (verified: file_index_document returns
    # status=indexed; enrichment failure is recorded, not fatal).
    await _arm_fault("/api/v1/chat/completions", "garbage", times=3)

    content = b"# Timetable\n\nThe kaleidoscope ferry timetable changes at dawn.\n"
    result = await _upload_and_index(api, mcp_session, "degraded-note.md", content)
    assert result.get("doc_id")

    # Degradation proof independent of search: document.indexed still fired.
    events = await get_hook_events()
    assert any(e.get("event") == "document.indexed" and e.get("doc_id") == result["doc_id"]
               for e in events), events

    payload = await mcp_session.call_tool_json(
        "file_search", {"query": "kaleidoscope ferry timetable", "top_k": 5})
    hits = search_hits(payload, "degraded-note")
    assert hits, payload["results"]

    # the doc round-trips fully despite the failed enrichment
    chunk = await mcp_session.call_tool_json(
        "file_get_chunk", {"doc_id": hits[0]["doc_id"], "loc": hits[0]["loc"]})
    assert "kaleidoscope" in chunk["text"].lower()


@pytest.mark.skipif(
    E2E_REAL,
    reason="enrichment is live in real mode; simulator truncation cannot be armed",
)
async def test_reasoning_only_enrichment_retries_with_populated_facets(
    indexed_corpus,
    api,
    mcp_session,
):
    await _arm_fault("/api/v1/chat/completions", "reasoning_only", times=1)

    content = b"# Renewal\n\nTenant requested a lease renewal for the apartment.\n"
    result = await _upload_and_index(api, mcp_session, "truncation-note.md", content)

    payload = await mcp_session.call_tool_json(
        "file_search", {"query": "tenant lease renewal", "top_k": 5}
    )
    hits = search_hits(payload, "truncation-note")
    assert hits, payload["results"]
    chunk = await mcp_session.call_tool_json(
        "file_get_chunk", {"doc_id": hits[0]["doc_id"], "loc": hits[0]["loc"]}
    )
    assert chunk["enr_summary"]
    assert chunk["enr_doc_type"]


@pytest.mark.skipif(
    E2E_REAL,
    reason="the real embeddings API is the thing being simulated here",
)
async def test_oversized_conversation_context_still_indexes(indexed_corpus, mcp_session):
    """#0569: an attachment whose stored conversation context is larger than the
    embed model's context window must still land in the index.

    That context block is embedded as ONE un-chunked input. Un-bounded, it
    overruns the model's window and the provider rejects the whole batch with a
    400 — deterministically, so the doc is re-fetched, re-OCR'd, re-enriched and
    re-embedded on every run and is never searchable. Both prod poison-pill docs
    (laura-sanchez msg693) failed exactly here.
    """
    stem = f"oversized-context-{uuid.uuid4().hex[:8]}"
    phrase = "zephyr cantilever manifest"
    sent_at = "2026-06-01T10:00:30Z"
    # A channel with no indexed messages, so the targeted index path falls back
    # to the sidecar's OWN stored context block — the prod shape for a freshly
    # deposited attachment whose channel history is not in the index yet.
    channel = f"ops-{stem}"
    sidecar = {
        "schema_version": 1,
        "source": "quo",
        "message": {
            "source_message_id": f"{stem}-msg",
            "sender": "Field Agent",
            "sent_at": sent_at,
        },
        "channel": {"source_channel_id": channel},
        "media": {
            "media_index": 0,
            "media_type": "document",
            "original_filename": f"{stem}.md",
        },
        "context": {
            "schema_version": 1,
            "same_channel_before": [
                {
                    "source_message_id": f"{stem}-ctx",
                    "sender": "Dispatch",
                    "sent_at": "2026-06-01T09:59:30Z",
                    "origin_source": "quo",
                    "channel_id": channel,
                    # Comfortably past the sim's stand-in context window.
                    "text": "the shipment manifest was revised again. " * 8000,
                }
            ],
        },
    }

    with tempfile.TemporaryDirectory() as tmp:
        doc = Path(tmp) / f"{stem}.md"
        doc.write_text(f"# Manifest\n\nThe {phrase} is attached.\n")
        (Path(tmp) / f"{stem}.json").write_text(json.dumps(sidecar))
        _compose_cp_into_documents(Path(tmp) / f"{stem}.json")
        _compose_cp_into_documents(doc)

    result = await mcp_session.call_tool_json(
        "file_index_document", {"target": f"{stem}.md", "source_name": "documents"})
    assert result.get("status") == "indexed", result

    payload = await mcp_session.call_tool_json(
        "file_search", {"query": phrase, "top_k": 5})
    assert search_hits(payload, stem), payload["results"]
