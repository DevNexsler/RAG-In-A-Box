"""Webhook delivery + provider-fault recovery, driven from outside.

Fault notes:
  - provider-sim faults auto-exhaust (times counts down and the armed entry is
    removed); the autouse sim_reset fixture ALSO resets before and after every
    test, so no fault can leak across tests even on assertion failure.
  - hook-delivery evidence for the initial sweep is the snapshot captured by
    indexed_corpus itself (the per-test /admin/reset wipes the live sink, so
    asserting on a later GET /hooks/received would race with reset ordering).
"""
import uuid

import httpx
import pytest

from tests.e2e.client import get_hook_events, search_hits
from tests.e2e.conftest import E2E_SIM_URL

pytestmark = pytest.mark.anyio


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
