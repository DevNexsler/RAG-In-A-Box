"""Loopback HTTP + real Lance retrieval through the semantic-search consumer."""

import asyncio
import json
import threading
import time

import httpx
import pytest

import mcp_server
from tests.qwen_search_support import ANSWER_ID, QUERY, corpus


@pytest.mark.parametrize("failure", [None, "partial", "truncated", "http", "timeout"])
def test_search_preserves_identity_and_honest_degradation(tmp_path, monkeypatch, http_peer, failure):
    store, embed, config = corpus(tmp_path / "index")
    config["search"]["reranker"].update(base_url=http_peer.url + "/v1", api_key="loopback-only", timeout=.5)
    monkeypatch.setattr(mcp_server, "_get_deps", lambda: (store, embed, config))
    monkeypatch.setattr(mcp_server, "_get_deep_health", lambda **kwargs: {})

    def respond(request):
        payload = request["body"]
        assert request["path"] == "/v1/chat/completions"
        passages = json.loads(payload["messages"][1]["content"])["passages"]
        scores = {p["id"]: (1.0 if "Tenant Mira" in p["text"] else 0.0) for p in reversed(passages)}
        if failure == "partial":
            scores.pop(next(iter(scores)))
        if failure == "timeout":
            time.sleep(.8)
        return (503 if failure == "http" else 200), {"choices": [{
            "finish_reason": "length" if failure == "truncated" else "stop",
            "message": {"content": json.dumps({"scores": scores})},
        }]}

    http_peer.respond = respond
    began = time.monotonic()
    result = mcp_server._file_search_impl(QUERY, top_k=25, return_mode="full", include_diagnostics=True)
    assert "error" not in result, result
    assert time.monotonic() - began < 2
    assert result["diagnostics"]["reranker_applied"] is (failure is None)
    assert result["diagnostics"]["degraded"] is (failure is not None)
    assert result["results"]
    if failure is None:
        assert result["results"][0]["doc_id"] == ANSWER_ID
        assert "Tenant Mira" in result["results"][0]["snippet"]
        assert len(http_peer.requests) == 3
        assert sum(len(json.loads(r["body"]["messages"][1]["content"])["passages"])
                   for r in http_peer.requests) == 72
    assert all(r["path"] == "/v1/chat/completions" for r in http_peer.requests)
    status = mcp_server._file_status_impl()
    assert status["health"]["reranker_responsive"] is (failure is None)


def test_mcp_dispatch_cancellation_stops_sync_worker(monkeypatch):
    from providers.rerank.qwen_bulk import QwenBulkReranker
    from core.storage import SearchHit

    started, ended = threading.Event(), threading.Event()
    async def slow_post(self, url, **kwargs):
        started.set()
        try:
            await asyncio.sleep(10)
        finally:
            ended.set()
    monkeypatch.setattr(httpx.AsyncClient, "post", slow_post)

    def search():
        return QwenBulkReranker(api_key="test-only").rerank("query", [
            SearchHit(doc_id="doc", loc="0", snippet="text", text="text", score=1),
        ])

    async def run():
        task = asyncio.create_task(mcp_server._with_tool_span(search)())
        while not started.is_set():
            await asyncio.sleep(.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        deadline = time.monotonic() + 1
        while not ended.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(.01)
        assert ended.is_set(), "cancelled MCP request left gateway work running"
    asyncio.run(run())
