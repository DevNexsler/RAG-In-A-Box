"""Real image/MCP regression for the default scorer and its degraded path."""

import time

import httpx
import pytest

from tests.e2e.client import E2E_SIM_URL
from tests.e2e.conftest import NOTE_PHRASE

pytestmark = pytest.mark.anyio

# Longer than a live search's 45 s rerank budget (providers/rerank/qwen_bulk.py).
STALL_SECONDS = 47


@pytest.mark.parametrize("fault", [None, "429", "garbage", "reasoning_only", "timeout"])
async def test_qwen_search_failure_and_recovery(indexed_corpus, mcp_session, fault):
    if fault:
        async with httpx.AsyncClient() as sim:
            response = await sim.post(f"{E2E_SIM_URL}/admin/fault", json={
                "route_prefix": "/v1/chat/completions", "fault": fault, "times": 1,
                "seconds": STALL_SECONDS,
            })
            assert response.status_code == 200
    began = time.monotonic()
    result = await mcp_session.call_tool_json("file_search", {
        "query": NOTE_PHRASE, "top_k": 5, "include_diagnostics": True,
    })
    assert time.monotonic() - began < 50
    assert not result.get("error"), result
    assert result["results"]
    assert result["diagnostics"]["reranker_applied"] is (fault is None)
    assert result["diagnostics"]["degraded"] is (fault is not None)
    status = await mcp_session.call_tool_json("file_status", {})
    assert status["health"]["reranker_responsive"] is True
    recovered = await mcp_session.call_tool_json("file_search", {
        "query": NOTE_PHRASE, "top_k": 5, "include_diagnostics": True,
    })
    assert recovered["diagnostics"]["reranker_applied"] is True
    assert recovered["diagnostics"]["degraded"] is False


async def test_background_search_waits_out_a_slow_reranker(indexed_corpus, mcp_session):
    # A scheduled caller refuses degraded scores (the comms review failed for
    # days on 30 s cosine fallbacks). The same stall that degrades a live
    # search above must still rerank a background one.
    async with httpx.AsyncClient() as sim:
        response = await sim.post(f"{E2E_SIM_URL}/admin/fault", json={
            "route_prefix": "/v1/chat/completions", "fault": "timeout", "times": 1,
            "seconds": STALL_SECONDS,
        })
        assert response.status_code == 200
    began = time.monotonic()
    result = await mcp_session.call_tool_json("file_search", {
        "query": NOTE_PHRASE, "top_k": 5, "include_diagnostics": True, "background": True,
    })
    assert STALL_SECONDS <= time.monotonic() - began < 60
    assert not result.get("error"), result
    assert result["results"]
    assert result["diagnostics"]["reranker_applied"] is True
    assert result["diagnostics"]["degraded"] is False
