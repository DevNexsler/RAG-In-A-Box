"""Opt-in real gateway + MCP HTTP + isolated Lance semantic-search E2E."""

import asyncio
import json
import os
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

import mcp_server
from tests.e2e.client import tool_payload
from tests.qwen_search_support import ANSWER_ID, QUERY, corpus

pytestmark = [pytest.mark.live, pytest.mark.skipif(os.getenv("RUN_QWEN_RERANK_LIVE") != "1",
                                                reason="explicit real gateway opt-in required")]


@pytest.mark.parametrize("count", [3, 70, 80, 90])
def test_real_gateway_mcp_semantic_search(tmp_path, monkeypatch, count):
    assert os.getenv("LITELLM_API_KEY") or os.getenv("LITELLM_MASTER_KEY"), "gateway credential required"
    assert os.getenv("LITELLM_BASE_URL"), "explicit test gateway URL required"
    from scripts.live_preflight import production_storage
    assert production_storage().path_conflict(tmp_path) is None, "live corpus must use isolated storage"
    store, embed, config = corpus(tmp_path / "index", count=count, dense=count > 3)
    monkeypatch.setattr(mcp_server, "_get_deps", lambda: (store, embed, config))
    wire = []
    original_post = httpx.AsyncClient.post

    async def measured_post(client, url, **kwargs):
        if str(url) == os.environ["LITELLM_BASE_URL"].rstrip("/") + "/chat/completions":
            payload = kwargs["json"]
            assert payload["priority"] == 0 and payload["model"] == "qwen-bulk"
            assert payload["chat_template_kwargs"] == {"enable_thinking": False}
            assert payload["num_retries"] == 0 and payload["disable_fallbacks"] is True
            passages = json.loads(payload["messages"][1]["content"])["passages"]
            wire.append({"passages": len(passages), "remaining_s": round(
                payload["metadata"]["inference_deadline_seconds"], 3)})
        return await original_post(client, url, **kwargs)
    monkeypatch.setattr(httpx.AsyncClient, "post", measured_post)
    monkeypatch.setattr(mcp_server.mcp, "_session_manager", None)
    app = mcp_server.mcp.streamable_http_app()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]})
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(.01)
        assert server.started, "isolated MCP server failed to start"

        async def run():
            async with streamablehttp_client(f"http://127.0.0.1:{port}/mcp") as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    began = time.monotonic()
                    result = tool_payload(await session.call_tool("file_search", {
                        "query": QUERY, "top_k": 25, "return": "full", "include_diagnostics": True,
                    }))
                    elapsed = time.monotonic() - began
                    assert "error" not in result, result
                    diagnostics = result["diagnostics"]
                    print(json.dumps({"candidates": count, "elapsed_s": round(elapsed, 3),
                                      "timing_ms": diagnostics.get("timing_ms"),
                                      "gateway_batches": wire,
                                      "reranker_applied": diagnostics["reranker_applied"],
                                      "degraded": diagnostics["degraded"]}))
                    assert diagnostics["reranker_applied"] and not diagnostics["degraded"], result
                    assert result["results"][0]["doc_id"] == ANSWER_ID, result["results"][:3]
                    assert "Tenant Mira" in result["results"][0]["snippet"]
                    assert sum(batch["passages"] for batch in wire) == count
                    assert elapsed < 12, f"MCP search exceeded bounded local rerank budget: {elapsed:.3f}s"
        asyncio.run(run())
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
        assert not thread.is_alive(), "isolated MCP server leaked"
