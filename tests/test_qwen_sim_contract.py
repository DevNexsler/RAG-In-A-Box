"""Default score-map protocol against the hermetic gateway simulator."""

import asyncio
import json

import httpx

from providers.rerank.qwen_bulk import QwenBulkReranker
from tests.test_provider_sim import _load_module


def test_sim_returns_shuffled_id_map_without_changing_enrichment():
    async def run():
        app = _load_module().app
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://sim") as client:
            payload = QwenBulkReranker(api_key="sim")._payload("paid invoice", [
                "paid invoice", "promised payment", "unrelated note",
            ], ["D64", "D65", "D66"], 10000)
            response = await client.post("/v1/chat/completions", json=payload)
            content = json.loads(response.json()["choices"][0]["message"]["content"])
            assert list(content["scores"]) == ["D66", "D65", "D64"]
            assert content["scores"]["D64"] == 1 and content["scores"]["D66"] == 0
            response = await client.post("/api/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "A normal document"}],
                "response_format": {"type": "json_object"},
            })
            assert "summary" in json.loads(response.json()["choices"][0]["message"]["content"])
    asyncio.run(run())
