"""Real reranker HTTP failures remain visible through the production health route."""

import httpx
import pytest
import uvicorn
import yaml

import mcp_server
from core.resilience import CIRCUITS
from core.storage import SearchHit
from search_hybrid import DeepInfraReranker


@pytest.mark.parametrize("failed_probe", [
    (503, {"error": "unavailable"}), (200, b"not-json"),
    (200, {"scores": []}), (200, {"scores": ["invalid"]}),
    (200, {"scores": [True]}), (200, {"scores": [float("nan")]}),
    (200, {"scores": [float("inf")]}), (200, {"scores": [0.8, 0.3]}), (200, []),
])
def test_http_health_requires_valid_reranker_recovery(tmp_path, monkeypatch, http_peer, failed_probe):
    documents = tmp_path / "documents"
    documents.mkdir()
    config = {"documents_root": str(documents), "index_root": str(tmp_path / "index"),
              "embeddings": {"provider": "openrouter"}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("INDEX_ROOT", config["index_root"])
    monkeypatch.setenv("DOCUMENTS_ROOT", str(documents))
    monkeypatch.setenv("API_KEY", "contract-only")
    now = [0.0]
    monkeypatch.setattr(CIRCUITS, "_clock", lambda: now[0])

    def rerank():
        return DeepInfraReranker(api_key="contract-only", base_url=http_peer.url).rerank(
            "invoice", [SearchHit("doc", "c:0", "Invoice 42", "Invoice 42", 1.0)],
        )

    async def exercise_routes(server):
        # Replace only the external ASGI serving boundary; use the exact app
        # assembled by run_server, including routes and authentication.
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.config.app),
                                     base_url="http://app") as client:
            assert (await client.get("/health/providers")).status_code == 200
            http_peer.respond = lambda request: (402, {"error": "account limit"})
            with pytest.raises(RuntimeError):
                rerank()
            refused = await client.get("/health/providers")
            assert refused.status_code == 503
            assert refused.json()["provider_failures"]["total_count"] == 1
            with pytest.raises(RuntimeError):
                rerank()  # a fresh client still shares the endpoint circuit
            assert len(http_peer.requests) == 1

            now[0] = 301.0
            http_peer.respond = lambda request: failed_probe
            with pytest.raises(RuntimeError):
                rerank()
            assert (await client.get("/health/providers")).status_code == 503
            assert len(http_peer.requests) == 2

            now[0] = 362.0
            http_peer.respond = lambda request: (200, {"scores": [0.8]})
            assert rerank()[0].doc_id == "doc"
            assert (await client.get("/health/providers")).status_code == 200
            assert len(http_peer.requests) == 3

    monkeypatch.setattr(uvicorn.Server, "serve", exercise_routes)
    mcp_server.run_server(transport="streamable-http")
