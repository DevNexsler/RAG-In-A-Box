"""Service health checks: fast connectivity tests for all active endpoints.

Each test class is independently skippable when its service is unavailable.
These tests verify "is the service up and reachable?" — not inference quality.

Run with:  pytest tests/test_service_health.py -v
"""

import os

import pytest
import httpx

from scripts.live_preflight import check_litellm_ocr

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# -----------------------------------------------------------------------
# OpenRouter (enrichment + embedding)
# -----------------------------------------------------------------------

_has_openrouter_key = bool(os.environ.get("OPENROUTER_API_KEY"))


@pytest.mark.live
@pytest.mark.skipif(not _has_openrouter_key, reason="OPENROUTER_API_KEY not set")
class TestOpenRouterHealth:
    """Verify OpenRouter API is reachable and accepts our key."""

    def _headers(self):
        return {
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        }

    def test_chat_completions_reachable(self):
        """Minimal chat completion returns 200."""
        resp = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "model": "minimax/minimax-m2.5",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            },
            headers=self._headers(),
            timeout=30.0,
        )
        assert resp.status_code == 200, f"OpenRouter chat: {resp.status_code} {resp.text[:200]}"

    def test_embeddings_reachable(self):
        """Minimal embedding returns 200 with vector data."""
        resp = httpx.post(
            "https://openrouter.ai/api/v1/embeddings",
            json={
                "model": "qwen/qwen3-embedding-8b",
                "input": ["health check"],
            },
            headers=self._headers(),
            timeout=30.0,
        )
        assert resp.status_code == 200, f"OpenRouter embed: {resp.status_code} {resp.text[:200]}"
        data = resp.json()
        assert len(data["data"]) == 1
        assert len(data["data"][0]["embedding"]) > 0


# -----------------------------------------------------------------------
# DeepInfra (reranker)
# -----------------------------------------------------------------------

_has_deepinfra_key = bool(os.environ.get("DEEPINFRA_API_KEY"))


@pytest.mark.live
@pytest.mark.skipif(not _has_deepinfra_key, reason="DEEPINFRA_API_KEY not set")
class TestDeepInfraRerankerHealth:
    """Verify DeepInfra reranker endpoint is reachable."""

    def test_rerank_reachable(self):
        """Minimal inference endpoint returns 200."""
        resp = httpx.post(
            "https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Reranker-8B",
            headers={
                "Authorization": f"Bearer {os.environ['DEEPINFRA_API_KEY']}",
                "Content-Type": "application/json",
            },
            json={
                "queries": ["test"],
                "documents": ["test document"],
            },
            timeout=15.0,
        )
        assert resp.status_code == 200, f"DeepInfra reranker: {resp.status_code} {resp.text[:200]}"


# -----------------------------------------------------------------------
# LiteLLM OCR/vision routing authority
# -----------------------------------------------------------------------

@pytest.mark.live
class TestLiteLLMOCRHealth:
    """Verify configured LiteLLM OCR and vision aliases are available."""

    def test_preflight_accepts_configured_aliases(self):
        ok, reason = check_litellm_ocr()
        assert ok, reason


# -----------------------------------------------------------------------
# LanceDB (local file DB)
# -----------------------------------------------------------------------

@pytest.mark.live
class TestLanceDBHealth:
    """Verify LanceDB index is accessible."""

    def test_store_opens(self):
        """LanceDBStore opens and lists doc_ids without error."""
        from core.config import load_config
        from lancedb_store import LanceDBStore

        config = load_config()
        store = LanceDBStore(
            config["index_root"],
            config.get("lancedb", {}).get("table", "chunks"),
        )
        doc_ids = store.list_doc_ids()
        assert isinstance(doc_ids, list)

    def test_fts_available(self):
        """FTS/tantivy index is operational."""
        from core.config import load_config
        from lancedb_store import LanceDBStore

        config = load_config()
        store = LanceDBStore(
            config["index_root"],
            config.get("lancedb", {}).get("table", "chunks"),
        )
        if not store.fts_available():
            pytest.skip("FTS index not available — run file_index_update to rebuild")
