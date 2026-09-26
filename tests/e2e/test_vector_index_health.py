"""Unauthenticated ANN metadata from the running candidate's persisted index."""

import httpx
import pytest

from tests.e2e.client import E2E_BASE

pytestmark = pytest.mark.anyio


async def test_health_exposes_live_vector_index(indexed_corpus):
    async with httpx.AsyncClient(base_url=E2E_BASE, timeout=5) as client:
        response = await client.get("/health")
    assert response.status_code == 200, response.text
    stats = response.json()["vector_index"]
    assert stats["available"] is True
    # The tiny staging corpus cannot train PQ; ensure_vector_index builds FLAT.
    assert stats["index_type"] == "IVF_FLAT"
    assert stats["indexed_rows"] > 0
    assert stats["num_indices"] >= 1
    assert stats["unindexed_rows"] >= 0
