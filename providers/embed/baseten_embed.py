"""Embedding provider using Baseten-hosted BEI (OpenAI-compatible /v1/embeddings).

Works with Baseten's BEI (Baseten Embeddings Inference) running TRT-LLM
compiled models for fast, dedicated embedding on cloud GPU.  No local model
swapping — embedding and enrichment run on separate instances concurrently.

Asymmetric retrieval is supported (same as Ollama provider):
  - embed_texts (indexing): raw document text, no prefix
  - embed_query (search): prepends task instruction for better retrieval
"""

from __future__ import annotations

import logging
import os
import time

import httpx

from providers.embed.base import EmbedProvider

logger = logging.getLogger(__name__)

DEFAULT_QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query: "
)

MAX_RETRIES = 2
RETRY_BACKOFF = (5.0, 15.0)


class BasetenEmbedProvider(EmbedProvider):
    """Embed via Baseten-hosted BEI /v1/embeddings endpoint.

    Features:
      - Asymmetric retrieval: query gets an instruction prefix, documents don't.
      - Auto-batch: splits large embed_texts calls into batches.
      - Retry with backoff on transient failures.
      - No local GPU needed: dedicated cloud instance.
    """

    def __init__(
        self,
        model_id: str,
        model_name: str = "model",
        api_key: str | None = None,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        batch_size: int = 64,
        timeout: float = 120.0,
    ):
        self.model_id = model_id
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("BASETEN_API_KEY", "")
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self.timeout = timeout
        self.base_url = (
            f"https://model-{model_id}.api.baseten.co"
            f"/environments/production/sync"
        )

        if not self.api_key:
            raise ValueError(
                "BASETEN_API_KEY not set. Set it in .env or pass api_key."
            )

        logger.info(
            "BasetenEmbedProvider: model_id=%s, model=%s",
            model_id, model_name,
        )

    def _call_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Call /v1/embeddings with retry and backoff."""
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
        }
        last_exc: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                resp = httpx.post(
                    f"{self.base_url}/v1/embeddings",
                    json={"model": self.model_name, "input": texts},
                    headers=headers,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                results = sorted(data["data"], key=lambda x: x["index"])
                return [r["embedding"] for r in results]

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc
                backoff = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                logger.warning(
                    "embed attempt %d/%d failed (%s: %s), retrying in %.0fs...",
                    attempt + 1, MAX_RETRIES,
                    type(exc).__name__, exc, backoff,
                )
                time.sleep(backoff)

            except httpx.HTTPStatusError as exc:
                logger.error(
                    "Baseten embedding API error: %d %s",
                    exc.response.status_code,
                    exc.response.text[:500],
                )
                raise

        raise last_exc  # type: ignore[misc]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts for indexing (no instruction prefix). Batches automatically."""
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vectors = self._call_embeddings(batch)
            all_vectors.extend(vectors)
        return all_vectors

    def embed_query(self, query: str) -> list[float]:
        """Embed a query for search (with instruction prefix for asymmetric retrieval)."""
        prefixed = f"{self.query_instruction}{query}"
        vectors = self._call_embeddings([prefixed])
        return vectors[0]
