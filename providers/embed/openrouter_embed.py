"""Embedding provider using OpenRouter's OpenAI-compatible /v1/embeddings endpoint.

Routes to hosted embedding models (Qwen3-Embedding, etc.) via OpenRouter.
No local GPU needed — all inference runs on cloud providers.

Asymmetric retrieval is supported:
  - embed_texts (indexing): raw document text, no prefix
  - embed_query (search): prepends task instruction for better retrieval
"""

from __future__ import annotations

import logging
import os

import httpx

from core.resilience import TransientError, call_with_retry
from providers.embed.base import EmbedProvider

logger = logging.getLogger(__name__)

DEFAULT_QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query: "
)

# Nebius (the qwen3-embedding upstream) throttles aggressively under
# sustained indexing load; 2 attempts exhausted on 71 docs in one run.
MAX_RETRIES = 5
RETRY_BACKOFF = (2.0, 5.0, 15.0, 30.0, 60.0)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterEmbedProvider(EmbedProvider):
    """Embed via OpenRouter's /v1/embeddings endpoint.

    Features:
      - Asymmetric retrieval: query gets an instruction prefix, documents don't.
      - Auto-batch: splits large embed_texts calls into batches.
      - Retry with backoff on transient failures.
    """

    def __init__(
        self,
        model: str = "qwen/qwen3-embedding-8b",
        api_key: str | None = None,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        batch_size: int = 64,
        timeout: float = 120.0,
        base_url: str | None = None,
    ):
        self.model = model
        self.model_name = model  # alias for SemanticEmbeddingAdapter compatibility
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self.timeout = timeout
        self._base_url = (base_url or OPENROUTER_BASE_URL).rstrip("/")

        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set. Set it in .env or pass api_key."
            )

        logger.info(
            "OpenRouterEmbedProvider: model=%s", model,
        )

    def _call_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Call /v1/embeddings via the shared resilience layer (retries transient
        5xx/429/timeouts; permanent 4xx raise straight through)."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        def _do() -> list[list[float]]:
            resp = httpx.post(
                f"{self._base_url}/embeddings",
                json={"model": self.model, "input": texts},
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()  # 5xx/429 -> HTTPStatusError -> retried by the layer
            data = resp.json()
            if "data" not in data:
                # OpenRouter wraps upstream provider failures (e.g. Nebius 429 quota)
                # in an HTTP 200 with an {"error": ...} body — surface as a transient
                # so the shared layer retries it (or a permanent error if it's 4xx).
                err = data.get("error") or {}
                code = err.get("code")
                message = str(err.get("message", data))[:300]
                if code == 429 or (isinstance(code, int) and 500 <= code < 600):
                    raise TransientError(f"OpenRouter upstream {code} in 200 body: {message}")
                raise RuntimeError(f"OpenRouter embeddings error {code}: {message}")
            results = sorted(data["data"], key=lambda x: x["index"])
            return [r["embedding"] for r in results]

        return call_with_retry(
            _do, attempts=MAX_RETRIES, backoff=RETRY_BACKOFF, label="openrouter-embed",
        )

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
