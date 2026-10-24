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
from functools import lru_cache

import httpx

from core.resilience import (
    RateLimitError,
    TransientError,
    call_with_retry,
    raise_for_status,
)
from providers.embed.base import EmbedProvider
from providers.embed.limits import bound_inputs, resolve_max_input_tokens

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

# DeepInfra — the upstream OpenRouter routes qwen3-embedding to — caps a single
# input at 131072 characters *on top of* the model's 40960-token window, and
# rejects anything longer with a permanent 422 (#1655):
#     Value error, The input sequence should have less than 131072 characters.
# The two limits are not interchangeable: ordinary English prose tokenizes at
# ~4.9 characters per token, so an input bounded only to the token window is
# still ~180k characters and fails every time. Measured against the live
# endpoint, 131072 characters is accepted and 131073 is not.
DEFAULT_MAX_INPUT_CHARS = 131_072

# The token window, by contrast, OpenRouter publishes per upstream, and the
# route may land a request on any of them — so the window that holds is the
# smallest one listed, not the model's own. qwen3-embedding is 40960 tokens
# natively, but SiliconFlow serves it with 32768 and rejects anything longer
# with a permanent 400 (`{"code":20015,"message":"The parameter is invalid."}`)
# that no retry clears (#4415). Measured 2026-10-24 with the route pinned to
# SiliconFlow: 32767 tokens (32766 + end-of-sequence) accepted, 32769 rejected.
ROUTE_LOOKUP_TIMEOUT = 10.0


@lru_cache(maxsize=None)
def _fetch_route_input_tokens(base_url: str, model: str) -> int:
    resp = httpx.get(f"{base_url}/models/{model}/endpoints", timeout=ROUTE_LOOKUP_TIMEOUT)
    resp.raise_for_status()
    windows = [
        endpoint.get("context_length")
        for endpoint in resp.json()["data"]["endpoints"]
    ]
    return min(w for w in windows if isinstance(w, int) and w > 0)


def _route_input_tokens(base_url: str, model: str) -> int | None:
    """Smallest input window among the upstreams the route may pick for `model`,
    or None when the route does not publish one (a non-OpenRouter base_url, or
    the listing is unreachable). Only a successful lookup is cached, so a blip
    at startup is retried by the next provider instead of pinning the fallback."""
    try:
        return _fetch_route_input_tokens(base_url, model)
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        logger.warning(
            "OpenRouterEmbedProvider: no per-upstream window listed for %s at %s "
            "(%s) — bounding to the model's own window",
            model, base_url, exc,
        )
        return None


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
        max_input_tokens: int | None = None,
        max_input_chars: int | None = None,
    ):
        self.model = model
        self.model_name = model  # alias for SemanticEmbeddingAdapter compatibility
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self.timeout = timeout
        self._base_url = (base_url or OPENROUTER_BASE_URL).rstrip("/")
        self.max_input_tokens = (
            max_input_tokens
            or _route_input_tokens(self._base_url, model)
            or resolve_max_input_tokens(model)
        )
        # Explicit None check, not `or`: 0 is the module's "no bound" value and
        # has to stay reachable as an escape hatch if the route's cap changes.
        self.max_input_chars = (
            DEFAULT_MAX_INPUT_CHARS if max_input_chars is None else max_input_chars
        )

        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set. Set it in .env or pass api_key."
            )

        logger.info(
            "OpenRouterEmbedProvider: model=%s max_input_tokens=%d",
            model, self.max_input_tokens,
        )

    def _call_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Call /v1/embeddings via the shared resilience layer (retries transient
        5xx/429/timeouts; permanent 4xx raise straight through, carrying the
        upstream reason from the response body)."""
        texts = bound_inputs(
            texts,
            self.max_input_tokens,
            max_input_chars=self.max_input_chars,
            model=self.model,
            label=f"openrouter-embed[{self.model}]",
        )
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
            # 5xx/429 -> HTTPStatusError -> retried by the layer; a permanent 4xx
            # raises straight through carrying OpenRouter's reason (#1657).
            raise_for_status(resp)
            data = resp.json()
            if "data" not in data:
                # OpenRouter wraps upstream provider failures (e.g. Nebius 429 quota)
                # in an HTTP 200 with an {"error": ...} body — surface as a transient
                # so the shared layer retries it (or a permanent error if it's 4xx).
                err = data.get("error") or {}
                code = err.get("code")
                message = str(err.get("message", data))[:300]
                if code == 429:
                    raise RateLimitError(
                        f"OpenRouter upstream {code} in 200 body: {message}"
                    )
                if isinstance(code, int) and 500 <= code < 600:
                    raise TransientError(f"OpenRouter upstream {code} in 200 body: {message}")
                raise RuntimeError(f"OpenRouter embeddings error {code}: {message}")
            results = sorted(data["data"], key=lambda x: x["index"])
            return [r["embedding"] for r in results]

        return call_with_retry(
            _do, attempts=MAX_RETRIES, backoff=RETRY_BACKOFF, label="openrouter-embed",
            circuit_key=self._base_url,
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
