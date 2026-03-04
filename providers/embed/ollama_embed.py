"""Embedding provider using Ollama's OpenAI-compatible /v1/embeddings endpoint.

Works with any model served by Ollama (e.g. qwen3-embedding:4b-q8_0).
Ollama handles its own lifecycle — no server management needed here.

For Qwen3-Embedding, asymmetric retrieval is supported:
  - embed_texts (indexing): raw document text, no prefix
  - embed_query (search): prepends task instruction for better retrieval

Cold-start handling: on first call, ensure_loaded() sends a single-token
embedding request with a generous timeout to absorb model loading time.
Transient timeouts are retried with backoff.
"""

from __future__ import annotations

import logging
import time

from providers.embed.base import EmbedProvider

logger = logging.getLogger(__name__)

DEFAULT_QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query: "
)

WARMUP_TIMEOUT = 300.0  # 5 min for cold-start model loading
MAX_RETRIES = 3
RETRY_BACKOFF = (5.0, 15.0, 30.0)


class OllamaEmbedProvider(EmbedProvider):
    """Embed via Ollama's OpenAI-compatible /v1/embeddings endpoint.

    Features:
      - Asymmetric retrieval: query gets an instruction prefix, documents don't.
      - Auto-batch: splits large embed_texts calls into batches.
      - Cold-start warmup: first call triggers model load with generous timeout.
      - Retry with backoff on transient failures.
      - No server management: Ollama daemon must be running externally.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "qwen3-embedding:4b-q8_0",
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        batch_size: int = 64,
        timeout: float = 300.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self.timeout = timeout
        self._model_loaded = False

        logger.info(
            "OllamaEmbedProvider: %s/v1/embeddings (model=%s)",
            self.base_url, model_name,
        )

    def _check_ollama(self) -> None:
        """Verify Ollama is reachable. Raises RuntimeError with setup instructions."""
        import httpx

        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}. "
                "Make sure Ollama is running (ollama serve) and the model is pulled "
                f"(ollama pull {self.model_name}). Error: {exc}"
            ) from exc

    def ensure_loaded(self) -> None:
        """Force-load the embedding model into memory if not already loaded."""
        if self._model_loaded:
            return

        import httpx

        logger.info(
            "Warming up embedding model %s (may take 10-30s on first load)...",
            self.model_name,
        )
        start = time.monotonic()
        try:
            resp = httpx.post(
                f"{self.base_url}/v1/embeddings",
                json={"model": self.model_name, "input": ["warmup"]},
                timeout=WARMUP_TIMEOUT,
            )
            elapsed = time.monotonic() - start
            if resp.status_code == 200:
                self._model_loaded = True
                logger.info(
                    "Embedding model %s loaded and ready (%.1fs)",
                    self.model_name, elapsed,
                )
            else:
                logger.warning(
                    "Embedding warmup got status %d after %.1fs",
                    resp.status_code, elapsed,
                )
        except httpx.TimeoutException:
            logger.warning(
                "Embedding warmup timed out after %.1fs — will retry with extended timeout",
                time.monotonic() - start,
            )
        except Exception as exc:
            self._check_ollama()
            logger.warning("Embedding warmup failed: %s", exc)

    def _call_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Call /v1/embeddings with retry and backoff for cold-start resilience."""
        import httpx

        use_timeout = WARMUP_TIMEOUT if not self._model_loaded else self.timeout
        last_exc: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                resp = httpx.post(
                    f"{self.base_url}/v1/embeddings",
                    json={"model": self.model_name, "input": texts},
                    timeout=use_timeout,
                )
                if resp.status_code != 200:
                    self._check_ollama()
                    resp.raise_for_status()

                self._model_loaded = True
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
                use_timeout = WARMUP_TIMEOUT
            except Exception:
                self._check_ollama()
                raise

        self._check_ollama()
        raise last_exc  # type: ignore[misc]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts for indexing (no instruction prefix). Batches automatically."""
        if not texts:
            return []

        self.ensure_loaded()

        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vectors = self._call_embeddings(batch)
            all_vectors.extend(vectors)
        return all_vectors

    def embed_query(self, query: str) -> list[float]:
        """Embed a query for search (with instruction prefix for asymmetric retrieval)."""
        self.ensure_loaded()
        prefixed = f"{self.query_instruction}{query}"
        vectors = self._call_embeddings([prefixed])
        return vectors[0]
