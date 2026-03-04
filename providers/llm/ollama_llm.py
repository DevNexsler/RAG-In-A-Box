"""Ollama-based generative LLM for document enrichment.

Calls Ollama's native /api/chat endpoint with structured JSON output
(format parameter) to guarantee valid responses.  No in-process model
loading — Ollama handles lifecycle, queuing, and memory management.

Thinking mode is disabled (think=false) because structured metadata
extraction doesn't benefit from chain-of-thought reasoning, and Qwen3's
internal thinking tokens would dominate generation time (~100s vs ~4s).

Cold-start handling: large models (e.g. 14B) can take 30-60s to load
into GPU memory.  The first generate() call triggers ensure_loaded(),
which sends a minimal warmup request with a generous timeout.  Subsequent
calls use the normal timeout.  Transient failures are retried with backoff.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a document metadata extractor. You analyze document text and "
    "return structured metadata as valid JSON. Never include explanations, "
    "markdown fences, or any text outside the JSON object."
)

_ENRICHMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "doc_type": {"type": "array", "items": {"type": "string"}},
        "entities_people": {"type": "array", "items": {"type": "string"}},
        "entities_places": {"type": "array", "items": {"type": "string"}},
        "entities_orgs": {"type": "array", "items": {"type": "string"}},
        "entities_dates": {"type": "array", "items": {"type": "string"}},
        "topics": {"type": "array", "items": {"type": "string"}},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "key_facts": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "summary", "doc_type", "entities_people", "entities_places",
        "entities_orgs", "entities_dates", "topics", "keywords", "key_facts",
    ],
}

WARMUP_TIMEOUT = 300.0  # 5 min for cold-start model loading
MAX_RETRIES = 3
RETRY_BACKOFF = (5.0, 15.0, 30.0)


class OllamaGenerator:
    """Text generation via Ollama for document enrichment.

    Uses Ollama's native /api/chat with structured output (format parameter)
    to guarantee valid JSON responses matching the enrichment schema.
    Interface: generate(prompt, max_tokens) -> str.

    On first call, sends a lightweight warmup request to ensure the model
    is loaded into memory before real enrichment begins.
    """

    def __init__(
        self,
        model_name: str = "qwen3:14b-udq6",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._model_loaded = False
        logger.info("OllamaGenerator initialized: %s (model=%s)", self.base_url, model_name)

    def _check_ollama(self) -> None:
        """Verify Ollama is reachable."""
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
        """Force-load the model into memory if not already loaded.

        Sends a minimal prompt with a generous timeout to absorb the
        cold-start delay.  Ollama keeps models in memory after the first
        request, so subsequent calls will be fast.
        """
        if self._model_loaded:
            return

        import httpx

        logger.info(
            "Warming up %s (this may take 30-60s on first load)...",
            self.model_name,
        )
        start = time.monotonic()
        try:
            resp = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "think": False,
                    "stream": False,
                    "options": {"num_predict": 1},
                },
                timeout=WARMUP_TIMEOUT,
            )
            elapsed = time.monotonic() - start
            if resp.status_code == 200:
                self._model_loaded = True
                logger.info(
                    "Model %s loaded and ready (%.1fs)",
                    self.model_name, elapsed,
                )
            else:
                logger.warning(
                    "Warmup got status %d after %.1fs — will retry on first generate()",
                    resp.status_code, elapsed,
                )
        except httpx.TimeoutException:
            elapsed = time.monotonic() - start
            logger.warning(
                "Warmup timed out after %.1fs — model may still be loading, "
                "will retry with extended timeout on generate()",
                elapsed,
            )
        except Exception as exc:
            self._check_ollama()
            logger.warning("Warmup failed: %s", exc)

    def generate(self, user_prompt: str, max_tokens: int = 512) -> str:
        """Generate structured JSON from a user prompt.

        Uses Ollama's native /api/chat with the format parameter to enforce
        the enrichment JSON schema.  Thinking is disabled for speed.
        Retries with backoff on transient failures (timeout, connection).
        Returns the raw JSON string.
        """
        import httpx

        self.ensure_loaded()

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "format": _ENRICHMENT_SCHEMA,
            "think": False,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
            },
        }

        use_timeout = WARMUP_TIMEOUT if not self._model_loaded else self.timeout
        last_exc: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                resp = httpx.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=use_timeout,
                )
                if resp.status_code != 200:
                    self._check_ollama()
                    resp.raise_for_status()

                self._model_loaded = True
                data = resp.json()
                content = data.get("message", {}).get("content", "")
                return content.strip()

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc
                self._model_loaded = False  # Model may have been unloaded; reset for warmup timeout
                backoff = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                logger.warning(
                    "generate() attempt %d/%d failed (%s: %s), retrying in %.0fs...",
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
