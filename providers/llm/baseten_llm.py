"""Baseten-hosted vLLM for document enrichment.

Calls a Baseten-deployed vLLM model via the OpenAI-compatible
/v1/chat/completions endpoint.  The model runs on a dedicated cloud GPU,
eliminating local model swapping and the ghost-request problem that
plagued Ollama (vLLM aborts on disconnect, Ollama does not).

Scale-to-zero may add ~30s cold start on the first request after idle;
the generous default timeout (300s) absorbs this.
"""

from __future__ import annotations

import logging
import os
import time

import httpx

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a document metadata extractor. You analyze document text and "
    "return structured metadata as valid JSON. Never include explanations, "
    "markdown fences, or any text outside the JSON object."
)

MAX_RETRIES = 2
RETRY_BACKOFF = (5.0, 15.0)


class BasetenGenerator:
    """Text generation via Baseten-hosted vLLM for document enrichment.

    Interface: generate(prompt, max_tokens) -> str (raw JSON string).
    """

    def __init__(
        self,
        model_id: str,
        model_name: str = "qwen3-14b",
        api_key: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.model_id = model_id
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("BASETEN_API_KEY", "")
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
            "BasetenGenerator initialized: model_id=%s, model=%s",
            model_id, model_name,
        )

    def generate(self, user_prompt: str, max_tokens: int = 512) -> str:
        """Generate structured JSON from a user prompt.

        Uses OpenAI-compatible chat completions with JSON output mode.
        Retries on transient network failures with backoff.
        Returns the raw response string (parsed by doc_enrichment).
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
        }

        last_exc: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                resp = httpx.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return content.strip()

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc
                backoff = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                logger.warning(
                    "generate() attempt %d/%d failed (%s: %s), retrying in %.0fs...",
                    attempt + 1, MAX_RETRIES,
                    type(exc).__name__, exc, backoff,
                )
                time.sleep(backoff)

            except httpx.HTTPStatusError as exc:
                logger.error(
                    "Baseten enrichment API error: %d %s",
                    exc.response.status_code,
                    exc.response.text[:500],
                )
                raise

        raise last_exc  # type: ignore[misc]
