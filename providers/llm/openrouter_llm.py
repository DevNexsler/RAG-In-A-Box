"""OpenRouter-based generative LLM for document enrichment.

Calls OpenRouter's OpenAI-compatible /v1/chat/completions endpoint.
OpenRouter routes to hundreds of models (OpenAI, MiniMax, Qwen, Llama, etc.)
with built-in load balancing, fallbacks, and usage tracking.

No local GPU needed — all inference runs on cloud providers.
"""

from __future__ import annotations

import logging
import os
import time

import httpx

from providers.llm.trace_recorder import LLMTraceRecorder

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a document metadata extractor. You analyze document text and "
    "return structured metadata as valid JSON. Never include explanations, "
    "markdown fences, or any text outside the JSON object."
)

MAX_RETRIES = 2
RETRY_BACKOFF = (5.0, 15.0)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# JSON Schema for structured output (enforced by models that support it)
_ENRICHMENT_SCHEMA = {
    "name": "enrichment",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "2-3 sentence summary of the document's purpose and key content",
            },
            "doc_type": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Document type classifications",
            },
            "entities_people": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Full names of people mentioned",
            },
            "entities_places": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Addresses, cities, locations",
            },
            "entities_orgs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Company and organization names",
            },
            "entities_dates": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Dates mentioned in YYYY-MM-DD format",
            },
            "topics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "5-10 high-level topics",
            },
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "10-20 specific terms and phrases",
            },
            "key_facts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Most important facts, conclusions, or action items",
            },
            "suggested_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Classification tags for this document, preferring taxonomy entries when available",
            },
            "suggested_folder": {
                "type": "string",
                "description": "Best folder path for filing this document from the taxonomy, or empty string",
            },
        },
        "required": [
            "summary", "doc_type", "entities_people", "entities_places",
            "entities_orgs", "entities_dates", "topics", "keywords", "key_facts",
            "suggested_tags", "suggested_folder",
        ],
        "additionalProperties": False,
    },
}


class OpenRouterGenerator:
    """Text generation via OpenRouter for document enrichment.

    Interface: generate(prompt, max_tokens) -> str (raw JSON string).
    """

    def __init__(
        self,
        model: str = "openai/gpt-4.1-mini",
        api_key: str | None = None,
        timeout: float = 300.0,
        trace_capture: dict | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.timeout = timeout
        trace_capture = trace_capture or {}
        self.trace_recorder = LLMTraceRecorder(
            provider="openrouter",
            model=model,
            enabled=bool(trace_capture.get("enabled", False)),
            directory=trace_capture.get("directory", ".evals/llm-traces"),
        )

        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set. Set it in .env or pass api_key."
            )

        logger.info(
            "OpenRouterGenerator initialized: model=%s", model,
        )

    def _build_response_format(self) -> dict:
        """Build the response_format payload based on model capabilities."""
        # Models supporting structured output (json_schema)
        if self.model.startswith(("openai/", "google/")):
            return {
                "type": "json_schema",
                "json_schema": _ENRICHMENT_SCHEMA,
            }
        # Fallback: basic JSON mode for other models
        return {"type": "json_object"}

    def generate(self, user_prompt: str, max_tokens: int = 512) -> str:
        """Generate structured JSON from a user prompt.

        Uses OpenAI-compatible chat completions with structured JSON output.
        Returns the raw response string (parsed by doc_enrichment).
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "response_format": self._build_response_format(),
        }
        # MiniMax models need reasoning exclusion to preserve token budget
        if "minimax" in self.model:
            payload["reasoning"] = {"exclude": True}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        trace_request = {
            "url": f"{OPENROUTER_BASE_URL}/chat/completions",
            "timeout": self.timeout,
            "payload": payload,
        }

        last_exc: Exception | None = None
        started = time.perf_counter()

        for attempt in range(MAX_RETRIES):
            try:
                resp = httpx.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                self.trace_recorder.record(
                    request=trace_request,
                    response=data,
                    success=True,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                )
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
                    "OpenRouter API error: %d %s",
                    exc.response.status_code,
                    exc.response.text[:500],
                )
                self.trace_recorder.record(
                    request=trace_request,
                    success=False,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    error={
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "status_code": exc.response.status_code,
                        "body": exc.response.text,
                    },
                )
                raise

        self.trace_recorder.record(
            request=trace_request,
            success=False,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error={
                "type": type(last_exc).__name__ if last_exc else "UnknownError",
                "message": str(last_exc) if last_exc else "Unknown OpenRouter error",
            },
        )
        raise last_exc  # type: ignore[misc]
