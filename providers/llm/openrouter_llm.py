"""OpenRouter-based generative LLM for document enrichment.

Calls OpenRouter's OpenAI-compatible /v1/chat/completions endpoint.
OpenRouter routes to hundreds of models (OpenAI, MiniMax, Qwen, Llama, etc.)
with built-in load balancing, fallbacks, and usage tracking.

No local GPU needed — all inference runs on cloud providers.
"""

from __future__ import annotations

import copy
import logging
import os
import time
from typing import Any, TypedDict

import httpx

from doc_enrichment import enrichment_response_schema
from providers.llm.trace_recorder import LLMTraceRecorder

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a document metadata extractor. You analyze document text and "
    "return structured metadata as valid JSON. Never include explanations, "
    "markdown fences, or any text outside the JSON object."
)

MAX_RETRIES = 2
RETRY_BACKOFF = (5.0, 15.0)
CONNECT_TIMEOUT_CAP = 10.0

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# JSON Schema for structured output (enforced by models that support it)
_ENRICHMENT_SCHEMA = {
    "name": "enrichment",
    "strict": True,
    "schema": enrichment_response_schema(),
}


def _is_response_format_rejection(response: httpx.Response) -> bool:
    body = response.text.lower()
    return any(
        phrase in body
        for phrase in (
            "response_format",
            "json_schema",
            "structured output",
            "schema",
        )
    )


class OpenRouterReplayMetadata(TypedDict):
    """Replay payload for benchmark re-runs."""

    content: str
    request: dict[str, Any]
    response: dict[str, Any]
    latency_ms: float


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
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.timeout = timeout
        self._base_url = (base_url or OPENROUTER_BASE_URL).rstrip("/")
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

    def _build_response_format(self, *, allow_schema: bool = True) -> dict:
        """Build the response_format payload based on model capabilities."""
        if allow_schema:
            return {
                "type": "json_schema",
                "json_schema": _ENRICHMENT_SCHEMA,
            }
        return {"type": "json_object"}

    def generate(self, user_prompt: str, max_tokens: int = 512) -> str:
        """Generate structured JSON from a user prompt.

        Uses OpenAI-compatible chat completions with structured JSON output.
        Returns the raw response string (parsed by doc_enrichment).
        """
        return self.generate_with_metadata(user_prompt, max_tokens=max_tokens)["content"]

    def generate_with_metadata(
        self, user_prompt: str, max_tokens: int = 512
    ) -> OpenRouterReplayMetadata:
        """Generate structured JSON plus request/response metadata."""
        return self._request_with_metadata(user_prompt, max_tokens=max_tokens)

    def _request_with_metadata(
        self, user_prompt: str, max_tokens: int = 512
    ) -> OpenRouterReplayMetadata:
        """Send completion request and return content with replay metadata."""
        request_timeout = self._build_request_timeout()
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
            "url": f"{self._base_url}/chat/completions",
            "timeout": {
                "connect": request_timeout.connect,
                "read": request_timeout.read,
                "write": request_timeout.write,
                "pool": request_timeout.pool,
            },
            "payload": payload,
        }

        last_exc: Exception | None = None
        started = time.perf_counter()

        for attempt in range(MAX_RETRIES):
            try:
                attempt_payload = copy.deepcopy(payload)
                trace_request["payload"] = attempt_payload
                resp = httpx.post(
                    f"{self._base_url}/chat/completions",
                    json=attempt_payload,
                    headers=headers,
                    timeout=request_timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                latency_ms = (time.perf_counter() - started) * 1000.0
                self.trace_recorder.record(
                    request=trace_request,
                    response=data,
                    success=True,
                    latency_ms=latency_ms,
                )
                content = data["choices"][0]["message"]["content"]
                return {
                    "content": content.strip(),
                    "request": copy.deepcopy(trace_request),
                    "response": copy.deepcopy(data),
                    "latency_ms": latency_ms,
                }

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc
                if attempt == MAX_RETRIES - 1:
                    break
                backoff = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                logger.warning(
                    "generate() attempt %d/%d failed (%s: %s), retrying in %.0fs...",
                    attempt + 1, MAX_RETRIES,
                    type(exc).__name__, exc, backoff,
                )
                time.sleep(backoff)

            except httpx.HTTPStatusError as exc:
                # 429 (rate limit) and 5xx are retryable; everything else is fatal.
                status = exc.response.status_code
                if (
                    status in (400, 422)
                    and payload["response_format"].get("type") == "json_schema"
                    and _is_response_format_rejection(exc.response)
                ):
                    logger.warning(
                        "OpenRouter model %s rejected json_schema response_format; "
                        "retrying with json_object.",
                        self.model,
                    )
                    payload["response_format"] = self._build_response_format(
                        allow_schema=False
                    )
                    continue
                if status == 429 or 500 <= status < 600:
                    last_exc = exc
                    if attempt == MAX_RETRIES - 1:
                        break
                    # Honor Retry-After header if present, else use exponential backoff.
                    retry_after = exc.response.headers.get("retry-after")
                    try:
                        backoff = float(retry_after) if retry_after else RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                    except ValueError:
                        backoff = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                    logger.warning(
                        "generate() attempt %d/%d failed (HTTP %d), retrying in %.0fs...",
                        attempt + 1, MAX_RETRIES, status, backoff,
                    )
                    time.sleep(backoff)
                    continue
                logger.error(
                    "OpenRouter API error: %d %s",
                    status,
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
        self._record_retry_failure(trace_request, started, last_exc)
        raise last_exc  # type: ignore[misc]

    def _record_retry_failure(
        self,
        trace_request: dict[str, Any],
        started: float,
        exc: Exception | None,
    ) -> None:
        error: dict[str, Any] = {
            "type": type(exc).__name__ if exc else "UnknownError",
            "message": str(exc) if exc else "Unknown OpenRouter error",
        }
        if isinstance(exc, httpx.HTTPStatusError):
            error.update(
                {
                    "status_code": exc.response.status_code,
                    "body": exc.response.text,
                }
            )
        self.trace_recorder.record(
            request=trace_request,
            success=False,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error=error,
        )

    def _build_request_timeout(self) -> httpx.Timeout:
        connect_timeout = min(self.timeout, CONNECT_TIMEOUT_CAP)
        return httpx.Timeout(
            timeout=self.timeout,
            connect=connect_timeout,
        )
