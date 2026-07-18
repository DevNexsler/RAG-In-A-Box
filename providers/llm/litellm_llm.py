"""LiteLLM/OpenAI-compatible generative LLM for document enrichment."""

from __future__ import annotations

import copy
import logging
import os
import time
from typing import Any, TypedDict

import httpx

from core.resilience import TransientError
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
DEFAULT_BASE_URL = "http://host.docker.internal:4000/v1"

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


def _truncation_signals(
    response_payload: dict[str, Any],
    request_payload: dict[str, Any],
) -> dict[str, Any]:
    """Return OpenAI-compatible truncation evidence without trusting finish_reason."""
    message: dict[str, Any] = {}
    choices = response_payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        raw_message = choices[0].get("message")
        if isinstance(raw_message, dict):
            message = raw_message

    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if not reasoning:
        provider_fields = message.get("provider_specific_fields")
        if isinstance(provider_fields, dict):
            reasoning = (
                provider_fields.get("reasoning_content")
                or provider_fields.get("reasoning")
                or ""
            )

    completion_tokens: int | None = None
    usage = response_payload.get("usage")
    if isinstance(usage, dict):
        try:
            completion_tokens = int(usage.get("completion_tokens"))
        except (TypeError, ValueError):
            completion_tokens = None

    requested_tokens: int | None = None
    try:
        requested_tokens = int(request_payload.get("max_tokens"))
    except (TypeError, ValueError):
        requested_tokens = None

    content = message.get("content")
    empty_content = (
        content is None
        or content == []
        or (isinstance(content, str) and not content.strip())
    )
    reasoning_length = len(reasoning) if isinstance(reasoning, str) else 0
    truncated = (
        completion_tokens is not None
        and requested_tokens is not None
        and requested_tokens > 0
        and completion_tokens >= requested_tokens
    ) or empty_content
    return {
        "completion_tokens": completion_tokens,
        "empty_content": empty_content,
        "reasoning_output_length": reasoning_length,
        "truncated": truncated,
    }


class LiteLLMReplayMetadata(TypedDict):
    content: str
    request: dict[str, Any]
    response: dict[str, Any]
    latency_ms: float


class LiteLLMGenerator:
    """Text generation via LiteLLM proxy using OpenAI-compatible chat completions."""

    def __init__(
        self,
        model: str = "ollama-deepseek-v4-pro",
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        timeout: float = 600.0,
        trace_capture: dict | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = (
            api_key
            or os.environ.get("LITELLM_API_KEY", "")
            or os.environ.get("LITELLM_MASTER_KEY", "")
        )
        self.timeout = timeout
        self.temperature = temperature
        trace_capture = trace_capture or {}
        self.trace_recorder = LLMTraceRecorder(
            provider="litellm",
            model=model,
            enabled=bool(trace_capture.get("enabled", False)),
            directory=trace_capture.get("directory", ".evals/llm-traces"),
        )

        if not self.api_key:
            raise ValueError(
                "LITELLM_API_KEY or LITELLM_MASTER_KEY not set. "
                "Set it in .env or pass enrichment.api_key."
            )

        logger.info("LiteLLMGenerator initialized: %s model=%s", self.base_url, model)

    def _build_response_format(self, *, allow_schema: bool = True) -> dict:
        if allow_schema:
            return {
                "type": "json_schema",
                "json_schema": _ENRICHMENT_SCHEMA,
            }
        return {"type": "json_object"}

    def generate(self, user_prompt: str, max_tokens: int = 512) -> str:
        return self.generate_with_metadata(user_prompt, max_tokens=max_tokens)["content"]

    def generate_with_metadata(
        self, user_prompt: str, max_tokens: int = 512
    ) -> LiteLLMReplayMetadata:
        initial = self._request_with_metadata(user_prompt, max_tokens=max_tokens)
        signals = _truncation_signals(
            initial["response"], initial["request"]["payload"]
        )
        if not signals["truncated"]:
            return initial

        logger.warning(
            "LiteLLM structured response was truncated or empty "
            "(completion_tokens=%s, reasoning_chars=%s, empty=%s); retrying "
            "once with reasoning disabled.",
            signals["completion_tokens"],
            signals["reasoning_output_length"],
            signals["empty_content"],
        )
        recovered = self._request_with_metadata(
            user_prompt,
            max_tokens=max_tokens,
            reasoning_effort="none",
        )
        recovery_signals = _truncation_signals(
            recovered["response"], recovered["request"]["payload"]
        )
        if recovery_signals["truncated"]:
            raise TransientError(
                "LiteLLM structured response remained truncated or empty "
                "after reasoning-disabled retry"
            )
        return recovered

    def _request_with_metadata(
        self,
        user_prompt: str,
        *,
        max_tokens: int,
        reasoning_effort: str | None = None,
    ) -> LiteLLMReplayMetadata:
        request_timeout = self._build_request_timeout()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "response_format": self._build_response_format(),
        }
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        trace_request = {
            "url": f"{self.base_url}/chat/completions",
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
                    f"{self.base_url}/chat/completions",
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
                raw_content = data["choices"][0]["message"].get("content")
                content = raw_content.strip() if isinstance(raw_content, str) else ""
                return {
                    "content": content,
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
                    "LiteLLM generate() attempt %d/%d failed (%s: %s), retrying in %.0fs...",
                    attempt + 1, MAX_RETRIES,
                    type(exc).__name__, exc, backoff,
                )
                time.sleep(backoff)

            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if (
                    status in (400, 422)
                    and payload["response_format"].get("type") == "json_schema"
                    and _is_response_format_rejection(exc.response)
                ):
                    logger.warning(
                        "LiteLLM model %s rejected json_schema response_format; "
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
                    retry_after = exc.response.headers.get("retry-after")
                    try:
                        backoff = (
                            float(retry_after)
                            if retry_after
                            else RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                        )
                    except ValueError:
                        backoff = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                    logger.warning(
                        "LiteLLM generate() attempt %d/%d failed (HTTP %d), "
                        "retrying in %.0fs...",
                        attempt + 1, MAX_RETRIES, status, backoff,
                    )
                    time.sleep(backoff)
                    continue
                logger.error("LiteLLM API error: HTTP %d", status)
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
            "message": str(exc) if exc else "Unknown LiteLLM error",
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
