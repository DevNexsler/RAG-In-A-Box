"""LiteLLM/OpenAI-compatible generative LLM for document enrichment."""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from typing import Any, TypedDict

import httpx

from core.enrichment_telemetry import (
    record_structured_attempt,
    record_structured_retry,
)
from core.resilience import CIRCUITS, TransientError
from core.route_contract import ROUTE_CONTRACTS
from doc_enrichment import enrichment_response_schema, structured_response_is_usable
from providers.llm.trace_recorder import LLMTraceRecorder

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a document metadata extractor. You analyze document text and "
    "return exactly one JSON object containing every required enrichment field "
    "requested in the user prompt. Never include explanations, markdown fences, "
    "or any text outside the JSON object."
)

_QWEN_BULK_QUALITY_INSTRUCTIONS = """
Evidence-quality rules:
- Ground every value in the primary document. Do not infer urgency, disputes,
  compliance, system states, or application-specific labels not stated in it.
- For doc_type, use plain-language document genres with spaces; do not use
  internal, database, or application labels.
- For key_facts, emit one atomic fact per item. Preserve each explicit person,
  action, amount, date, address, warning, and requested next step. Do not
  merge or omit material facts.
- For a brief message, keep the summary and key facts close to its actual
  wording and stated purpose.
- For suggested_tags, use concise evidence-grounded terms only.
- For suggested_folder, use a concise, stable filing path grounded in the
  document. Do not synthesize operational inbox, state, organization, or date
  hierarchies from the document. When an Available Folders taxonomy is
  supplied, use one exact path from it instead.
"""

MAX_RETRIES = 2
RETRY_BACKOFF = (5.0, 15.0)
CONNECT_TIMEOUT_CAP = 10.0
DEFAULT_BASE_URL = "http://host.docker.internal:4000/v1"

# Every ``reasoning_effort`` an OpenAI-compatible gateway we speak to accepts.
# A gateway answers anything outside its own set with an HTTP 400, so a typo in
# config has to fail when the provider is built rather than on every document.
# ollama.com takes none|low|medium|high|max and only "none" actually disables
# reasoning (knowledge/runbooks/ollama-cloud-reasoning-truncation.md); OpenAI
# adds "minimal".
REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "max"})

# Enrichment keeps only the JSON body, so the recovery attempt buys no
# reasoning at all: it is the one lever that frees completion budget for the
# answer when reasoning has eaten it.
RECOVERY_REASONING_EFFORT = "none"

_ENRICHMENT_SCHEMA = {
    "name": "enrichment",
    "strict": True,
    "schema": enrichment_response_schema(),
}

_REQUIRED_ENRICHMENT_FIELDS = ("summary", "doc_type")


def _enrichment_request_policy(model: str, temperature: float) -> dict[str, Any]:
    """Return one centralized request policy for an enrichment model."""
    if model.casefold() == "qwen-bulk":
        return {
            "payload": {
                "response_format": {"type": "json_object"},
                "temperature": 0.4,
                "top_p": 0.8,
                "presence_penalty": 0.5,
                "extra_body": {
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            },
            "system_prompt": _SYSTEM_PROMPT + _QWEN_BULK_QUALITY_INSTRUCTIONS,
            "validate_structured_json": True,
            "retry_without_reasoning": False,
        }
    return {
        "payload": {
            "response_format": {
                "type": "json_schema",
                "json_schema": _ENRICHMENT_SCHEMA,
            },
            "temperature": temperature,
        },
        "system_prompt": _SYSTEM_PROMPT,
        "validate_structured_json": False,
        "retry_without_reasoning": True,
    }


def _validate_enrichment_content(content: str) -> None:
    """Reject malformed JSON and missing core enrichment fields before return."""
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("structured response must be a JSON object")
    missing = [field for field in _REQUIRED_ENRICHMENT_FIELDS if not parsed.get(field)]
    if missing:
        raise ValueError(
            "structured response missing required fields: " + ", ".join(missing)
        )


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
    """Diagnose *why* a response was unusable, without trusting ``stop``.

    These are evidence for the log and for classifying a failed retry, not the
    usability verdict: that is ``structured_response_is_usable`` on the payload
    itself.
    """
    message: dict[str, Any] = {}
    finish_reason = ""
    choices = response_payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        finish_reason = str(choices[0].get("finish_reason") or "").strip().lower()
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
    # A budget that truncates stops generation *at* the budget. Billing above
    # it proves the budget never bound, so it cannot be what cut the answer
    # short — this proxy bills reasoning tokens it then strips from the message,
    # so complete answers routinely report more than they were allowed (#1097).
    saturated_budget = (
        completion_tokens is not None
        and requested_tokens is not None
        and requested_tokens > 0
        and completion_tokens == requested_tokens
    )
    truncated = saturated_budget or empty_content or finish_reason == "length"
    return {
        "completion_tokens": completion_tokens,
        "requested_tokens": requested_tokens,
        "empty_content": empty_content,
        "finish_reason": finish_reason,
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
        reasoning_effort: str | None = None,
    ) -> None:
        if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(
                "enrichment.reasoning_effort must be one of "
                f"{sorted(REASONING_EFFORTS)}, got {reasoning_effort!r}"
            )
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url.rstrip("/")
        self.api_key = (
            api_key
            or os.environ.get("LITELLM_API_KEY", "")
            or os.environ.get("LITELLM_MASTER_KEY", "")
        )
        self.timeout = timeout
        self.temperature = temperature
        self._request_policy = _enrichment_request_policy(model, temperature)
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

    @property
    def route(self) -> str:
        """The alias this generator calls, as an identity a log line can name."""
        return f"{self.model}@{self.base_url}"

    def generate(self, user_prompt: str, max_tokens: int = 512) -> str:
        return self.generate_with_metadata(user_prompt, max_tokens=max_tokens)["content"]

    def generate_with_metadata(
        self, user_prompt: str, max_tokens: int = 512
    ) -> LiteLLMReplayMetadata:
        initial = self._request_with_metadata(
            user_prompt,
            max_tokens=max_tokens,
            reasoning_effort=self.reasoning_effort,
        )
        signals = _truncation_signals(
            initial["response"], initial["request"]["payload"]
        )
        first_pass_usable = self._is_usable(initial["content"], signals)
        record_structured_attempt(first_pass_usable=first_pass_usable)
        if first_pass_usable:
            return initial

        retry_without_reasoning = self._request_policy["retry_without_reasoning"]
        # Disabling reasoning is the only lever that frees completion budget for
        # the JSON.  Pulling it twice buys nothing: a retry with the parameters
        # that just failed burns exactly the same way (#0260).
        recoverable = self.reasoning_effort != RECOVERY_REASONING_EFFORT
        logger.warning(
            "LiteLLM structured response was not usable "
            "(completion_tokens=%s/%s, reasoning_chars=%s, empty=%s, "
            "finish_reason=%s, truncated=%s); %s",
            signals["completion_tokens"],
            signals["requested_tokens"],
            signals["reasoning_output_length"],
            signals["empty_content"],
            signals["finish_reason"],
            signals["truncated"],
            (
                "retrying once with reasoning disabled."
                if retry_without_reasoning
                else "retrying once."
            )
            if recoverable
            else "reasoning is already disabled, not retrying.",
        )
        if not recoverable:
            raise TransientError(
                "LiteLLM structured response was truncated or empty with "
                "reasoning already disabled"
            )
        recovered = self._request_with_metadata(
            user_prompt,
            max_tokens=max_tokens,
            reasoning_effort=(
                RECOVERY_REASONING_EFFORT if retry_without_reasoning else None
            ),
        )
        recovery_signals = _truncation_signals(
            recovered["response"], recovered["request"]["payload"]
        )
        recovery_usable = self._is_usable(recovered["content"], recovery_signals)
        record_structured_retry(usable=recovery_usable)
        if recovery_usable:
            return recovered

        if recovery_signals["truncated"]:
            raise TransientError(
                "LiteLLM structured response remained truncated or empty "
                "after reasoning-disabled retry"
            )
        # The model answered in full and still withheld required metadata:
        # that is a fact about this document, not a provider outage. Return it
        # so the enrichment consumer records a permanent degradation instead of
        # a transient one the degraded ledger would re-queue forever (#0251).
        return recovered

    @staticmethod
    def _is_usable(content: str, signals: dict[str, Any]) -> bool:
        """Whether a structured response is worth keeping.

        The payload decides: it either yields the enrichment fields we asked
        for or it does not. Truncation evidence only vetoes — a cut-short
        answer can still salvage-parse with the required fields present while
        silently dropping every later field.
        """
        return not signals["truncated"] and structured_response_is_usable(content)

    def _request_with_metadata(
        self,
        user_prompt: str,
        *,
        max_tokens: int,
        reasoning_effort: str | None = None,
    ) -> LiteLLMReplayMetadata:
        request_timeout = self._build_request_timeout()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._request_policy["system_prompt"]},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
        }
        payload.update(copy.deepcopy(self._request_policy["payload"]))
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
                # The breaker fails fast (CircuitOpenError, a TransientError) while
                # this proxy is in cooldown, so a proxy recreate is discovered once
                # per provider instead of once per document (#0619).
                with CIRCUITS.guard(self.base_url):
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
                # The alias is fixed in config; the contract behind it is the
                # provider's to change. Every answer carries that evidence, so
                # it is read here rather than off the returned one: an answer a
                # retry below discards — cut mid-JSON at the budget, or refused
                # for its response_format — is often the only one that ever
                # reaches the boundary that proves the contract (#1154, #1629).
                observed = _truncation_signals(data, attempt_payload)
                ROUTE_CONTRACTS.observe(
                    self.route,
                    requested_tokens=observed["requested_tokens"],
                    completion_tokens=observed["completion_tokens"],
                    finish_reason=observed["finish_reason"],
                    backend=data.get("system_fingerprint"),
                )
                raw_content = data["choices"][0]["message"].get("content")
                content = raw_content.strip() if isinstance(raw_content, str) else ""
                if self._request_policy["validate_structured_json"]:
                    try:
                        _validate_enrichment_content(content)
                    except (TypeError, ValueError, json.JSONDecodeError) as exc:
                        last_exc = TransientError(
                            f"LiteLLM structured response validation failed: {exc}"
                        )
                        if attempt == MAX_RETRIES - 1:
                            break
                        backoff = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                        logger.warning(
                            "LiteLLM structured response failed validation; "
                            "retrying in %.0fs...",
                            backoff,
                        )
                        time.sleep(backoff)
                        continue
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
                    payload["response_format"] = {"type": "json_object"}
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
