"""Generalized retry/backoff for EVERY external call — local OCR/vision and cloud
LLM/embeddings alike.

Before this, each provider reinvented its own retry loop (litellm MAX_RETRIES=2,
openrouter MAX_RETRIES=5, ollama bespoke) and deepseek had none — so a transient
blip (a 504, a connection reset, a rate limit, a model still cold-loading) failed
differently depending on which provider you hit. This is the single place that:

  1. classifies a failure as TRANSIENT (upstream slow/unavailable/rate-limited —
     worth retrying) vs PERMANENT (bad input / auth / not-found — don't retry), and
  2. backs off and retries the transient ones.

On exhaustion the original exception is re-raised UNCHANGED, so the caller's existing
degrade path still fires (note_degradation/failed_enrichment -> the degraded ledger ->
the doc is retried on a later run). Retry handles the seconds-scale blip; the degraded
ledger handles the minutes/hours-scale outage. Together that is the self-healing.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Sensible default for any external call. Providers may pass their own.
DEFAULT_ATTEMPTS = 3
DEFAULT_BACKOFF: tuple[float, ...] = (2.0, 5.0, 15.0)

# HTTP statuses worth retrying: request-timeout, too-early, rate-limit, and all 5xx
# (502/503/504 = upstream gateway/overload, 500 = transient server error).
TRANSIENT_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})

# Network-layer failures that are transient by nature.
_TRANSIENT_EXC = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
    httpx.PoolTimeout,
    ConnectionError,
    TimeoutError,
)


class TransientError(RuntimeError):
    """Raise from inside a wrapped call to FORCE a retry — for failures that don't
    surface as an exception on their own, e.g. an upstream error returned inside an
    HTTP 200 body (OpenRouter) or a model that returned an empty result.

    Subclasses RuntimeError so that if retries exhaust, the re-raised error is still a
    RuntimeError to callers that catch that (backward-compatible)."""


def is_transient(exc: BaseException) -> bool:
    """True if `exc` is a failure we expect to clear on its own."""
    if isinstance(exc, TransientError):
        return True
    if isinstance(exc, _TRANSIENT_EXC):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        s = exc.response.status_code
        return s in TRANSIENT_STATUSES or s >= 500
    return False


def call_with_retry(
    fn: Callable[[], T],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    backoff: tuple[float, ...] = DEFAULT_BACKOFF,
    label: str = "external call",
    classify: Callable[[BaseException], bool] = is_transient,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call `fn()`; retry on TRANSIENT failures with backoff, raise PERMANENT ones
    immediately. After `attempts` transient failures, re-raise the last exception so
    the caller can degrade the doc (-> degraded ledger -> self-heal next run).

    `backoff[i]` is the delay before attempt i+1 (the last value repeats). A caller can
    override `classify` (e.g. to honor a Retry-After) or inject `sleep` (tests).
    """
    last: BaseException | None = None
    for i in range(max(1, attempts)):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — classify, then retry or re-raise
            last = exc
            if not classify(exc) or i >= attempts - 1:
                raise
            delay = backoff[min(i, len(backoff) - 1)] if backoff else 0.0
            logger.warning(
                "%s: attempt %d/%d failed transiently (%s: %s) — retrying in %.0fs",
                label, i + 1, attempts, type(exc).__name__, str(exc)[:160], delay,
            )
            sleep(delay)
    assert last is not None  # pragma: no cover — loop always runs >=1
    raise last
