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

Between the two sits the per-endpoint circuit breaker (EndpointCircuits): a refused
connection is not information about the document, it is information about the provider,
so it is remembered per base_url instead of being re-discovered once per document.
"""
from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator, TypeVar

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


# --- Per-endpoint circuit breaker -------------------------------------------
#
# A connection-level failure says the provider is gone, not that this document is
# bad — yet without a breaker every document rediscovers it, each burning its own
# retry ladder against a socket that refuses instantly (#0619: 209 connection-refused
# warnings in one 2-minute litellm-proxy recreate). After CIRCUIT_FAILURE_THRESHOLD
# consecutive connection-level failures to one base_url, calls to it fail fast for
# CIRCUIT_COOLDOWN_SECONDS; one probe is admitted once the cooldown elapses.

CIRCUIT_FAILURE_THRESHOLD = 3
CIRCUIT_COOLDOWN_SECONDS = 60.0

# Failures that mean "no one answered at this address". A 5xx or a malformed body
# is deliberately NOT here: the endpoint answered, so that is per-request news.
_CONNECTION_LEVEL_EXC = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    ConnectionError,
)


class CircuitOpenError(TransientError):
    """The endpoint is in cooldown after repeated connection-level failures.

    Transient by inheritance: the caller degrades the doc to the ledger and it
    self-heals once the provider is back, exactly as an outage-time timeout would."""


def is_connection_level(exc: BaseException) -> bool:
    """True if `exc` means the endpoint could not be reached at all."""
    return isinstance(exc, _CONNECTION_LEVEL_EXC)


class EndpointCircuits:
    """Circuit state for every endpoint, keyed by base_url. Thread-safe: indexing
    processes documents concurrently against the same providers."""

    def __init__(
        self,
        *,
        threshold: int = CIRCUIT_FAILURE_THRESHOLD,
        cooldown: float = CIRCUIT_COOLDOWN_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = max(1, threshold)
        self._cooldown = cooldown
        self._clock = clock
        self._lock = threading.Lock()
        # key -> {"failures": int, "open_until": float | None}
        self._state: dict[str, dict] = {}

    def reset(self) -> None:
        with self._lock:
            self._state.clear()

    def is_open(self, key: str | None) -> bool:
        """True while `key` is in cooldown — no point retrying it right now."""
        if not key:
            return False
        with self._lock:
            state = self._state.get(key)
            open_until = state and state["open_until"]
            return open_until is not None and self._clock() < open_until

    def _enter(self, key: str) -> None:
        with self._lock:
            state = self._state.get(key)
            open_until = state and state["open_until"]
            if open_until is None:
                return
            if self._clock() < open_until:
                raise CircuitOpenError(
                    f"circuit open for {key} after {state['failures']} consecutive "
                    f"connection failures — {open_until - self._clock():.0f}s of cooldown left"
                )
            # Cooldown elapsed: admit exactly one probe. Concurrent callers keep
            # failing fast until the probe reports back.
            state["open_until"] = self._clock() + self._cooldown

    def _record(self, key: str, exc: BaseException | None) -> None:
        with self._lock:
            state = self._state.setdefault(key, {"failures": 0, "open_until": None})
            if exc is None or not is_connection_level(exc):
                # Answered (or succeeded) — the endpoint is reachable.
                state["failures"] = 0
                state["open_until"] = None
                return
            state["failures"] += 1
            if state["failures"] >= self._threshold and state["open_until"] is None:
                state["open_until"] = self._clock() + self._cooldown
                logger.warning(
                    "%s refused %d consecutive connections — pausing calls to it for %.0fs",
                    key, state["failures"], self._cooldown,
                )
            elif state["open_until"] is not None:
                # A failed probe re-arms the cooldown from now.
                state["open_until"] = self._clock() + self._cooldown

    @contextmanager
    def guard(self, key: str | None) -> Iterator[None]:
        """Short-circuit the block when `key`'s endpoint is in cooldown, and feed
        its outcome back into that endpoint's failure run. `key=None` disables it."""
        if not key:
            yield
            return
        self._enter(key)
        try:
            yield
        except CircuitOpenError:
            raise
        except BaseException as exc:  # noqa: BLE001 — classify, then re-raise
            self._record(key, exc)
            raise
        else:
            self._record(key, None)


CIRCUITS = EndpointCircuits()


def call_with_retry(
    fn: Callable[[], T],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    backoff: tuple[float, ...] = DEFAULT_BACKOFF,
    label: str = "external call",
    classify: Callable[[BaseException], bool] = is_transient,
    sleep: Callable[[float], None] = time.sleep,
    circuit_key: str | None = None,
    circuits: EndpointCircuits | None = None,
) -> T:
    """Call `fn()`; retry on TRANSIENT failures with backoff, raise PERMANENT ones
    immediately. After `attempts` transient failures, re-raise the last exception so
    the caller can degrade the doc (-> degraded ledger -> self-heal next run).

    `backoff[i]` is the delay before attempt i+1 (the last value repeats). A caller can
    override `classify` (e.g. to honor a Retry-After) or inject `sleep` (tests).

    Pass `circuit_key` (the provider's base_url) to route the call through the
    per-endpoint breaker: while that endpoint is in cooldown the call fails
    immediately with CircuitOpenError instead of re-walking the retry ladder.
    """
    breaker = circuits if circuits is not None else CIRCUITS
    last: BaseException | None = None
    for i in range(max(1, attempts)):
        try:
            with breaker.guard(circuit_key):
                return fn()
        except CircuitOpenError:
            raise  # known-down provider: never sleep on it
        except Exception as exc:  # noqa: BLE001 — classify, then retry or re-raise
            last = exc
            if not classify(exc) or i >= attempts - 1:
                raise
            if breaker.is_open(circuit_key):
                # This attempt tripped the breaker: the endpoint is down for
                # everyone, so re-raise the real error now instead of sleeping.
                raise
            delay = backoff[min(i, len(backoff) - 1)] if backoff else 0.0
            logger.warning(
                "%s: attempt %d/%d failed transiently (%s: %s) — retrying in %.0fs",
                label, i + 1, attempts, type(exc).__name__, str(exc)[:160], delay,
            )
            sleep(delay)
    assert last is not None  # pragma: no cover — loop always runs >=1
    raise last
