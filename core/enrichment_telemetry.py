"""Per-run counters for structured enrichment output quality.

How healthy a run's enrichment calls were used to be answerable only by
grepping per-document WARNING lines out of `indexer.log`, which cannot
distinguish "the model returned unusable metadata" from "we misjudged a good
response and paid for a second call" (#1097).

The enrichment LLM client records each structured call's outcome here; the
indexing flow resets the counters when a run starts and reads a snapshot at
finalize, so the run summary and `index_metadata.json` carry first-pass
structured validity and the retry count next to the degraded-write count.

Counters are process-wide and thread-safe: the flow enriches documents on a
worker pool, and a run is one process.
"""

from __future__ import annotations

import threading
from collections import Counter

_lock = threading.Lock()
_counters: Counter[str] = Counter()


def reset() -> None:
    """Drop counts from an earlier run in this process."""
    with _lock:
        _counters.clear()


def record_structured_attempt(*, first_pass_usable: bool) -> None:
    """Record one enrichment call and whether its first response was usable."""
    with _lock:
        _counters["attempts"] += 1
        if first_pass_usable:
            _counters["first_pass_usable"] += 1


def record_structured_retry(*, usable: bool) -> None:
    """Record a second request made because the first response was unusable."""
    with _lock:
        _counters["retries"] += 1
        if usable:
            _counters["retries_recovered"] += 1


def snapshot() -> dict[str, int]:
    """Current counts, always with every key present so consumers can index."""
    with _lock:
        return {
            key: _counters[key]
            for key in (
                "attempts",
                "first_pass_usable",
                "retries",
                "retries_recovered",
            )
        }
