"""Durable freshness watermark for the index tail and its pending backlog.

Completed tasks measure throughput, not currency. On 2026-08-26 the indexer
logged 2,201 completed documents and 2,129 chunk writes inside one sampled
window while the newest content it had indexed was 17h old and 19 newer source
requests still sat at ``attempts=0`` (#1625): a long historical drain holds the
table writer lock for hours and writes real, valid, *old* records the whole
time. No success count can tell that state apart from a current tail.

Two numbers can, and this module owns both:

* the content timestamp of the newest item the indexer has processed — stamped
  at the store write seam into a file of its own, so the run-scoped rewrite of
  ``index_metadata.json`` cannot drop it;
* the age of the oldest still-pending targeted request — read live from the
  durable request queue, which is the authority on what has *not* been served.

Both outlive the run that produced them (the full sweep runs in a subprocess,
the targeted path in the server), so ``/health`` can mark a stale tail while
tasks keep succeeding.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from core.index_request_queue import pending_backlog

WATERMARK_FILENAME = "index_freshness.json"

# Freshness SLO for queued source work. The scheduled drain runs about once a
# minute and the sweep now serves the queue mid-run, so anything still pending
# after half an hour means newer work is not reaching the index — the #1625
# state, where the oldest pending request had waited 5h28m.
DEFAULT_MAX_PENDING_AGE_S = 1800.0

# Content-time fields, most specific first. Communication records carry
# ``sent_at``; other sources fall back to their own creation stamp and finally
# to file mtime, so the watermark is defined for every source type rather than
# for the one that happened to fail.
_CONTENT_TIMESTAMP_KEYS = ("sent_at", "created_at", "timestamp", "created")

_WRITE_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse(value: Any) -> datetime | None:
    """Parse an ISO-8601 stamp, treating a naive one as UTC. None when unusable."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):  # fromisoformat only accepts Z from 3.11
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _normalize(value: Any) -> str | None:
    """Normalize a content stamp to UTC ISO-8601 so recorded values compare."""
    parsed = _parse(value)
    return parsed.astimezone(timezone.utc).isoformat() if parsed else None


def _newer(current: Any, candidate: Any) -> str | None:
    """The later of two recorded stamps, either of which may be missing."""
    current_at, candidate_at = _parse(current), _parse(candidate)
    if current_at is None:
        return _normalize(candidate)
    if candidate_at is None:
        return _normalize(current)
    return _normalize(candidate if candidate_at > current_at else current)


def content_timestamp(
    metadata: Mapping[str, Any] | None,
    *,
    mtime: float | None = None,
) -> str | None:
    """When the content of one document was *created*, not when it was indexed.

    This is the number that separates a current tail from a historical drain:
    an attachment sent this morning and one sent in 2023 are both a successful
    write, and only their content time says which one the index just caught up
    to.
    """
    for key in _CONTENT_TIMESTAMP_KEYS:
        stamped = _normalize((metadata or {}).get(key))
        if stamped:
            return stamped
    if mtime is None:
        return None
    try:
        return datetime.fromtimestamp(float(mtime), timezone.utc).isoformat()
    except (OverflowError, OSError, TypeError, ValueError):
        return None


def read_watermark(index_root: str | Path) -> dict[str, Any]:
    """Last recorded watermark, or an empty one. Never raises for the caller."""
    path = Path(index_root) / WATERMARK_FILENAME
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def record_indexed_content(
    index_root: str | Path,
    *,
    doc_id: str,
    content_at: str | None,
) -> None:
    """Advance the watermark for one document the indexer just wrote.

    Stamped per written document, so it is an atomic replace rather than an
    fsync — the same durability the indexer heartbeat settles for. Sweep worker
    threads serialize on a process lock; a targeted index racing in the server
    process can only lose its update to a newer one, never tear the file.
    """
    root = Path(index_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / WATERMARK_FILENAME
    stamped = _normalize(content_at)
    with _WRITE_LOCK:
        previous = read_watermark(root)
        payload = {
            "last_indexed_at": _utc_now(),
            "last_indexed_doc_id": str(doc_id),
            "last_content_at": stamped,
            "newest_content_at": _newer(previous.get("newest_content_at"), stamped),
        }
        temp_path = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        temp_path.write_text(json.dumps(payload, sort_keys=True))
        os.replace(temp_path, path)


def _age_seconds(stamp: Any, now: datetime) -> float | None:
    parsed = _parse(stamp)
    if parsed is None:
        return None
    return round(max(0.0, (now - parsed).total_seconds()), 1)


def freshness_summary(
    index_root: str | Path,
    table_name: str,
    *,
    max_pending_age_s: float = DEFAULT_MAX_PENDING_AGE_S,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Tail currency and queue backlog as one alertable payload.

    ``stale_tail`` keys on the pending backlog rather than on ``content_lag_s``:
    a historical drain legitimately writes years-old content, so content lag
    alone would alarm on healthy backfill. Requests that stay queued past the
    SLO cannot be legitimate — something newer is not reaching the index. It
    clears by itself once the backlog drains, so the alert recovers without an
    operator acknowledging anything.
    """
    now = now or datetime.now(timezone.utc)
    watermark = read_watermark(index_root)
    backlog = pending_backlog(index_root, table_name)
    oldest_pending_age_s = _age_seconds(backlog["oldest_created_at"], now)
    threshold = float(max_pending_age_s)
    return {
        "last_indexed_at": watermark.get("last_indexed_at"),
        "last_indexed_doc_id": watermark.get("last_indexed_doc_id"),
        "last_content_at": watermark.get("last_content_at"),
        "newest_content_at": watermark.get("newest_content_at"),
        "content_lag_s": _age_seconds(watermark.get("newest_content_at"), now),
        "pending_requests": backlog["pending"],
        "oldest_pending_created_at": backlog["oldest_created_at"],
        "oldest_pending_age_s": oldest_pending_age_s,
        "max_pending_age_s": threshold,
        "stale_tail": (
            oldest_pending_age_s is not None and oldest_pending_age_s > threshold
        ),
    }
