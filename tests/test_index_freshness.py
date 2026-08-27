"""Freshness watermark and pending-request backlog (#1625).

Thousands of `Completed()` tasks and `Inserted N chunks` lines are compatible
with a tail 17h stale: the sweep drains history under the writer lock and every
one of those writes is real. These cover the two numbers that can tell the
difference — the content time of the newest item written, and the age of the
oldest source request still queued — plus the alert state built on them.
"""

from datetime import datetime, timedelta, timezone

from core.index_freshness import (
    DEFAULT_MAX_PENDING_AGE_S,
    WATERMARK_FILENAME,
    content_timestamp,
    freshness_summary,
    read_watermark,
    record_indexed_content,
)
from core.index_request_queue import IndexRequestQueue, pending_backlog


def _now() -> datetime:
    return datetime(2026, 8, 26, 6, 9, 39, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# content_timestamp — when the item was created, not when it was indexed
# ---------------------------------------------------------------------------


def test_content_timestamp_prefers_communication_sent_at():
    """`sent_at` is the item's own time; mtime is only when the file landed."""
    stamped = content_timestamp(
        {"sent_at": "2026-06-15T09:30:00+00:00", "created_at": "2026-08-26T00:00:00Z"},
        mtime=1_756_000_000.0,
    )

    assert stamped == "2026-06-15T09:30:00+00:00"


def test_content_timestamp_normalizes_zulu_and_offset_stamps_to_utc():
    """Recorded stamps have to be comparable across sources, so normalize."""
    assert content_timestamp({"sent_at": "2026-08-26T00:32:26Z"}) == (
        "2026-08-26T00:32:26+00:00"
    )
    assert content_timestamp({"sent_at": "2026-08-25T20:32:26-04:00"}) == (
        "2026-08-26T00:32:26+00:00"
    )


def test_content_timestamp_falls_through_unparseable_fields_to_mtime():
    """A source with no usable content time still gets a watermark — the class
    of document without `sent_at` must not silently stop the tail signal."""
    mtime = datetime(2026, 8, 26, 0, 41, 8, tzinfo=timezone.utc).timestamp()

    assert content_timestamp({"created": "last thursday"}, mtime=mtime) == (
        "2026-08-26T00:41:08+00:00"
    )
    assert content_timestamp({}, mtime=None) is None


# ---------------------------------------------------------------------------
# watermark — durable across runs and processes
# ---------------------------------------------------------------------------


def test_watermark_records_last_write_and_keeps_the_newest_content_time(tmp_path):
    """A historical drain writes older and older content; `newest_content_at`
    is a high-water mark, `last_content_at` is what the run just touched."""
    record_indexed_content(
        tmp_path, doc_id="documents::002L8", content_at="2026-08-25T12:46:28+00:00"
    )
    record_indexed_content(
        tmp_path, doc_id="documents::00old", content_at="2023-06-09T11:00:00+00:00"
    )

    watermark = read_watermark(tmp_path)

    assert watermark["last_indexed_doc_id"] == "documents::00old"
    assert watermark["last_content_at"] == "2023-06-09T11:00:00+00:00"
    assert watermark["newest_content_at"] == "2026-08-25T12:46:28+00:00"
    assert watermark["last_indexed_at"] > "2026"


def test_watermark_survives_a_document_with_no_content_time(tmp_path):
    """A missing content time records the write without erasing the tail."""
    record_indexed_content(
        tmp_path, doc_id="documents::00aaa", content_at="2026-08-25T12:46:28+00:00"
    )
    record_indexed_content(tmp_path, doc_id="documents::00bbb", content_at=None)

    watermark = read_watermark(tmp_path)

    assert watermark["last_content_at"] is None
    assert watermark["newest_content_at"] == "2026-08-25T12:46:28+00:00"


def test_read_watermark_tolerates_a_missing_or_corrupt_file(tmp_path):
    assert read_watermark(tmp_path) == {}
    (tmp_path / WATERMARK_FILENAME).write_text("{not json")
    assert read_watermark(tmp_path) == {}


# ---------------------------------------------------------------------------
# backlog — read-only; a probe must not create the queue it is measuring
# ---------------------------------------------------------------------------


def test_pending_backlog_is_empty_and_side_effect_free_without_a_queue(tmp_path):
    assert pending_backlog(tmp_path, "chunks") == {
        "pending": 0,
        "oldest_created_at": None,
    }
    assert list(tmp_path.iterdir()) == []


def test_pending_backlog_counts_pending_rows_and_reports_the_oldest(tmp_path):
    queue = IndexRequestQueue(tmp_path)
    first = queue.enqueue("chunks", "documents", "older.bin")
    queue.enqueue("chunks", "documents", "newer.bin")
    queue.enqueue("other_table", "documents", "unrelated.bin")

    backlog = pending_backlog(tmp_path, "chunks")

    assert backlog["pending"] == 2
    assert backlog["oldest_created_at"] == first.created_at


# ---------------------------------------------------------------------------
# freshness_summary — the alertable state, and its recovery
# ---------------------------------------------------------------------------


def test_freshness_summary_is_clean_with_an_empty_queue(tmp_path):
    record_indexed_content(
        tmp_path, doc_id="documents::002L8", content_at="2026-08-26T06:00:00+00:00"
    )

    summary = freshness_summary(tmp_path, "chunks", now=_now())

    assert summary["stale_tail"] is False
    assert summary["pending_requests"] == 0
    assert summary["oldest_pending_age_s"] is None
    assert summary["content_lag_s"] == 579.0
    assert summary["max_pending_age_s"] == DEFAULT_MAX_PENDING_AGE_S


def test_freshness_summary_marks_a_stale_tail_and_clears_when_it_drains(tmp_path):
    """The #1625 state and its recovery: a request queued past the SLO is
    stale-tail regardless of how many documents completed meanwhile, and the
    state clears by itself once the request is served."""
    queue = IndexRequestQueue(tmp_path)
    request = queue.enqueue("chunks", "documents", "2026-08-26__msg678132__mm0.bin")
    queued_at = datetime.fromisoformat(request.created_at)

    stale = freshness_summary(
        tmp_path, "chunks", now=queued_at + timedelta(hours=5, minutes=28)
    )
    assert stale["stale_tail"] is True
    assert stale["pending_requests"] == 1
    assert stale["oldest_pending_age_s"] == 19680.0

    queue.complete(request)
    recovered = freshness_summary(
        tmp_path, "chunks", now=queued_at + timedelta(hours=5, minutes=29)
    )
    assert recovered["stale_tail"] is False
    assert recovered["pending_requests"] == 0


def test_freshness_summary_tolerates_a_request_inside_the_slo(tmp_path):
    """Normal queue traffic must not alarm — only work that waits past the SLO."""
    queue = IndexRequestQueue(tmp_path)
    request = queue.enqueue("chunks", "documents", "fresh.bin")
    queued_at = datetime.fromisoformat(request.created_at)

    summary = freshness_summary(
        tmp_path,
        "chunks",
        max_pending_age_s=1800,
        now=queued_at + timedelta(seconds=1799),
    )

    assert summary["stale_tail"] is False
    assert summary["pending_requests"] == 1


def test_freshness_summary_alarms_on_a_stale_tail_with_no_watermark_yet(tmp_path):
    """A backlog older than the SLO is actionable even before the first write —
    the alert may not assume the index has ever recorded anything."""
    queue = IndexRequestQueue(tmp_path)
    request = queue.enqueue("chunks", "documents", "first-ever.bin")
    queued_at = datetime.fromisoformat(request.created_at)

    summary = freshness_summary(
        tmp_path, "chunks", now=queued_at + timedelta(hours=2)
    )

    assert summary["stale_tail"] is True
    assert summary["newest_content_at"] is None
    assert summary["content_lag_s"] is None
