import datetime as dt

import pytest

from cds_history import fetch_conversation, history_request
from context_builder import derive_flags
import mcp_server as srv


@pytest.mark.parametrize("limit", [0, 101, True, "50"])
def test_invalid_limit_is_rejected_before_sources(limit, monkeypatch):
    monkeypatch.setattr(srv, "_ctx_cds_source", lambda *a: pytest.fail("must not query"))
    assert "history_limit" in srv._context_builder_impl(phone="2025550123", history_limit=limit)["error"]


@pytest.mark.parametrize("since", ["not-a-date", "2026-06-15", "2099-01-01T00:00:00Z"])
def test_invalid_history_time_is_rejected(since):
    with pytest.raises(ValueError):
        history_request(since)


def test_name_only_cannot_claim_conversation_coverage():
    result = fetch_conversation(None, {"name": "Test Prospect"})
    assert result["status"] == "no_identifiers"
    assert result["coverage_complete"] is False


@pytest.mark.anyio
async def test_public_tool_forwards_history_options(monkeypatch):
    if not srv.HAS_MCP:
        pytest.skip("mcp package not installed")
    captured = {}
    def impl(**kwargs):
        captured.update(kwargs)
        return {"ok": True}
    monkeypatch.setattr(srv, "_context_builder_impl", impl)
    result = await srv.context_builder(phone="2025550123", include=["cds"],
        history_since="2026-01-01T00:00:00Z", history_limit=7, history_cursor="opaque")
    assert result == {"ok": True}
    assert captured["history_since"] == "2026-01-01T00:00:00Z"
    assert captured["history_limit"] == 7
    assert captured["history_cursor"] == "opaque"


@pytest.mark.parametrize("cursor", ["!", "e30=", "bnVsbA==", "W10="])
def test_malformed_cursors_fail_before_history_query(cursor):
    with pytest.raises(ValueError, match="history_cursor"):
        fetch_conversation(None, {"phone_e164": "+12025550123", "history": {"cursor": cursor}})


def test_caller_stale_timestamp_does_not_override_newer_cds_inbound():
    result = derive_flags({"latest_inbound_at": "2026-06-19T00:00:00Z"}, {
        "status": "ok", "latest_inbound_at": "2026-06-25T00:00:00Z",
        "outbound_evidence": [{"at": "2026-06-20T00:00:00Z"}]})
    assert result["our_outbound_after_latest_inbound"] is False


def test_cursor_cannot_switch_identity_or_window():
    now = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=90)
    class Cursor:
        def execute(self, sql, params):
            self.media = "message_media" in sql
        def fetchall(self):
            return [] if self.media else [(2, "quo", "two", now, "inbound", None, None, "two", False),
                                          (1, "quo", "one", now, "inbound", None, None, "one", False)]
    first = fetch_conversation(Cursor(), {"phone_e164": "+12025550123", "history": {"limit": 1}})
    assert first["has_more"] and not first["coverage_complete"]
    with pytest.raises(ValueError, match="history_cursor"):
        fetch_conversation(Cursor(), {"phone_e164": "+12025550124", "history": {"cursor": first["next_cursor"]}})
    with pytest.raises(ValueError, match="history_cursor"):
        fetch_conversation(Cursor(), {"phone_e164": "+12025550123", "history": {"since": now.isoformat(), "cursor": first["next_cursor"]}})


def test_truncated_body_blocks_complete_coverage():
    class Cursor:
        def execute(self, sql, params):
            self.media = "message_media" in sql
        def fetchall(self):
            return [] if self.media else [
                (1, "quo", "one", dt.datetime.now(dt.timezone.utc), "inbound", None, None, "x" * 4000, True)]
    result = fetch_conversation(Cursor(), {"phone_e164": "+12025550123"})
    assert result["status"] == "degraded"
    assert result["window_exhausted"] is True
    assert result["coverage_complete"] is False


class _Cursor:
    """Answers each query with the next canned result, recording the SQL."""

    def __init__(self, *results):
        self.results, self.queries = list(results), []

    def execute(self, sql, params):
        self.queries.append((sql, params))

    def fetchall(self):
        return self.results.pop(0)


MARKER = ("[Message body is 2 inline images and no text. Their content is in this "
          "message's media attachments.]")


def test_blank_body_with_media_is_flagged_not_extracted_not_empty():
    # CDS 839634: a letter pasted into an email as two images stored body "\n".
    # The chronology showed it as an empty email and the reviewer read it so.
    now = dt.datetime.now(dt.timezone.utc)
    cur = _Cursor(
        [(3, "zoho_mail", "letter", now, "inbound", None, "To whom", "\n", False),
         (2, "quo", "photo", now, "inbound", None, None, "", False),
         (1, "quo", "text", now, "inbound", None, None, "Rent sent", False)],
        [(3, 2, None, False)],
    )
    result = fetch_conversation(cur, {"phone_e164": "+12025550123"})
    letter, photo, text = result["messages"]
    assert letter["content_status"] == "not_extracted"
    assert letter["media_count"] == 2
    assert letter["body"] == "\n"
    assert "media_text" not in letter
    # Blank with no media really is empty; text never takes the flag.
    assert "content_status" not in photo and "media_count" not in photo
    assert "content_status" not in text
    media_sql, media_params = cur.queries[1]
    assert "message_media" in media_sql and "enrichment" in media_sql
    assert media_params[-1] == [3, 2, 1]


def test_media_only_body_carries_its_extracted_text():
    """r6 review: after the AES backfill 839634's body is the inline-image marker,
    not blank, and its letter text sits in message_media OCR the chronology never
    read. A media-only body (blank or the marker) now carries that text, and says
    whether it was extracted; 548 of 550 blank-body Quo photos already had it."""
    now = dt.datetime.now(dt.timezone.utc)
    letter_text = "Heat not fixed by October 15; city inspectors; 12 months of Elizabethtown Gas bills."
    cur = _Cursor(
        [(4, "zoho_mail", "letter", now, "inbound", None, "Heat", MARKER, False),
         (3, "quo", "photo", now, "inbound", None, None, "", False),
         (2, "quo", "pending", now, "inbound", None, None, "", False)],
        [(4, 2, letter_text, False), (3, 1, "A stairwell under renovation.", False), (2, 1, None, False)],
    )
    letter, photo, pending = fetch_conversation(cur, {"phone_e164": "+12025550123"})["messages"]
    assert letter["body"] == MARKER
    assert (letter["content_status"], letter["media_count"], letter["media_text"]) == ("extracted", 2, letter_text)
    assert letter["media_text_truncated"] is False
    assert (photo["content_status"], photo["media_text"]) == ("extracted", "A stairwell under renovation.")
    assert pending["content_status"] == "not_extracted" and "media_text" not in pending


def test_text_with_media_says_it_has_media_and_its_text():
    """r6 review: 703071 'Still there.' plus a 4000x3000 photo showed only the text."""
    now = dt.datetime.now(dt.timezone.utc)
    cur = _Cursor([(1, "zoho_mail", "still", now, "inbound", None, None, "Still there.", False)],
                  [(1, 1, "Photo of a cold radiator.", True)])
    message = fetch_conversation(cur, {"phone_e164": "+12025550123"})["messages"][0]
    assert message["media_count"] == 1 and message["media_text"] == "Photo of a cold radiator."
    assert message["media_text_truncated"] is True
    assert "content_status" not in message


def test_media_lookup_is_one_query_for_the_page():
    now = dt.datetime.now(dt.timezone.utc)
    cur = _Cursor([(1, "quo", "text", now, "inbound", None, None, "Rent sent", False)], [])
    fetch_conversation(cur, {"phone_e164": "+12025550123"})
    assert len(cur.queries) == 2


def test_calls_are_not_looked_up_for_media():
    now = dt.datetime.now(dt.timezone.utc)
    cur = _Cursor([(1, "quo", "call", now, "inbound", None, None, "", False)])
    fetch_conversation(cur, {"phone_e164": "+12025550123", "history": {"kind": "calls"}})
    assert len(cur.queries) == 1
