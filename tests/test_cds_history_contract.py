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
            pass
        def fetchall(self):
            return [(2, "quo", "two", now, "inbound", None, None, "two", False),
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
            pass
        def fetchall(self):
            return [(1, "quo", "one", dt.datetime.now(dt.timezone.utc), "inbound", None, None, "x" * 4000, True)]
    result = fetch_conversation(Cursor(), {"phone_e164": "+12025550123"})
    assert result["status"] == "degraded"
    assert result["window_exhausted"] is True
    assert result["coverage_complete"] is False
