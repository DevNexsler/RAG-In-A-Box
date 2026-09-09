"""Call discovery is opt-in; exact-event retrieval still owns transcripts."""
import datetime as dt

import pytest

from cds_history import fetch_conversation, history_request
import mcp_server as srv


def test_call_reference_pages_are_typed_and_cursor_bound():
    now = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)
    class Cursor:
        def execute(self, sql, params):
            self.sql, self.params = sql, params

        def fetchall(self):
            return [(2, "twilio", "AC" + "2" * 32, now, None, None, None, "", False),
                    (1, "twilio", "AC" + "1" * 32, now, None, None, None, "", False)]

    cursor = Cursor()
    contact = {"phone_e164": "+12025550123", "history": {"kind": "calls", "limit": 1}}
    result = fetch_conversation(cursor, contact)
    assert "FROM calls" in cursor.sql
    assert "+12025550123" not in cursor.sql
    assert result["messages"][0]["id"] == "call:2"
    assert result["messages"][0]["event_kind"] == "call_reference"
    assert result["has_more"]
    with pytest.raises(ValueError, match="history_cursor"):
        fetch_conversation(None, {"phone_e164": contact["phone_e164"],
                                  "history": {"cursor": result["next_cursor"]}})


def test_history_kind_rejects_unknown_and_preserves_default_contract():
    assert history_request() == {"since": None, "limit": 50, "cursor": None}
    with pytest.raises(ValueError, match="history_kind"):
        history_request(kind="anything")


@pytest.mark.anyio
async def test_public_tool_forwards_opt_in_call_history(monkeypatch):
    if not srv.HAS_MCP:
        pytest.skip("mcp package not installed")
    captured = {}
    def impl(**kwargs):
        captured.update(kwargs)
        return {"ok": True}
    monkeypatch.setattr(srv, "_context_builder_impl", impl)
    assert await srv.context_builder(phone="2025550123", history_kind="calls") == {"ok": True}
    assert captured["history_kind"] == "calls"


def test_exact_event_mode_rejects_call_history_option():
    result = srv._context_builder_impl(event_refs=["AC" + "1" * 32], history_kind="calls")
    assert "cannot be mixed" in result["error"]
