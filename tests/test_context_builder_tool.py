"""Unit tests for the context_builder MCP tool (task B4).

_context_builder_impl orchestrates the three injected sources via
context_builder.build_context; _ctx_comm_source is the `comm` dep, wrapping
comm_lookup with an exact-hit filter (comm_context hits are semantic context
only -- never proof of handling). _ctx_factbook_source/_ctx_cds_source are
thin module-level aliases so tests (and this file) can monkeypatch them on
the `mcp_server` module directly.
"""
import pytest

import mcp_server as srv


def test_impl_happy_path(monkeypatch):
    monkeypatch.setattr(srv, "_ctx_factbook_source", lambda c: {"status": "no_match", "entities": [], "flags": {}})
    monkeypatch.setattr(srv, "_ctx_cds_source", lambda c: {"status": "ok", "inbound_count_30d": 0,
        "latest_inbound_at": None, "latest_outbound_at": None, "outbound_evidence": []})
    monkeypatch.setattr(srv, "_ctx_comm_source", lambda c: {"status": "no_exact_hit", "hits": []})
    out = srv._context_builder_impl(email="a@b.com")
    assert out["derived"]["our_outbound_after_latest_inbound"] == "unknown"
    assert out["contact"]["email"] == "a@b.com"


def test_impl_invalid_input_is_error_dict():
    out = srv._context_builder_impl()
    assert out.get("error") and "identifier" in out["error"]


def test_comm_source_filters(monkeypatch):
    monkeypatch.setattr(srv, "_comm_lookup_impl", lambda **kw: {"hits": [
        {"sender": "Leslie H", "channel": "+16107095575",
         "snippet": "415leslie@gmail.com", "source_id": "AC1"}]})
    out = srv._ctx_comm_source({"email": "jessbrown816@gmail.com",
                                "phone_e164": "+14847614094",
                                "name": "Jessica Ann Brown"})
    assert out == {"status": "no_exact_hit", "hits": []}


def test_comm_source_dedupes_across_identifiers_and_sorts_newest_first(monkeypatch):
    contact = {"email": "jess@example.com", "phone_e164": "+14847614094", "name": None}
    hit_old = {"sender": "Jess", "channel": "+14847614094", "snippet": "old",
               "source_id": "AC1", "sent_at": "2026-08-01T00:00:00Z"}
    hit_new = {"sender": "Jess", "channel": "jess@example.com", "snippet": "new",
               "source_id": "AC2", "sent_at": "2026-08-20T00:00:00Z"}

    def fake_lookup(**kw):
        # Both identifier queries return the same overlapping pair of hits;
        # the duplicate (by source_id) must collapse to one entry.
        return {"hits": [hit_old, hit_new]}

    monkeypatch.setattr(srv, "_comm_lookup_impl", fake_lookup)
    out = srv._ctx_comm_source(contact)
    assert out["status"] == "ok"
    assert [h["source_id"] for h in out["hits"]] == ["AC2", "AC1"]


def test_comm_source_caps_at_ten_hits(monkeypatch):
    contact = {"email": "jess@example.com", "phone_e164": None, "name": None}
    hits = [
        {"sender": "Jess", "channel": "jess@example.com", "snippet": f"msg {i}",
         "source_id": f"AC{i}", "sent_at": f"2026-08-{i + 1:02d}T00:00:00Z"}
        for i in range(15)
    ]

    monkeypatch.setattr(srv, "_comm_lookup_impl", lambda **kw: {"hits": hits})
    out = srv._ctx_comm_source(contact)
    assert len(out["hits"]) == 10
    # newest first
    assert out["hits"][0]["source_id"] == "AC14"


def test_impl_include_skips_excluded_sources(monkeypatch):
    called = {"factbook": False, "cds": False}

    def factbook_should_not_run(c):
        called["factbook"] = True
        return {"status": "ok", "entities": [], "flags": {}}

    def cds_should_not_run(c):
        called["cds"] = True
        return {"status": "ok", "inbound_count_30d": 0, "latest_inbound_at": None,
                "latest_outbound_at": None, "outbound_evidence": []}

    monkeypatch.setattr(srv, "_ctx_factbook_source", factbook_should_not_run)
    monkeypatch.setattr(srv, "_ctx_cds_source", cds_should_not_run)
    monkeypatch.setattr(srv, "_ctx_comm_source", lambda c: {"status": "no_exact_hit", "hits": []})

    out = srv._context_builder_impl(email="a@b.com", include=["comm"])

    assert called == {"factbook": False, "cds": False}
    assert out["factbook"]["status"] == "skipped"
    assert out["factbook"]["entities"] == []
    assert out["cds"]["status"] == "skipped"
    assert out["cds"]["outbound_evidence"] == []
    assert out["comm_context"]["status"] == "no_exact_hit"


@pytest.mark.anyio
async def test_tool_registered_and_delegates(monkeypatch):
    if not srv.HAS_MCP:
        pytest.skip("mcp package not installed")

    captured = {}

    def fake_impl(**kw):
        captured.update(kw)
        return {"ok": True}

    monkeypatch.setattr(srv, "_context_builder_impl", fake_impl)
    assert hasattr(srv, "context_builder")
    # Registered tools are awaitable: a sync body is dispatched to a worker
    # thread (mcp_server.py's _traced_mcp_tool offload wrapper).
    out = await srv.context_builder(
        email="a@b.com", phone="4847614094", name="Jess",
        lead_id="123", latest_inbound_at="2026-08-27T00:00:00Z",
        include=["cds"],
    )
    assert out == {"ok": True}
    assert captured == {
        "email": "a@b.com", "phone": "4847614094", "name": "Jess",
        "lead_id": "123", "latest_inbound_at": "2026-08-27T00:00:00Z",
        "include": ["cds"],
    }
