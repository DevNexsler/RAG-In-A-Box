"""context_builder e2e: full dossier shape + degrade-loud source statuses in
the hermetic staging stack, plus the invalid-input error path.

Staging has no factbook-rpc (FACTBOOK_RPC_TOKEN/URL are never passed to
doc-organizer-staging in docker-compose.staging.yml), and the comm-postgres
fixture (staging/comm_postgres/init.sql) is the flattened SOR-style schema
used by sor_query/file_search — it does NOT carry the CDS live-schema tables
cds_live.py queries (message_participants, participants, outbound_actions,
raw_events). So both `factbook` and `cds` deterministically degrade to
`status: "error:..."` here; the tool must still answer with the full dossier
shape instead of raising. That degrade contract is exactly what this test
asserts (see context_builder.py / factbook_client.py / cds_live.py
docstrings for the per-source contract)."""
import pytest

pytestmark = pytest.mark.anyio


async def test_context_builder_returns_dossier(mcp_session):
    out = await mcp_session.call_tool_json(
        "context_builder", {"email": "nobody@example.com"})
    assert isinstance(out, dict) and not out.get("error"), out
    assert out["contact"]["email"] == "nobody@example.com"
    for key in ("contact", "factbook", "cds", "comm_context", "derived", "elapsed_ms"):
        assert key in out, out

    # factbook-rpc is not deployed to staging: FACTBOOK_RPC_TOKEN is unset,
    # so factbook_source degrades before ever making a call.
    assert out["factbook"]["status"].startswith("error:"), out["factbook"]
    assert out["factbook"]["entities"] == []
    assert out["factbook"]["flags"] == {}

    # comm-postgres in staging is the flattened SOR fixture schema, not the
    # CDS live schema cds_live.py's SQL targets — every query raises and
    # cds_source degrades loud instead of propagating the exception.
    assert out["cds"]["status"].startswith("error:"), out["cds"]
    assert out["cds"]["outbound_evidence"] == []

    # comm_context is backed by the doc-organizer index (independent of
    # FactBook/CDS); nobody@example.com matches none of the seeded fixtures
    # exactly, so it comes back well-formed with no exact hits.
    assert out["comm_context"]["status"] in ("no_exact_hit", "ok"), out["comm_context"]
    assert isinstance(out["comm_context"]["hits"], list)

    # Both sources errored and no latest_inbound_at was supplied, so the
    # derived flag can't be computed either way.
    assert out["derived"]["our_outbound_after_latest_inbound"] == "unknown"
    assert isinstance(out["elapsed_ms"], int) and 0 <= out["elapsed_ms"] < 10_000


async def test_context_builder_requires_identifier(mcp_session):
    out = await mcp_session.call_tool_json("context_builder", {})
    assert "identifier" in out.get("error", ""), out


async def test_context_builder_exact_event_mode_validates_public_arguments(mcp_session):
    out = await mcp_session.call_tool_json('context_builder', {'event_refs': []})
    assert 'event_refs' in out.get('error', ''), out
    mixed = await mcp_session.call_tool_json('context_builder', {'event_refs': ['one'], 'phone': '2025550123'})
    assert 'cannot be mixed' in mixed.get('error', ''), mixed
