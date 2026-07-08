"""SOR (postgres comm-store) tool surface: schema, guarded queries, and the
postgres-source → index sweep path."""
import json

import pytest

pytestmark = pytest.mark.anyio

_COMM_LOOKUP_BUDGET = 3000


async def test_sor_schema_lists_messages_table(mcp_session):
    tables = await mcp_session.call_tool_json("sor_schema", {})
    assert isinstance(tables, str), tables
    assert "messages" in tables

    columns = await mcp_session.call_tool_json("sor_schema", {"table": "messages"})
    assert isinstance(columns, str), columns
    for col in ("body", "sender", "direction", "sent_at"):
        assert col in columns, f"{col!r} missing from schema:\n{columns}"


async def test_sor_query_select_returns_seeded_rows(mcp_session):
    out = await mcp_session.call_tool_json("sor_query", {
        "sql": "SELECT sender, direction, body FROM messages ORDER BY id",
        "limit": 50,
    })
    assert isinstance(out, str), out
    lines = [line for line in out.strip().splitlines() if line.strip()]
    # header + 5 seeded fixture rows
    data_rows = [line for line in lines if "\t" in line][1:]
    assert len(data_rows) == 5, out
    assert "zephyr" in out, out
    assert "Alice Nguyen" in out and "inbound" in out and "outbound" in out


async def test_sor_query_rejects_writes(mcp_session):
    for sql in (
        "UPDATE messages SET body = 'pwned'",
        "INSERT INTO messages (source, source_message_id, sent_at, updated_at) "
        "VALUES ('x', 'x', now(), now())",
    ):
        out = await mcp_session.call_tool_json("sor_query", {"sql": sql})
        assert isinstance(out, str), out
        assert "read-only" in out, f"write not rejected: {out!r}"
        assert "SELECT" in out  # the error must steer toward SELECT/WITH


async def test_sor_sweep_indexed_messages_searchable(indexed_corpus, mcp_session):
    # The full sweep in indexed_corpus also indexes the postgres "sor" source;
    # 'periwinkle' appears only in seeded message msg-005.
    payload = await mcp_session.call_tool_json(
        "file_search", {"query": "periwinkle substation", "top_k": 8})
    assert not payload.get("error"), payload
    hits = [r for r in payload["results"] if r.get("source_type") == "pg_message"]
    assert hits, f"no pg_message hit for periwinkle: {payload['results']}"
    top = hits[0]
    assert "periwinkle" in (top.get("snippet") or "").lower(), top
    assert top.get("direction") == "inbound", top
    assert top.get("sender") == "Erin Walsh", top


async def test_comm_lookup_finds_seeded_message_compactly(indexed_corpus, mcp_session):
    """comm_lookup returns a compact verdict envelope (not a raw dump) for a
    person/comm query — the safe path Hermes should use instead of SQL (#0128)."""
    payload = await mcp_session.call_tool_json(
        "comm_lookup", {"query": "periwinkle substation inspection", "limit": 3})
    assert isinstance(payload, dict) and not payload.get("error"), payload

    assert payload["verdict"] in ("found", "ambiguous"), payload
    assert payload["top_hit"]["source_type"] == "pg_message", payload
    # source ids are returned so a follow-up exact query can be targeted
    assert payload["source_ids"], payload
    assert payload["sql_needed"] is False, payload

    blob = json.dumps(payload)
    # compact: whole response well under the 3k budget, no raw metadata blobs
    assert len(blob) <= _COMM_LOOKUP_BUDGET, f"comm_lookup output {len(blob)} chars: {blob}"
    for bad in ("_node_content", "embedding", "vector", "custom_meta"):
        assert bad not in blob, f"{bad!r} leaked into comm_lookup output"


async def test_comm_lookup_no_hit_is_small_not_found(indexed_corpus, mcp_session):
    """A no-match query yields a small not_found/ambiguous response with a SQL
    hint — never a stack trace or a giant dump."""
    payload = await mcp_session.call_tool_json(
        "comm_lookup", {"query": "zzzq nonexistent unobtanium xyzzy", "limit": 3})
    assert isinstance(payload, dict) and not payload.get("error"), payload
    assert payload["verdict"] in ("not_found", "ambiguous"), payload
    assert len(json.dumps(payload)) <= _COMM_LOOKUP_BUDGET, payload
    if payload["verdict"] == "not_found":
        assert payload["sql_needed"] is True, payload
