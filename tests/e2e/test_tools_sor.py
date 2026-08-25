"""SOR (postgres comm-store) tool surface: schema, guarded queries, and the
postgres-source → index sweep path."""
import json
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.anyio

_COMM_LOOKUP_BUDGET = 3000
_ROOT = Path(__file__).resolve().parents[2]
_COMPOSE_FILE = _ROOT / "docker-compose.staging.yml"


def _raw_lance_row(doc_id: str) -> dict:
    script = (
        "import json,sys; from lancedb_store import LanceDBStore; "
        "store=LanceDBStore('/data/index','chunks'); "
        "rows=store._vs.table.search(None).where("
        "f\"doc_id = '{sys.argv[1]}'\",prefilter=True)"
        ".select(['doc_id','text','metadata']).limit(1).to_list(); "
        "print(json.dumps(rows[0] if rows else {},default=str))"
    )
    completed = subprocess.run(
        [
            "docker", "compose", "-f", str(_COMPOSE_FILE), "exec", "-T",
            "doc-organizer-staging", "python", "-c", script, doc_id,
        ],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


async def test_sor_schema_lists_messages_table(mcp_session):
    tables = await mcp_session.call_tool_json("sor_schema", {})
    assert isinstance(tables, str), tables
    assert "messages" in tables

    columns = await mcp_session.call_tool_json("sor_schema", {"table": "messages"})
    assert isinstance(columns, str), columns
    for col in ("body", "subject", "sender", "direction", "sent_at"):
        assert col in columns, f"{col!r} missing from schema:\n{columns}"


async def test_sor_query_select_returns_seeded_rows(mcp_session):
    out = await mcp_session.call_tool_json("sor_query", {
        "sql": "SELECT sender, direction, body FROM messages ORDER BY id",
        "limit": 50,
    })
    assert isinstance(out, str), out
    lines = [line for line in out.strip().splitlines() if line.strip()]
    # header + 6 seeded fixture rows
    data_rows = [line for line in lines if "\t" in line][1:]
    assert len(data_rows) == 6, out
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

    subject_payload = await mcp_session.call_tool_json(
        "file_search", {"query": "cobalt courthouse filing", "top_k": 8}
    )
    assert not subject_payload.get("error"), subject_payload
    subject_hits = [
        result
        for result in subject_payload["results"]
        if result.get("doc_id") == "sor::email/msg-006"
    ]
    assert subject_hits, subject_payload["results"]
    assert "cobalt courthouse filing" in subject_hits[0]["snippet"].lower()


def test_sor_unit_title_reaches_raw_lance_metadata_and_chunk_header(indexed_corpus):
    expected_titles = {
        "sor::unit/104": "South Main Apartments Unit 5",
        "sor::unit/105": "125 S 13TH STREET LLC Unit B",
    }

    for doc_id, expected_title in expected_titles.items():
        row = _raw_lance_row(doc_id)
        assert row, f"no Lance row for {doc_id}"
        assert row["metadata"]["title"] == expected_title
        assert f"[Document: {expected_title}" in row["text"]


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
