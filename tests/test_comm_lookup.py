"""Unit tests for the comm_lookup MCP tool (#0128).

comm_lookup wraps file_search with safe defaults and distills the hits into a
small verdict envelope so agents get a compact, info-rich answer for
comm/person/phone/call/voicemail questions instead of falling back to raw
Comm-Data-Store SQL dumps. These tests isolate the distillation logic by
patching _file_search_impl with canned compact payloads.
"""
import json

from unittest.mock import patch

import mcp_server

# The full set of spec fields every non-error envelope must carry.
_ENVELOPE_KEYS = {
    "verdict",
    "top_hit",
    "key_facts",
    "source_ids",
    "snippet",
    "hits",
    "missing_exact_fields",
    "sql_needed",
    "note",
}

# Keys that must NEVER leak into the compact output (raw metadata/blobs).
_FORBIDDEN_SUBSTRINGS = (
    "_node_content",
    "embedding",
    "vector",
    "custom_meta",
    "enr_entities",
    "relationships",
    "metadata_template",
)


def _compact_hit(**over):
    """A representative compact file_search hit (shape of _compact_hit_to_dict),
    including noisy fields that must be stripped from comm_lookup output."""
    hit = {
        "doc_id": "sor::message/msg-005",
        "source_type": "pg_message",
        "score": 0.91,
        "title": "",
        "snippet": "Crew reached the periwinkle substation, inspection starts at noon.",
        "content": "Crew reached the periwinkle substation, inspection starts at noon.",
        "sender": "Erin Walsh",
        "direction": "inbound",
        "sent_at": "2026-06-01T10:04:00Z",
        "channel_name": "field",
        "source_message_id": "msg-005",
        "enr_key_facts": '["Crew arrived at periwinkle substation", "Inspection begins at noon"]',
        # Noise that must be dropped:
        "custom_meta": "{\"a\": 1}",
        "enr_entities_people": "Erin Walsh, Bob Ramirez",
        "vector": [0.1] * 32,
    }
    hit.update(over)
    return hit


def _patch_search(payload):
    return patch("mcp_server._file_search_impl", return_value=payload)


def test_found_returns_full_compact_envelope():
    with _patch_search({"results": [_compact_hit()]}):
        out = mcp_server._comm_lookup_impl("Erin Walsh periwinkle voicemail")

    assert set(out) >= _ENVELOPE_KEYS, out
    assert out["verdict"] == "found"
    assert out["top_hit"]["doc_id"] == "sor::message/msg-005"
    assert out["top_hit"]["source_type"] == "pg_message"
    assert out["sql_needed"] is False
    # source_id preferred over doc_id
    assert out["source_ids"] == ["msg-005"]
    # key_facts parsed from the JSON array
    assert out["key_facts"] == [
        "Crew arrived at periwinkle substation",
        "Inspection begins at noon",
    ]
    # comm fields flattened into the hit
    top = out["hits"][0]
    assert top["sender"] == "Erin Walsh"
    assert top["direction"] == "inbound"
    assert top["source_id"] == "msg-005"


def test_output_has_no_raw_metadata_blobs():
    with _patch_search({"results": [_compact_hit()]}):
        out = mcp_server._comm_lookup_impl("periwinkle")
    blob = json.dumps(out)
    for bad in _FORBIDDEN_SUBSTRINGS:
        assert bad not in blob, f"{bad!r} leaked into comm_lookup output: {blob}"


def test_no_hits_is_small_not_found_with_sql_needed():
    with _patch_search({"results": []}):
        out = mcp_server._comm_lookup_impl("nobody at all zzz")
    assert out["verdict"] == "not_found"
    assert out["sql_needed"] is True
    assert out["top_hit"] is None
    assert out["source_ids"] == []
    assert len(json.dumps(out)) <= mcp_server._COMM_LOOKUP_OUTPUT_BUDGET


def test_close_scores_are_ambiguous():
    hits = [_compact_hit(score=0.90), _compact_hit(doc_id="sor::message/msg-006", score=0.89)]
    with _patch_search({"results": hits}):
        out = mcp_server._comm_lookup_impl("periwinkle")
    assert out["verdict"] == "ambiguous"


def test_clear_score_gap_is_found():
    hits = [_compact_hit(score=0.90), _compact_hit(doc_id="sor::message/msg-006", score=0.40)]
    with _patch_search({"results": hits}):
        out = mcp_server._comm_lookup_impl("periwinkle")
    assert out["verdict"] == "found"


def test_passthrough_error_stays_small_and_structured():
    err = {"error": True, "code": "empty_query", "message": "Query must not be empty.",
           "fix": "Provide a query."}
    with _patch_search(err):
        out = mcp_server._comm_lookup_impl("")
    assert out == err  # passed straight through, still tiny
    assert len(json.dumps(out)) <= mcp_server._COMM_LOOKUP_OUTPUT_BUDGET


def test_service_error_becomes_small_not_found_not_a_dump():
    err = {"error": True, "code": "search_failed", "message": "Search operation failed: boom"}
    with _patch_search(err):
        out = mcp_server._comm_lookup_impl("anything")
    assert out["verdict"] == "not_found"
    assert out["sql_needed"] is True
    assert out.get("degraded") is True
    assert len(json.dumps(out)) <= mcp_server._COMM_LOOKUP_OUTPUT_BUDGET


def test_degraded_results_flagged_but_still_found():
    with _patch_search({"results": [_compact_hit()], "degraded": True}):
        out = mcp_server._comm_lookup_impl("periwinkle")
    assert out["verdict"] == "found"
    assert out.get("degraded") is True


def test_output_stays_under_budget_with_huge_content():
    big = "x" * 6000
    hits = [
        _compact_hit(doc_id=f"sor::message/msg-{i}", snippet=big, content=big,
                     enr_key_facts=json.dumps([big, big, big]))
        for i in range(3)
    ]
    with _patch_search({"results": hits}):
        out = mcp_server._comm_lookup_impl("periwinkle", limit=3)
    assert len(json.dumps(out)) <= mcp_server._COMM_LOOKUP_OUTPUT_BUDGET
    # core fields survive the trimming
    assert out["verdict"] in {"found", "ambiguous"}
    assert out["top_hit"] and out["source_ids"]


def test_source_ids_fall_back_to_doc_id_when_no_source_message_id():
    hit = _compact_hit()
    del hit["source_message_id"]
    with _patch_search({"results": [hit]}):
        out = mcp_server._comm_lookup_impl("periwinkle")
    assert out["source_ids"] == ["sor::message/msg-005"]


def test_limit_is_clamped_and_forwarded():
    with _patch_search({"results": []}) as m:
        mcp_server._comm_lookup_impl("q", limit=999)
    assert m.call_args.kwargs["top_k"] == mcp_server._COMM_LOOKUP_MAX_LIMIT
    assert m.call_args.kwargs["return_mode"] == "compact"


def test_key_facts_parsed_from_newline_prose():
    hit = _compact_hit(enr_key_facts="Missed 3 calls\nVoicemail unclear\n")
    with _patch_search({"results": [hit]}):
        out = mcp_server._comm_lookup_impl("callback")
    assert out["key_facts"] == ["Missed 3 calls", "Voicemail unclear"]
