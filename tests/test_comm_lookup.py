"""Unit tests for the comm_lookup MCP tool (#0128 + verdict-gating follow-up).

comm_lookup wraps file_search with safe defaults and distills the hits into a
small verdict envelope. The verdict is *evidence-gated*: hybrid search always
returns a nearest neighbor, so "has a hit" is not a match — a hit must carry a
matching phone number or enough of the query's words. A comm rerank makes the
actual call/voicemail record beat a bare contact card. These tests isolate that
logic by patching _file_search_impl with canned compact payloads.
"""
import json

from unittest.mock import patch

import mcp_server

_ENVELOPE_KEYS = {
    "verdict", "top_hit", "key_facts", "source_ids", "snippet", "hits",
    "missing_exact_fields", "sql_needed", "note",
}

_FORBIDDEN_SUBSTRINGS = (
    "_node_content", "embedding", "vector", "custom_meta", "enr_entities",
    "relationships", "metadata_template",
)

_BUDGET = mcp_server._COMM_LOOKUP_OUTPUT_BUDGET


def _pretty_len(obj) -> int:
    """Length of the pretty (indent=2) JSON — what mcporter --output json prints."""
    return len(json.dumps(obj, indent=2))


def _message_hit(**over):
    """A seeded-message-style hit (msg-005): strong lexical body, no phone."""
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
        "enr_entities_people": "Erin Walsh",
        "vector": [0.1] * 32,
    }
    hit.update(over)
    return hit


def _contact_hit(**over):
    """A contact-card hit: only the person's name, no phone/call evidence."""
    hit = {
        "doc_id": "sor::contact/189",
        "source_type": "sor_contact",
        "score": 0.90,
        "title": "Aaron Curet",
        "snippet": "Contact name: Aaron Curet",
        "content": "Contact name: Aaron Curet",
        "source_message_id": "189",
        "enr_key_facts": '["Contact name: Aaron Curet"]',
    }
    hit.update(over)
    return hit


def _call_task_hit(**over):
    """A callback/voicemail task hit carrying the phone number + call terms."""
    hit = {
        "doc_id": "sor::task/1698",
        "source_type": "sor_task",
        "score": 0.70,
        "title": "Callback: +1 484-735-8527 missed calls + unclear voicemail",
        "snippet": "Callback: +1 484-735-8527 missed calls + unclear voicemail",
        "content": "Aaron Curet left an unclear voicemail after several missed calls. Callback +1 484-735-8527.",
        "source_message_id": "1698",
        "enr_key_facts": '["Callback +1 484-735-8527", "missed calls, unclear voicemail"]',
    }
    hit.update(over)
    return hit


def _patch_search(payload):
    return patch("mcp_server._file_search_impl", return_value=payload)


# --- happy path: strong, specific evidence -> found -------------------------

def test_found_on_strong_lexical_non_call_query():
    with _patch_search({"results": [_message_hit()]}):
        out = mcp_server._comm_lookup_impl("Erin Walsh periwinkle substation")
    assert set(out) >= _ENVELOPE_KEYS, out
    assert out["verdict"] == "found"
    assert out["top_hit"]["doc_id"] == "sor::message/msg-005"
    assert out["sql_needed"] is False
    assert out["source_ids"] == ["msg-005"]
    assert out["key_facts"] == [
        "Crew arrived at periwinkle substation", "Inspection begins at noon"]
    top = out["hits"][0]
    assert top["sender"] == "Erin Walsh" and top["source_id"] == "msg-005"


def test_output_has_no_raw_metadata_blobs():
    with _patch_search({"results": [_message_hit()]}):
        out = mcp_server._comm_lookup_impl("Erin Walsh periwinkle substation")
    blob = json.dumps(out)
    for bad in _FORBIDDEN_SUBSTRINGS:
        assert bad not in blob, f"{bad!r} leaked into comm_lookup output: {blob}"


# --- the live-reported false positives --------------------------------------

def test_nonsense_query_is_not_found_not_found_on_semantic_neighbor():
    """The exact live failure: a nonsense query semantically lands on some hit,
    but with zero lexical/phone evidence it must NOT read as found."""
    noise = _message_hit(snippet="It's not here.", content="It's not here.", sender="")
    with _patch_search({"results": [noise]}):
        out = mcp_server._comm_lookup_impl("zzzq nonexistent unobtanium xyzzy")
    assert out["verdict"] == "not_found", out
    assert out["sql_needed"] is True
    assert out["top_hit"] is None
    assert len(json.dumps(out)) <= _BUDGET


def test_contact_only_for_call_query_is_ambiguous_not_found():
    """Aaron callback case with only a contact card indexed: never `found`."""
    with _patch_search({"results": [_contact_hit()]}):
        out = mcp_server._comm_lookup_impl(
            "Callback 4847358527 missed calls unclear voicemail Aaron Curet")
    assert out["verdict"] == "ambiguous", out
    assert out["sql_needed"] is True
    assert out["missing_exact_fields"] == ["call/voicemail record"]
    assert out["top_hit"]["doc_id"] == "sor::contact/189"
    assert "not found" in out["note"].lower()


# --- comm rerank: the call record must beat the contact card ----------------

def test_call_task_outranks_contact_card():
    """Contact card is ranked higher by raw search, but the callback task has the
    phone + call terms — the rerank must surface the task as top_hit, `found`."""
    # contact first (higher search score), task second — rerank must flip them.
    hits = [_contact_hit(score=0.92), _call_task_hit(score=0.61)]
    with _patch_search({"results": hits}):
        out = mcp_server._comm_lookup_impl(
            "Callback 4847358527 missed calls unclear voicemail Aaron Curet")
    assert out["verdict"] == "found", out
    assert out["top_hit"]["doc_id"] == "sor::task/1698", out
    # both records' ids are still available for a targeted follow-up
    assert "1698" in out["source_ids"] and "189" in out["source_ids"]


def test_phone_match_normalizes_digit_forms():
    for q in ("4847358527", "+14847358527", "484-735-8527 callback"):
        with _patch_search({"results": [_call_task_hit()]}):
            out = mcp_server._comm_lookup_impl(q)
        assert out["verdict"] == "found", (q, out)
        assert out["top_hit"]["doc_id"] == "sor::task/1698", (q, out)


def test_partial_lexical_match_is_ambiguous_not_found():
    hit = _message_hit(snippet="budget report attached", content="budget report attached",
                       sender="", title="")
    with _patch_search({"results": [hit]}):
        out = mcp_server._comm_lookup_impl("quarterly zephyr marmalade budget report")
    assert out["verdict"] == "ambiguous", out
    assert out["sql_needed"] is False


# --- errors / edges ---------------------------------------------------------

def test_no_hits_is_small_not_found_with_sql_needed():
    with _patch_search({"results": []}):
        out = mcp_server._comm_lookup_impl("nobody at all zzz")
    assert out["verdict"] == "not_found"
    assert out["sql_needed"] is True
    assert out["top_hit"] is None
    assert len(json.dumps(out)) <= _BUDGET


def test_passthrough_error_stays_small_and_structured():
    err = {"error": True, "code": "empty_query", "message": "Query must not be empty.",
           "fix": "Provide a query."}
    with _patch_search(err):
        out = mcp_server._comm_lookup_impl("")
    assert out == err
    assert len(json.dumps(out)) <= _BUDGET


def test_service_error_becomes_small_not_found_not_a_dump():
    err = {"error": True, "code": "search_failed", "message": "Search operation failed: boom"}
    with _patch_search(err):
        out = mcp_server._comm_lookup_impl("anything")
    assert out["verdict"] == "not_found"
    assert out["sql_needed"] is True
    assert out.get("degraded") is True
    assert len(json.dumps(out)) <= _BUDGET


def test_degraded_results_flagged_but_still_found():
    with _patch_search({"results": [_message_hit()], "degraded": True}):
        out = mcp_server._comm_lookup_impl("Erin Walsh periwinkle substation")
    assert out["verdict"] == "found"
    assert out.get("degraded") is True


def test_output_stays_under_budget_with_huge_content():
    big = "periwinkle substation " + "x" * 6000
    hits = [
        _message_hit(doc_id=f"sor::message/msg-{i}", snippet=big, content=big,
                     enr_key_facts=json.dumps([big, big, big]))
        for i in range(3)
    ]
    with _patch_search({"results": hits}):
        out = mcp_server._comm_lookup_impl("periwinkle substation", limit=3)
    # pretty output (mcporter) is the ceiling that matters, not compact JSON
    assert _pretty_len(out) <= _BUDGET, _pretty_len(out)
    assert out["verdict"] in {"found", "ambiguous"}
    assert out["top_hit"] and out["source_ids"]


def test_oversized_limit_stays_within_pretty_budget():
    """Agent over-asks with limit=20 and fat hits — the PRETTY output mcporter
    prints must still hold <=3000 (the live-reported 3225-char regression)."""
    big = "Callback +1 484-735-8527 missed calls unclear voicemail " + "detail " * 90
    hits = [
        _call_task_hit(doc_id=f"sor::task/{i}", source_message_id=f"AC{i:034d}",
                       snippet=big, content=big,
                       enr_key_facts=json.dumps([big, big, big, big]))
        for i in range(20)
    ]
    with _patch_search({"results": hits}):
        out = mcp_server._comm_lookup_impl(
            "Callback 4847358527 missed calls unclear voicemail Aaron Curet", limit=20)
    assert _pretty_len(out) <= _BUDGET, _pretty_len(out)
    # core answer survives the trimming
    assert out["verdict"] == "found"
    assert out["top_hit"] and out["source_ids"]


def test_limit_is_clamped_to_exposed_max():
    """A huge limit is clamped to the exposed cap (not silently honored)."""
    hits = [_call_task_hit(doc_id=f"sor::task/{i}", source_message_id=f"id{i}")
            for i in range(20)]
    with _patch_search({"results": hits}):
        out = mcp_server._comm_lookup_impl("4847358527 callback voicemail", limit=20)
    assert len(out["hits"]) <= mcp_server._COMM_LOOKUP_MAX_LIMIT


def test_source_ids_fall_back_to_doc_id_when_no_source_message_id():
    hit = _message_hit()
    del hit["source_message_id"]
    with _patch_search({"results": [hit]}):
        out = mcp_server._comm_lookup_impl("Erin Walsh periwinkle substation")
    assert out["source_ids"] == ["sor::message/msg-005"]


def test_limit_clamped_and_pool_widened_for_rerank():
    with _patch_search({"results": []}) as m:
        mcp_server._comm_lookup_impl("q", limit=999)
    # the search pool is widened past `limit` so the rerank has candidates
    assert m.call_args.kwargs["top_k"] == mcp_server._COMM_LOOKUP_POOL
    assert m.call_args.kwargs["return_mode"] == "compact"


def test_key_facts_parsed_from_newline_prose():
    hit = _message_hit(enr_key_facts="Missed 3 calls\nVoicemail unclear\n")
    with _patch_search({"results": [hit]}):
        out = mcp_server._comm_lookup_impl("Erin Walsh periwinkle substation")
    assert out["key_facts"] == ["Missed 3 calls", "Voicemail unclear"]
