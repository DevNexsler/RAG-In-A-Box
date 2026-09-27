"""Consumer contract over HTTP: authoritative IDs never fall back to fuzzy identity."""

import json

import pytest

from factbook_client import factbook_source


@pytest.mark.parametrize("blocked_flag", [
    "ambiguous", "agency_identifier", "shared_channel_identifier", "cardinality_blocked_identifier",
])
def test_contradictory_identity_flags_cannot_bind_person(http_peer, monkeypatch, blocked_flag):
    monkeypatch.setenv("FACTBOOK_RPC_URL", http_peer.url)
    monkeypatch.setenv("FACTBOOK_RPC_TOKEN", "contract-token")

    def response(request):
        return 200, {
            "jsonrpc": "2.0", "id": request["body"]["id"],
            "result": {"content": [{"type": "text", "text": json.dumps({
                "resolved": True, blocked_flag: True,
                "entities": [{"uuid": "person-1", "name": "Alex"}],
            })}], "isError": False},
        }

    http_peer.respond = response
    result = factbook_source({
        "platform_id": "zoho_cliq:user:AbC123", "name": "Alex",
        "email": "unrelated@example.test",
    })
    assert result["status"] == "ambiguous"
    assert result["entities"] == []
    assert len(http_peer.requests) == 1, "authoritative ID must not fall back to another identity"


def _envelope(request, payload, *, is_error=False):
    return {
        "jsonrpc": "2.0", "id": request["body"]["id"],
        "result": {"content": [{"type": "text", "text": json.dumps(payload)}], "isError": is_error},
    }


@pytest.fixture
def identity_peer(http_peer, monkeypatch):
    monkeypatch.setenv("FACTBOOK_RPC_URL", http_peer.url)
    monkeypatch.setenv("FACTBOOK_RPC_TOKEN", "contract-token")
    return http_peer


def test_qualified_identity_wire_contract_preserves_case_and_ambiguity_window(identity_peer):
    identity_peer.respond = lambda request: (200, _envelope(request, {
        "resolved": True, "ambiguous": False,
        "entities": [{"uuid": "person-1", "name": "Alex"}],
    }))
    result = factbook_source({"platform_id": "zoho_cliq:user:AbC123"})
    assert result["status"] == "ok"
    assert result["entities"] == [{"uuid": "person-1", "name": "Alex"}]
    [request] = identity_peer.requests
    assert request["path"] == "/mcp"
    assert request["headers"]["Authorization"] == "Bearer contract-token"
    assert request["body"]["jsonrpc"] == "2.0"
    assert request["body"]["method"] == "tools/call"
    assert request["body"]["params"] == {
        "name": "find_entity_by_attribute",
        "arguments": {"attribute_key": "platform_id", "attribute_value": "zoho_cliq:user:AbC123",
                      "entity_type": "Person", "num_results": 2},
    }


@pytest.mark.parametrize("failure", ["unauthorized", "rpc_error", "tool_error", "malformed_json", "missing_content"])
def test_failed_authoritative_lookup_never_falls_back(identity_peer, failure):
    def response(request):
        if failure == "unauthorized":
            return 401, {"error": "unauthorized"}
        if failure == "rpc_error":
            return 200, {"jsonrpc": "2.0", "id": request["body"]["id"], "error": {"message": "unavailable"}}
        if failure == "tool_error":
            return 200, _envelope(request, {"error": "query failed"}, is_error=True)
        if failure == "malformed_json":
            return 200, b"not-json"
        return 200, {"result": {"content": []}}

    identity_peer.respond = response
    result = factbook_source({"platform_id": "zoho_cliq:user:123", "name": "Alex", "email": "other@example.test"})
    assert result["status"].startswith("error:platform_id:")
    assert result["entities"] == []
    assert len(identity_peer.requests) == 1


def test_transient_http_failure_retries_same_rpc_then_resolves(identity_peer):
    def response(request):
        if len(identity_peer.requests) == 1:
            return 503, {"error": "restarting"}
        return 200, _envelope(request, {"resolved": True, "entities": [{"uuid": "exact-person"}]})

    identity_peer.respond = response
    assert factbook_source({"platform_id": "zoho_cliq:user:123"})["entities"] == [{"uuid": "exact-person"}]
    assert len(identity_peer.requests) == 2
    assert identity_peer.requests[0]["body"] == identity_peer.requests[1]["body"]


def test_response_for_another_rpc_cannot_bind_person(identity_peer):
    def response(request):
        envelope = _envelope(request, {"resolved": True, "entities": [{"uuid": "wrong-person"}]})
        envelope["id"] = "another-request"
        return 200, envelope

    identity_peer.respond = response
    result = factbook_source({"platform_id": "zoho_cliq:user:123", "email": "other@example.test"})
    assert result["status"].startswith("error:platform_id:")
    assert result["entities"] == []
    assert len(identity_peer.requests) == 1


@pytest.mark.parametrize("entities", [{"uuid": "person"}, [None], [{}], [{"uuid": ""}], [{"uuid": 123}]])
def test_malformed_identity_cannot_bind_person(identity_peer, entities):
    identity_peer.respond = lambda request: (200, _envelope(request, {"resolved": True, "entities": entities}))
    result = factbook_source({"platform_id": "zoho_cliq:user:123", "name": "Alex"})
    assert result["status"] != "ok"
    assert result["entities"] == []
    assert len(identity_peer.requests) == 1


@pytest.mark.parametrize("other_identity", [
    {"name": "Alex"}, {"email": "other@example.test"},
    {"name": "Alex", "email": "other@example.test"},
])
def test_missing_authoritative_id_never_binds_another_identity(identity_peer, other_identity):
    def response(request):
        arguments = request["body"]["params"]["arguments"]
        payload = {"resolved": False, "entities": []} if arguments.get("attribute_key") == "platform_id" else {
            "resolved": True, "entities": [{"uuid": "unrelated-person"}],
        }
        return 200, _envelope(request, payload)

    identity_peer.respond = response
    result = factbook_source({"platform_id": "zoho_cliq:user:missing", **other_identity})
    assert result["status"] == "no_match"
    assert result["entities"] == []
    assert len(identity_peer.requests) == 1
