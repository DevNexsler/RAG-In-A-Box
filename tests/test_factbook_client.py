"""Unit tests for factbook_client (httpx -> factbook-rpc).

The mock transport()'s envelope shape is pinned against the real factbook
server source (read-only review of Graphiti-Factbook, factbook-rpc not yet
deployed): rpc_http.py's /mcp endpoint returns
FactbookMCPServer.handle_message()'s return value verbatim as JSON, and a
successful tools/call routes through _mcp_result() (server.py ~2369-2394),
which wraps the tool's own result dict as MCP content:
    {"jsonrpc": "2.0", "id": ..., "result":
        {"content": [{"type": "text", "text": json.dumps(tool_dict)}],
         "isError": bool}}
So this fixture matches production exactly — see factbook_client.py's module
docstring for the full citation, including the JSON-RPC-level error shape.
"""
import json
from functools import partial

import httpx
import pytest

import factbook_client as fc
from core.resilience import call_with_retry

# Repo convention for driving retry exhaustion without real sleeps
# (tests/test_openrouter_embed.py:49): swap the module's call_with_retry for
# a partial with sleep=no-op, keeping the real retry/classify logic intact.
_NO_SLEEP_RETRY = partial(call_with_retry, sleep=lambda _: None)

HIT = {"entities": [{"uuid": "u1", "name": "Jessica Ann Brown"}], "count": 1,
       "resolved": True, "ambiguous": False, "agency_identifier": False,
       "shared_channel_identifier": False, "cardinality_blocked_identifier": False}
MISS = {"entities": [], "count": 0, "resolved": False, "ambiguous": False,
        "agency_identifier": False, "shared_channel_identifier": False,
        "cardinality_blocked_identifier": False}


def transport(result_by_tool):
    def handler(request):
        body = json.loads(request.content)
        assert request.headers["authorization"] == "Bearer tok"
        tool = body["params"]["name"]
        # factbook wraps tool output as MCP content; mirror the real envelope
        return httpx.Response(200, json={
            "jsonrpc": "2.0", "id": body["id"],
            "result": {"content": [{"type": "text",
                                     "text": json.dumps(result_by_tool[tool])}]}})
    return httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("FACTBOOK_RPC_URL", "http://factbook-rpc:8695")
    monkeypatch.setenv("FACTBOOK_RPC_TOKEN", "tok")


def test_hit_by_email(monkeypatch):
    monkeypatch.setattr(fc, "_transport",
                        lambda: transport({"find_entity_by_attribute": HIT}))
    out = fc.factbook_source({"email": "a@b.com", "phone_e164": None, "name": None})
    assert out["status"] == "ok" and out["flags"]["resolved"] is True


def test_miss_falls_back_to_name(monkeypatch):
    monkeypatch.setattr(fc, "_transport", lambda: transport(
        {"find_entity_by_attribute": MISS,
         "resolve_entities": {"results": [], "entities": [], "count": 0}}))
    out = fc.factbook_source({"email": "a@b.com", "phone_e164": "+15550000000",
                              "name": "Nobody"})
    assert out["status"] == "no_match"


def test_http_error_degrades(monkeypatch):
    monkeypatch.setattr(fc, "_transport", lambda: httpx.MockTransport(
        lambda req: httpx.Response(503)))
    monkeypatch.setattr(fc, "call_with_retry", _NO_SLEEP_RETRY)
    out = fc.factbook_source({"email": "a@b.com", "phone_e164": None, "name": None})
    assert out["status"].startswith("error:")


def test_email_transport_failure_then_phone_miss_is_no_match(monkeypatch):
    """Mixed-outcome rule: one identifier's lookup fails outright (503 across
    every retry) but a LATER identifier gets a clean answer — even a miss.
    That real verdict outranks the earlier failure, so the overall status is
    "no_match", not "error:...". (factbook_client.py's final `if errors and
    not flags` check exists exactly for this: flags gets populated by the
    phone MISS, so the all-errored branch is skipped.)"""
    def handler(request):
        body = json.loads(request.content)
        args = body["params"]["arguments"]
        if args.get("attribute_key") == "email":
            return httpx.Response(503)  # transient — retried, then exhausted
        return httpx.Response(200, json={
            "jsonrpc": "2.0", "id": body["id"],
            "result": {"content": [{"type": "text", "text": json.dumps(MISS)}]}})

    monkeypatch.setattr(fc, "_transport", lambda: httpx.MockTransport(handler))
    monkeypatch.setattr(fc, "call_with_retry", _NO_SLEEP_RETRY)
    out = fc.factbook_source({"email": "a@b.com", "phone_e164": "+15550000000",
                              "name": None})
    assert out["status"] == "no_match"
    assert out["flags"]["resolved"] is False


def test_missing_token_degrades_without_network(monkeypatch):
    monkeypatch.delenv("FACTBOOK_RPC_TOKEN", raising=False)

    def _boom():  # pragma: no cover - must never be called
        raise AssertionError("no network call should happen without a token")

    monkeypatch.setattr(fc, "_transport", _boom)
    out = fc.factbook_source({"email": "a@b.com", "phone_e164": None, "name": None})
    assert out["status"] == "error:FACTBOOK_RPC_TOKEN unset"
    assert out["entities"] == [] and out["flags"] == {}


def test_email_hit_stops_before_trying_phone(monkeypatch):
    monkeypatch.setattr(fc, "_transport", lambda: transport(
        {"find_entity_by_attribute": HIT}))
    calls = []
    real_call_tool = fc._call_tool

    def spy(name, arguments, *, token):
        calls.append((name, arguments.get("attribute_key")))
        return real_call_tool(name, arguments, token=token)

    monkeypatch.setattr(fc, "_call_tool", spy)
    out = fc.factbook_source({"email": "a@b.com", "phone_e164": "+15550000000",
                              "name": None})
    assert out["status"] == "ok"
    # email is tried before phone; the fixture returns a HIT for either, so the
    # lookup should stop at the FIRST call (email) rather than also calling phone.
    assert calls == [("find_entity_by_attribute", "email")]
