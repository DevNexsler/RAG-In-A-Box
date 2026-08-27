"""factbook_client: httpx client for the factbook-rpc HTTP service.

POST http://factbook-rpc:8695/mcp — the same JSON-RPC "tools/call" messages
the factbook stdio transport reads (Graphiti-Factbook repo,
``factbook/rpc_http.py``: the ``/mcp`` endpoint is a bearer-auth wrapper that
returns ``FactbookMCPServer.handle_message(message)``'s return value verbatim
as JSON — ``rpc_http.py:43-53``).

Response envelope, pinned by READ-ONLY review of that server's source
(factbook-rpc is not deployed yet, so this could not be live-verified over
HTTP — see the Task B3 controller notes):

  * JSON-RPC transport-level failure — unknown method, or an exception raised
    inside ``handle_tool_call`` itself — comes back as
    ``{"jsonrpc": "2.0", "id": ..., "error": {"code": ..., "message": ...}}``
    with NO ``"result"`` key (``factbook/server.py:4995-5004``,
    ``handle_message``).
  * Success comes back as
    ``{"jsonrpc": "2.0", "id": ..., "result": {"content": [{"type": "text",
    "text": <json-encoded tool result dict>}], "isError": <bool>}}``
    — every ``tools/call`` result is wrapped by ``_mcp_result()``
    (``factbook/server.py:2369-2394``), which sets ``isError`` from the
    wrapped dict's OWN ``error``/``status`` fields, not from the transport.
    So a tool-level failure (e.g. a bad ``find_entity_by_attribute`` query)
    still arrives as HTTP 200 with a JSON-RPC ``result`` — it does NOT show
    up as the JSON-RPC ``error`` shape above.

``_call_tool`` unwraps ``result["content"][0]["text"]`` (JSON) and raises a
plain ``RuntimeError`` for either failure shape, so ``factbook_source`` can
decide how each identifier lookup degrades.
"""
from __future__ import annotations

import json
import os
import uuid

import httpx

from core.logging_setup import MAX_ERROR_CHARS, collapse
from core.resilience import call_with_retry, raise_for_status

DEFAULT_RPC_URL = "http://factbook-rpc:8695"
TOKEN_ENV_VAR = "FACTBOOK_RPC_TOKEN"
URL_ENV_VAR = "FACTBOOK_RPC_URL"

# factbook-rpc lives on the same docker network (mcp-backplane) as this
# service, not across the open internet like the LLM/embedding providers
# that also use core.resilience — a short ladder is enough to ride out a
# reconnect/redeploy blip without stalling the whole context_builder call.
ATTEMPTS = 3
BACKOFF: tuple[float, ...] = (0.3, 0.8)

# httpx's own read/connect timeout, independent of core.resilience's retry
# ladder above: at the old 30s, one hung backplane peer could cost
# 30s x ATTEMPTS x up to 3 identifiers (email/phone/name) ~= 4.5 minutes for
# a single context_builder call, and httpx.ReadTimeout alone never trips
# core.resilience's circuit breaker. factbook-rpc is same-network
# (mcp-backplane), not a WAN hop, so 8s is generous for a live peer and cuts
# the hung-peer worst case to well under a minute.
RPC_TIMEOUT_SECONDS = 8.0

# find_entity_by_attribute's own resolution flags (factbook/server.py, the
# payload dict built in _tool_find_entity_by_attribute, ~line 4137-4147).
# resolve_entities carries none of these (_tool_resolve_entities returns
# only entities/count/message) — flags are deliberately sourced ONLY from
# find_entity_by_attribute calls, per the Task B3 brief.
_FLAG_KEYS = (
    "resolved", "ambiguous", "agency_identifier",
    "shared_channel_identifier", "cardinality_blocked_identifier",
)


def _transport():
    """``None`` in production (httpx uses the real network transport);
    monkeypatched to an ``httpx.MockTransport`` in tests."""
    return None


def _rpc_url() -> str:
    return os.environ.get(URL_ENV_VAR, DEFAULT_RPC_URL).rstrip("/")


def _call_tool(name: str, arguments: dict, *, token: str) -> dict:
    """POST one JSON-RPC ``tools/call`` to factbook-rpc and return the
    unwrapped tool result dict. Raises ``RuntimeError`` (or an
    ``httpx``/network exception, after retries) on any transport, JSON-RPC,
    or tool-level failure — the caller decides how that degrades."""
    base_url = _rpc_url()
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    def _do() -> dict:
        with httpx.Client(transport=_transport(), timeout=RPC_TIMEOUT_SECONDS) as client:
            resp = client.post(f"{base_url}/mcp", json=payload, headers=headers)
        # 5xx/429/timeouts retried by the layer; a permanent 4xx raises
        # straight through carrying factbook-rpc's own body (#1657 pattern).
        raise_for_status(resp)
        return resp.json()

    data = call_with_retry(
        _do, attempts=ATTEMPTS, backoff=BACKOFF,
        label=f"factbook-rpc:{name}", circuit_key=base_url,
    )

    if data.get("error") is not None:
        # JSON-RPC transport-level failure (server.py:4995-5004).
        raise RuntimeError(f"factbook-rpc error: {data['error'].get('message')}")

    result = data.get("result") or {}
    content = result.get("content") or []
    text = content[0].get("text") if content else None
    if text is None:
        raise RuntimeError(f"factbook-rpc: malformed result for {name!r}: {result!r}")
    tool_result = json.loads(text)
    if result.get("isError") or tool_result.get("error") is not None:
        # Tool-level failure carried inside an otherwise-successful JSON-RPC
        # envelope (server.py:_mcp_result, isError set from the tool dict's
        # own error/status field — see module docstring).
        raise RuntimeError(
            f"factbook tool {name!r} failed: {tool_result.get('error') or tool_result}"
        )
    return tool_result


def factbook_source(contact: dict) -> dict:
    """Injectable ``factbook`` dep for ``context_builder.build_context``.

    Lookup order: email -> phone via ``find_entity_by_attribute``, stopping
    at the first call whose response carries entities; a name fallback via
    ``resolve_entities`` if neither identifier resolved (or was present).
    ``flags`` always reflect the LAST ``find_entity_by_attribute`` response —
    never ``resolve_entities``, which carries none of those keys (see module
    docstring). If no ``find_entity_by_attribute`` call was ever made (e.g.
    only a name was given), ``flags`` stays ``{}``.

    Never raises: every failure — missing token, transport, JSON-RPC, or
    tool-level — degrades to ``status: "error:<detail>"`` with empty
    ``entities``/``flags``, EXCEPT when a later identifier resolves cleanly
    (even to no match) despite an earlier one failing; a real verdict from
    one source outranks a transient failure on another.
    """
    token = os.environ.get(TOKEN_ENV_VAR)
    if not token:
        return {"status": f"error:{TOKEN_ENV_VAR} unset", "entities": [], "flags": {}}

    errors: list[str] = []
    flags: dict = {}

    for attribute_key, value in (
        ("email", contact.get("email")),
        ("phone", contact.get("phone_e164")),
    ):
        if not value:
            continue
        try:
            out = _call_tool(
                "find_entity_by_attribute",
                {"attribute_key": attribute_key, "attribute_value": value},
                token=token,
            )
        except Exception as exc:  # noqa: BLE001 — degrade per identifier
            errors.append(f"{attribute_key}: {collapse(exc, MAX_ERROR_CHARS)}")
            continue
        flags = {k: out.get(k) for k in _FLAG_KEYS}
        entities = out.get("entities") or []
        if entities:
            return {"status": "ok", "entities": entities, "flags": flags}

    name = contact.get("name")
    if name:
        try:
            out = _call_tool("resolve_entities", {"query": name}, token=token)
        except Exception as exc:  # noqa: BLE001 — degrade, keep prior flags
            errors.append(f"name: {collapse(exc, MAX_ERROR_CHARS)}")
        else:
            entities = out.get("entities") or []
            if entities:
                return {"status": "ok", "entities": entities, "flags": flags}

    if errors and not flags:
        # Every attempted identifier failed outright and nothing even
        # produced a "no match" verdict — a real source failure, not an
        # empty result.
        return {"status": f"error:{'; '.join(errors)}", "entities": [], "flags": {}}

    return {"status": "no_match", "entities": [], "flags": flags}
