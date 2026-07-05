"""MCP client wrapper for e2e: real streamable-HTTP transport + tool-coverage recording.

The staging container mounts the FastMCP streamable-HTTP app at "/" with the
FastMCP-default path "/mcp" (mcp_server.py run_server → mcp.streamable_http_app()),
so the endpoint is {E2E_BASE}/mcp. Every request needs the Bearer API key —
the AuthMiddleware in mcp_server.py rejects anything else with 401.
"""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

E2E_BASE = os.environ.get("E2E_BASE_URL", "http://localhost:17788")
E2E_API_KEY = os.environ.get("E2E_API_KEY", "staging-test-key")
E2E_SIM_URL = os.environ.get("E2E_SIM_URL", "http://localhost:19999")
MCP_URL = f"{E2E_BASE}/mcp"
COVERAGE_FILE = Path(os.environ.get("E2E_COVERAGE_FILE", ".evals/e2e-tool-coverage.jsonl"))

AUTH_HEADERS = {"Authorization": f"Bearer {E2E_API_KEY}"}


class RecordingSession:
    """Wraps a ClientSession; records tool coverage AFTER each successful call."""

    def __init__(self, session: ClientSession, test_name: str):
        self._s, self._test = session, test_name

    async def call_tool(self, name: str, arguments: dict):
        result = await self._s.call_tool(name, arguments)
        COVERAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with COVERAGE_FILE.open("a") as f:
            f.write(json.dumps({"tool": name, "test": self._test}) + "\n")
        return result

    async def call_tool_json(self, name: str, arguments: dict):
        """call_tool + parse the JSON payload out of the content blocks."""
        result = await self.call_tool(name, arguments)
        return tool_payload(result)

    async def list_tools(self):
        return await self._s.list_tools()


def tool_payload(result):
    """Extract the structured payload from a CallToolResult.

    FastMCP(json_response=True) returns tool results as JSON text content;
    structuredContent (when present) carries the parsed object, possibly
    wrapped in {"result": ...} for non-dict returns.
    """
    sc = getattr(result, "structuredContent", None)
    if sc is not None:
        if isinstance(sc, dict) and set(sc.keys()) == {"result"}:
            return sc["result"]
        return sc
    texts = [c.text for c in result.content if getattr(c, "type", None) == "text"]
    joined = "\n".join(texts)
    try:
        return json.loads(joined)
    except (json.JSONDecodeError, ValueError):
        return joined


async def get_hook_events() -> list[dict]:
    """Snapshot the provider-sim webhook sink (events since the last reset)."""
    import httpx

    async with httpx.AsyncClient(timeout=10) as sim:
        resp = await sim.get(f"{E2E_SIM_URL}/hooks/received")
        resp.raise_for_status()
        return resp.json()["events"]


def search_hits(payload, needle: str | None = None) -> list[dict]:
    """Assert a file_search payload is well-formed and return its results.

    With needle, filter to hits whose rel_path contains it.
    """
    assert isinstance(payload, dict) and not payload.get("error"), payload
    results = payload.get("results")
    assert isinstance(results, list), payload
    if needle is None:
        return results
    return [r for r in results if needle in r.get("rel_path", "")]


@asynccontextmanager
async def open_mcp_session(test_name: str):
    """Open a real streamable-HTTP MCP session against the staging container."""
    async with streamablehttp_client(MCP_URL, headers=dict(AUTH_HEADERS)) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield RecordingSession(session, test_name)
