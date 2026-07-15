"""Unified VPS entrypoint — starts MCP + REST API on $PORT (default 7788).

Routes:
    /mcp/*          MCP streamable-HTTP (for AI assistants)
    /api/upload     POST file upload
    /api/documents  GET list / GET download

Prefect lifecycle (#0325): the entrypoint runs the server inside a single
persistent ``PrefectServer`` (the same context ``run_index.py`` and
``mcp_server.__main__`` use) so there is exactly one stable Prefect server for
the container's lifetime and ``PREFECT_API_URL`` is exported process-wide. The
index runs launched by the ``file_index_update`` MCP tool are background
subprocesses that inherit this environment, so they attach to that one server
instead of each auto-starting — and, on an OOM/earlyoom kill, orphaning — its
own throwaway temporary server. The deployment env also disables Prefect's
ephemeral-server auto-start, so a missing ``PREFECT_API_URL`` fails loudly
rather than silently leaking orphan servers.
"""

import logging
import os

from core.config import load_config
from core.tracing import setup_tracing
from mcp_server import run_server
from prefect_server import PrefectServer


def main() -> None:
    config = load_config()

    log_level = config.get("logging", {}).get("level", "WARNING").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.WARNING))

    # No-op unless config has tracing.enabled: true; never raises.
    setup_tracing(config, "doc-organizer")

    host = config.get("mcp", {}).get("host", "0.0.0.0")
    port = int(os.environ.get("PORT", config.get("mcp", {}).get("port", 7788)))

    # One persistent Prefect server for the container's lifetime; index
    # subprocesses inherit PREFECT_API_URL and reuse it (never spawn their own).
    with PrefectServer():
        run_server(transport="streamable-http", host=host, port=port)


if __name__ == "__main__":
    main()
