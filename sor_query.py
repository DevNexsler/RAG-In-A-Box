"""Guarded read-only SOR (NocoDB-backed Postgres) query tool for the MCP server.

Safety is layered, not regex-based:
  * psycopg3's extended query protocol forbids multiple statements in one execute.
  * The connection runs with default_transaction_read_only=on, so any write/DDL
    fails at the database with a clear error.
  * validate_select() is only a fast-fail UX guard so obvious misuse returns a
    helpful message before touching the DB. It is intentionally NOT a keyword
    blocklist (those false-positive on string literals like '%please update%').
"""

from __future__ import annotations


def validate_select(sql: str) -> str | None:
    """Return an error string if the query is not a single SELECT/WITH, else None."""
    s = (sql or "").strip().rstrip(";").strip()
    if not s:
        return "empty query"
    low = s.lower()
    if not (low.startswith("select") or low.startswith("with")):
        return "sor_query is read-only; pass a single SELECT or WITH query"
    return None
