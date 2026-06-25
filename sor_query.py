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

import json
import os

import psycopg
from psycopg.rows import dict_row

from core.config import load_config


def validate_select(sql: str) -> str | None:
    """Return an error string if the query is not a single SELECT/WITH, else None."""
    s = (sql or "").strip().rstrip(";").strip()
    if not s:
        return "empty query"
    low = s.lower()
    if not (low.startswith("select") or low.startswith("with")):
        return "sor_query is read-only; pass a single SELECT or WITH query"
    return None


HARD_MAX_ROWS = 1000


def wrap_with_limit(sql: str, limit: int) -> tuple[str, int]:
    """Wrap an arbitrary SELECT in an outer LIMIT so unbounded queries can't dump.

    Always wraps (no fragile LIMIT-parsing): an outer LIMIT caps everything,
    including queries that already have their own inner LIMIT. We request eff+1
    rows so the caller can detect truncation.
    """
    eff = max(1, min(int(limit), HARD_MAX_ROWS))
    inner = (sql or "").strip().rstrip(";").strip()
    wrapped = f"SELECT * FROM (\n{inner}\n) AS _sub LIMIT {eff + 1}"
    return wrapped, eff


DEFAULT_CELL_CAP = 500


def _cap_cell(value, cell_cap: int) -> str:
    s = "" if value is None else str(value)
    s = s.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    if len(s) > cell_cap:
        s = s[:cell_cap] + "…"
    return s


def serialize(rows: list[dict], fmt: str, eff: int,
              cell_cap: int = DEFAULT_CELL_CAP) -> str:
    truncated = len(rows) > eff
    rows = rows[:eff]
    if fmt == "json":
        body = json.dumps(
            [{k: _cap_cell(v, cell_cap) if isinstance(v, str) else v
              for k, v in r.items()} for r in rows],
            default=str, ensure_ascii=False,
        )
    else:
        if not rows:
            body = "(0 rows)"
        else:
            cols = list(rows[0].keys())
            lines = ["\t".join(cols)]
            for r in rows:
                lines.append("\t".join(_cap_cell(r.get(c), cell_cap) for c in cols))
            body = "\n".join(lines)
    if truncated:
        return f"[{eff} of >{eff} rows shown — add LIMIT or narrow the WHERE]\n" + body
    return body


_CONN: "psycopg.Connection | None" = None


def resolve_sor_dsn(config_path: str = "config.yaml") -> str:
    dsn = ""
    try:
        config = load_config(config_path)
        for src in config.get("sources", []):
            if src.get("type") == "postgres" and src.get("name") == "sor":
                dsn = src.get("dsn", "") or ""
                break
    except Exception:
        dsn = ""
    if not dsn:
        dsn = os.environ.get("SOR_DSN", "")
    if dsn.startswith("${") and dsn.endswith("}"):
        dsn = os.environ.get(dsn[2:-1], "")
    if not dsn:
        raise ValueError("SOR DSN not configured (sources['sor'].dsn / $SOR_DSN)")
    return dsn


def _get_readonly_conn() -> "psycopg.Connection":
    """Lazy, recovering, read-only connection in the MCP process.

    Read-only + a 15s statement timeout are enforced at the libpq level via
    connection options, so they apply to every query and survive reconnects.
    """
    global _CONN
    from psycopg.pq import TransactionStatus
    stale = (
        _CONN is None
        or _CONN.closed
        or _CONN.pgconn.transaction_status == TransactionStatus.INERROR
    )
    if stale:
        if _CONN is not None and not _CONN.closed:
            try:
                _CONN.close()
            except Exception:
                pass
        _CONN = psycopg.connect(
            resolve_sor_dsn(),
            row_factory=dict_row,
            options="-c default_transaction_read_only=on -c statement_timeout=15000",
        )
    return _CONN
