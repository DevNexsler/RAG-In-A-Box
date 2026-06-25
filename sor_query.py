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
import re
import time

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


def _reset_conn() -> None:
    """Drop the cached connection so the next call opens a fresh one."""
    global _CONN
    if _CONN is not None and not _CONN.closed:
        try:
            _CONN.close()
        except Exception:
            pass
    _CONN = None


def _run_readonly(sql: str) -> list[dict]:
    """Execute a read-only query, fetch all rows, and close the txn cleanly.

    A cached connection can be reaped server-side while idle (Postgres
    idle-in-transaction timeout, or a dropped socket after the MCP server
    restarts); the client only discovers this on the next use, raising an
    OperationalError. Transparently discard the dead connection and retry once
    so that one-shot staleness never reaches the caller. Real SQL errors (bad
    table/column) are not OperationalErrors and propagate on the first attempt.
    """
    last_exc: "psycopg.OperationalError | None" = None
    for _attempt in range(2):
        try:
            conn = _get_readonly_conn()
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
            conn.rollback()  # close the read-only txn cleanly
            return rows
        except psycopg.OperationalError as e:
            last_exc = e
            _reset_conn()  # force a fresh connection on the retry
    raise last_exc  # both attempts failed — DB genuinely unreachable


_SCHEMA_CACHE: dict = {"data": None, "ts": 0.0}
_SCHEMA_TTL_SECONDS = 600.0

_SCHEMA_SQL = (
    "SELECT table_name, column_name, data_type "
    "FROM information_schema.columns "
    "WHERE table_schema = 'public' "
    "ORDER BY table_name, ordinal_position"
)

# Hot tables shown inline in the sor_query description (Tier 1). Authoritative
# identifiers always come from the live schema; this is only an ordering hint.
CORE_TABLES = [
    "Buildings", "Building Units", "Contacts",
    "Collection Tickets", "Tasks", "Legal Entities",
]

_IDENT_OK = re.compile(r"^[a-z_][a-z0-9_]*$")


def _q(ident: str) -> str:
    """Quote an identifier unless it's already a safe lowercase snake_case name."""
    return ident if _IDENT_OK.match(ident) else f'"{ident}"'


def get_sor_schema(refresh: bool = False) -> dict[str, list[tuple[str, str]]]:
    now = time.monotonic()
    if (not refresh and _SCHEMA_CACHE["data"] is not None
            and now - _SCHEMA_CACHE["ts"] < _SCHEMA_TTL_SECONDS):
        return _SCHEMA_CACHE["data"]
    rows = _run_readonly(_SCHEMA_SQL)
    schema: dict[str, list[tuple[str, str]]] = {}
    for r in rows:
        schema.setdefault(r["table_name"], []).append(
            (r["column_name"], r["data_type"])
        )
    _SCHEMA_CACHE["data"] = schema
    _SCHEMA_CACHE["ts"] = now
    return schema


def format_schema_text(schema: dict, tables: list[str] | None = None) -> str:
    if tables is None:
        ordered = [t for t in CORE_TABLES if t in schema]
        ordered += [t for t in sorted(schema) if t not in CORE_TABLES]
        tables = ordered
    lines = []
    for t in tables:
        cols = schema.get(t, [])
        # Columns are always quoted so the agent can copy them verbatim (NocoDB
        # exposes CamelCase columns like "Unit_Name"/"Status"). Table names are
        # quoted only when they need it (a plain snake_case table reads cleaner).
        col_str = ", ".join(f'"{c}"' for c, _ in cols)
        lines.append(f"{_q(t)}({col_str})")
    return "\n".join(lines)


_DESCRIPTION_INTRO = (
    "Run a READ-ONLY SQL query against the SOR Postgres (NocoDB-backed; "
    "identifiers are quoted CamelCase, e.g. \"Building Units\", \"Nick_Name\", "
    "\"Status\"). SELECT/WITH only. Results are capped (default LIMIT 50) and "
    "returned as TSV. Use joins and COUNT/SUM/GROUP BY freely. Call sor_schema "
    "for tables not listed below or to see all columns."
)


def sor_query_impl(sql: str, limit: int = 50, fmt: str = "tsv") -> str:
    err = validate_select(sql)
    if err:
        return f"ERROR: {err}"
    wrapped, eff = wrap_with_limit(sql, limit)
    try:
        rows = _run_readonly(wrapped)
    except psycopg.Error as e:
        # Clear any lingering error state so the next call starts clean.
        try:
            _get_readonly_conn().rollback()
        except Exception:
            pass
        return f"SOR query error: {str(e).strip()}"
    return serialize(rows, fmt, eff)


def sor_schema_impl(table: str | None = None) -> str:
    try:
        schema = get_sor_schema()
    except Exception as e:
        return f"SOR schema error: {str(e).strip()}"
    if table:
        if table not in schema:
            return (f"Unknown table {table!r}. Available: "
                    + ", ".join(sorted(schema)))
        return format_schema_text(schema, [table])
    return "\n".join(sorted(schema))


def build_sor_query_description() -> str:
    try:
        tables = format_schema_text(get_sor_schema())
    except Exception:
        tables = "(schema unavailable at startup; call sor_schema to list tables)"
    return f"{_DESCRIPTION_INTRO}\n\nTables and key columns:\n{tables}"
