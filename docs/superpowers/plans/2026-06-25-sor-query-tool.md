# SOR structured query tool (`sor_query`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a guarded, read-only `sor_query` (+ `sor_schema`) MCP tool to RAG-in-a-Box so Hermes agents get exact/joined/aggregated SOR data in one bounded call, with the schema handed to them — killing the `information_schema` rediscovery turns and the `SELECT *` dumps.

**Architecture:** A new focused module `sor_query.py` holds all logic (DSN resolution, a lazy read-only psycopg connection, statement validation, auto-LIMIT wrapping, TTL-cached schema generation, TSV/JSON serialization, and two `*_impl` functions). `mcp_server.py` only adds two thin `@mcp.tool()` wrappers delegating to it — mirroring the existing `file_search` → `_file_search_impl` pattern. Safety is layered: psycopg3's extended protocol blocks multi-statement injection, `default_transaction_read_only=on` makes any write/DDL fail at the DB, and a light "must start with SELECT/WITH" check gives fast UX errors.

**Tech Stack:** Python 3.12, FastMCP (`mcp.server.fastmcp`), psycopg 3 (`psycopg[binary]>=3.3`, already a dep), `core.config.load_config`, pytest.

**Spec:** `docs/superpowers/specs/2026-06-25-sor-query-tool-design.md`

---

## File Structure

- **Create:** `sor_query.py` (top-level, alongside `mcp_server.py`). One responsibility: everything needed to safely run a read-only SOR query and to expose the SOR schema. Pure functions where possible (validation, wrapping, serialization, schema formatting) so they unit-test without a DB; the only DB-touching functions are `_get_readonly_conn`, `get_sor_schema`, `sor_query_impl`, `sor_schema_impl`.
- **Modify:** `mcp_server.py` — add `import sor_query` near the other imports, and two `@mcp.tool()` wrappers inside the existing `if HAS_MCP and FastMCP is not None:` block (after `file_search`).
- **Create test:** `tests/test_sor_query.py` — unit tests (no DB, via mocked connection) for every pure function + the impls; reuses the `mcp_server`-import contract pattern from `tests/test_mcp_contract.py` for tool registration.
- **Create test:** `tests/test_sor_query_integration.py` — integration tests behind a `@pytest.mark.skipif` guard that run only when `SOR_TEST_DSN` is set (disposable Postgres).

---

## Conventions to follow (from the existing codebase)

- DSN resolution mirrors `sources/postgres.py:60-69`: a `dsn` of the form `"${ENV_NAME}"` is resolved from `os.environ`.
- Connection recovery mirrors `sources/postgres.py:73-92`: reconnect if `None`/closed/`INERROR`.
- Config access: `from core.config import load_config; config = load_config("config.yaml")`; sources are `config["sources"]` (a list of dicts), the SOR one has `type: postgres, name: sor, dsn: "${SOR_DSN}"`.
- Run tests from repo root: `pytest tests/test_sor_query.py -v` (pytest config in `pyproject.toml [tool.pytest.ini_options]`).

---

## Task 1: Statement validation (read-only fast-fail)

**Files:**
- Create: `sor_query.py`
- Test: `tests/test_sor_query.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sor_query.py
import sor_query


def test_validate_accepts_select():
    assert sor_query.validate_select("SELECT 1") is None
    assert sor_query.validate_select("  select * from \"Contacts\"  ") is None


def test_validate_accepts_with_cte():
    assert sor_query.validate_select("WITH x AS (SELECT 1) SELECT * FROM x") is None


def test_validate_rejects_non_select():
    assert sor_query.validate_select("UPDATE t SET a=1") is not None
    assert sor_query.validate_select("") is not None
    assert sor_query.validate_select("   ") is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sor_query.py -k validate -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sor_query'` (or `AttributeError`).

- [ ] **Step 3: Write minimal implementation**

```python
# sor_query.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sor_query.py -k validate -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add sor_query.py tests/test_sor_query.py
git commit -m "feat(sor_query): statement validation fast-fail"
```

---

## Task 2: Auto-LIMIT wrapping + truncation detection

**Files:**
- Modify: `sor_query.py`
- Test: `tests/test_sor_query.py`

- [ ] **Step 1: Write the failing test**

```python
def test_wrap_adds_outer_limit_and_returns_effective():
    wrapped, eff = sor_query.wrap_with_limit("SELECT * FROM \"Contacts\"", 50)
    assert eff == 50
    assert "LIMIT 51" in wrapped          # fetch n+1 to detect truncation
    assert "_sub" in wrapped


def test_wrap_clamps_to_hard_max():
    _, eff = sor_query.wrap_with_limit("SELECT 1", 99999)
    assert eff == sor_query.HARD_MAX_ROWS


def test_wrap_floor_of_one():
    _, eff = sor_query.wrap_with_limit("SELECT 1", 0)
    assert eff == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sor_query.py -k wrap -v`
Expected: FAIL — `AttributeError: module 'sor_query' has no attribute 'wrap_with_limit'`.

- [ ] **Step 3: Write minimal implementation**

```python
# sor_query.py  (add near top, after imports)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sor_query.py -k wrap -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add sor_query.py tests/test_sor_query.py
git commit -m "feat(sor_query): auto-LIMIT wrapping with truncation probe"
```

---

## Task 3: Result serialization (TSV/JSON + cell capping + truncation notice)

**Files:**
- Modify: `sor_query.py`
- Test: `tests/test_sor_query.py`

- [ ] **Step 1: Write the failing test**

```python
def test_serialize_tsv_basic():
    rows = [{"id": 1, "name": "Rosado"}, {"id": 2, "name": "Fedak"}]
    out = sor_query.serialize(rows, "tsv", eff=50)
    assert out.splitlines()[0] == "id\tname"
    assert "1\tRosado" in out
    assert "of >" not in out             # not truncated


def test_serialize_truncation_notice():
    rows = [{"id": i} for i in range(51)]   # eff+1 fetched
    out = sor_query.serialize(rows, "tsv", eff=50)
    assert out.startswith("[50 of >50 rows")
    assert len(out.splitlines()) == 1 + 1 + 50   # notice + header + 50 rows


def test_serialize_cell_capping():
    rows = [{"notes": "x" * 1000}]
    out = sor_query.serialize(rows, "tsv", eff=50, cell_cap=100)
    assert "…" in out
    assert "x" * 1000 not in out


def test_serialize_json_format():
    import json
    rows = [{"id": 1, "name": "Rosado"}]
    out = sor_query.serialize(rows, "json", eff=50)
    assert json.loads(out) == [{"id": 1, "name": "Rosado"}]


def test_serialize_empty():
    assert "0 rows" in sor_query.serialize([], "tsv", eff=50)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sor_query.py -k serialize -v`
Expected: FAIL — no attribute `serialize`.

- [ ] **Step 3: Write minimal implementation**

```python
# sor_query.py
import json

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sor_query.py -k serialize -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add sor_query.py tests/test_sor_query.py
git commit -m "feat(sor_query): TSV/JSON serialization with cell capping"
```

---

## Task 4: DSN resolution + read-only connection

**Files:**
- Modify: `sor_query.py`
- Test: `tests/test_sor_query.py`

- [ ] **Step 1: Write the failing test**

```python
from unittest.mock import patch


def test_resolve_dsn_from_config_env(monkeypatch):
    monkeypatch.setenv("SOR_DSN", "postgresql://u@localhost/sor")
    fake = {"sources": [{"type": "postgres", "name": "sor", "dsn": "${SOR_DSN}"}]}
    with patch("sor_query.load_config", return_value=fake):
        assert sor_query.resolve_sor_dsn() == "postgresql://u@localhost/sor"


def test_resolve_dsn_missing_raises(monkeypatch):
    monkeypatch.delenv("SOR_DSN", raising=False)
    with patch("sor_query.load_config", return_value={"sources": []}):
        import pytest
        with pytest.raises(ValueError):
            sor_query.resolve_sor_dsn()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sor_query.py -k resolve_dsn -v`
Expected: FAIL — no attribute `resolve_sor_dsn` / `load_config`.

- [ ] **Step 3: Write minimal implementation**

```python
# sor_query.py
import os

import psycopg
from psycopg.rows import dict_row

from core.config import load_config

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sor_query.py -k resolve_dsn -v`
Expected: PASS (2 tests). (`_get_readonly_conn` is exercised by integration tests in Task 8.)

- [ ] **Step 5: Commit**

```bash
git add sor_query.py tests/test_sor_query.py
git commit -m "feat(sor_query): DSN resolution and read-only connection"
```

---

## Task 5: Schema generation, cache, and formatting

**Files:**
- Modify: `sor_query.py`
- Test: `tests/test_sor_query.py`

- [ ] **Step 1: Write the failing test**

```python
def test_format_schema_text_quotes_camelcase():
    schema = {
        "Building Units": [("id", "integer"), ("Unit_Name", "text")],
        "collection_tickets": [("id", "integer"), ("Status", "text")],
    }
    txt = sor_query.format_schema_text(schema, ["Building Units", "collection_tickets"])
    assert '"Building Units"("id", "Unit_Name")' in txt
    assert 'collection_tickets("id", "Status")' in txt   # snake table unquoted


def test_get_sor_schema_caches(monkeypatch):
    calls = {"n": 0}

    class FakeCur:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, *_): calls["n"] += 1
        def fetchall(self):
            return [{"table_name": "Contacts", "column_name": "id",
                     "data_type": "integer"}]

    class FakeConn:
        def cursor(self): return FakeCur()

    monkeypatch.setattr(sor_query, "_get_readonly_conn", lambda: FakeConn())
    sor_query._SCHEMA_CACHE["data"] = None
    s1 = sor_query.get_sor_schema(refresh=True)
    s2 = sor_query.get_sor_schema()           # cached, no 2nd execute
    assert s1 == s2
    assert calls["n"] == 1
    assert s1["Contacts"] == [("id", "integer")]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sor_query.py -k schema -v`
Expected: FAIL — no `format_schema_text` / `get_sor_schema`.

- [ ] **Step 3: Write minimal implementation**

```python
# sor_query.py
import re
import time

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
    return ident if _IDENT_OK.match(ident) else f'"{ident}"'


def get_sor_schema(refresh: bool = False) -> dict[str, list[tuple[str, str]]]:
    now = time.monotonic()
    if (not refresh and _SCHEMA_CACHE["data"] is not None
            and now - _SCHEMA_CACHE["ts"] < _SCHEMA_TTL_SECONDS):
        return _SCHEMA_CACHE["data"]
    conn = _get_readonly_conn()
    with conn.cursor() as cur:
        cur.execute(_SCHEMA_SQL)
        rows = cur.fetchall()
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
        col_str = ", ".join(_q(c) for c, _ in cols)
        lines.append(f"{_q(t)}({col_str})")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sor_query.py -k schema -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add sor_query.py tests/test_sor_query.py
git commit -m "feat(sor_query): TTL-cached schema generation and formatting"
```

---

## Task 6: The two impl functions + dynamic description

**Files:**
- Modify: `sor_query.py`
- Test: `tests/test_sor_query.py`

- [ ] **Step 1: Write the failing test**

```python
def _fake_conn(rows):
    class FakeCur:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, *_): pass
        def fetchall(self): return rows
    class FakeConn:
        def cursor(self): return FakeCur()
        def rollback(self): pass
    return FakeConn()


def test_sor_query_impl_happy_path(monkeypatch):
    monkeypatch.setattr(sor_query, "_get_readonly_conn",
                        lambda: _fake_conn([{"id": 1, "Status": "Open"}]))
    out = sor_query.sor_query_impl('SELECT id, "Status" FROM "Collection Tickets"')
    assert out.splitlines()[0] == "id\tStatus"
    assert "1\tOpen" in out


def test_sor_query_impl_rejects_write(monkeypatch):
    out = sor_query.sor_query_impl("DELETE FROM x")
    assert out.startswith("ERROR")


def test_sor_query_impl_db_error_is_returned(monkeypatch):
    import psycopg

    class BoomConn:
        def cursor(self): raise psycopg.Error("relation \"Nope\" does not exist")
        def rollback(self): pass

    monkeypatch.setattr(sor_query, "_get_readonly_conn", lambda: BoomConn())
    out = sor_query.sor_query_impl("SELECT * FROM \"Nope\"")
    assert "SOR query error" in out and "Nope" in out


def test_sor_schema_impl_unknown_table(monkeypatch):
    monkeypatch.setattr(sor_query, "get_sor_schema",
                        lambda **k: {"Contacts": [("id", "integer")]})
    out = sor_query.sor_schema_impl("Missing")
    assert "Unknown table" in out and "Contacts" in out


def test_build_description_falls_back_on_db_down(monkeypatch):
    def boom(**k): raise RuntimeError("db down")
    monkeypatch.setattr(sor_query, "get_sor_schema", boom)
    desc = sor_query.build_sor_query_description()
    assert "SELECT" in desc  # still returns usable static guidance
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sor_query.py -k "impl or description" -v`
Expected: FAIL — no `sor_query_impl` / `sor_schema_impl` / `build_sor_query_description`.

- [ ] **Step 3: Write minimal implementation**

```python
# sor_query.py
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
        conn = _get_readonly_conn()
        with conn.cursor() as cur:
            cur.execute(wrapped)
            rows = cur.fetchall()
        conn.rollback()  # close the read-only txn cleanly
    except psycopg.Error as e:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sor_query.py -k "impl or description" -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add sor_query.py tests/test_sor_query.py
git commit -m "feat(sor_query): sor_query_impl, sor_schema_impl, dynamic description"
```

---

## Task 7: Register the MCP tools in `mcp_server.py`

**Files:**
- Modify: `mcp_server.py` (add import near line 12-16; add two tools after `file_search`, ~line 1919)
- Test: `tests/test_sor_query.py`

- [ ] **Step 1: Write the failing test** (FastMCP contract — mirrors `tests/test_mcp_contract.py`)

```python
import pytest


@pytest.mark.anyio
async def test_sor_tools_registered():
    import mcp_server
    if not mcp_server.HAS_MCP:
        pytest.skip("mcp not installed")
    names = {t.name for t in await mcp_server.mcp.list_tools()}
    assert {"sor_query", "sor_schema"} <= names


@pytest.mark.anyio
async def test_sor_query_tool_dispatches(monkeypatch):
    import mcp_server
    if not mcp_server.HAS_MCP:
        pytest.skip("mcp not installed")
    from unittest.mock import patch
    with patch("sor_query.sor_query_impl", return_value="id\n1") as mock:
        await mcp_server.mcp.call_tool("sor_query", {"sql": "SELECT 1"})
    mock.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sor_query.py -k "registered or dispatches" -v`
Expected: FAIL — `sor_query`/`sor_schema` not in tool list.

- [ ] **Step 3: Add the import** near the other top-level imports in `mcp_server.py` (after line 15, with the other local imports):

```python
import sor_query
```

- [ ] **Step 4: Add the two tools** inside `if HAS_MCP and FastMCP is not None:`, immediately after the `file_search` function (after line 1919). Note the dynamic `description=` is computed once at registration (server start) from the live schema, with a built-in fallback:

```python
    @mcp.tool(description=sor_query.build_sor_query_description())
    def sor_query(sql: str, limit: int = 50, format: str = "tsv") -> str:
        return sor_query_module_sor_query_impl(sql, limit, format)

    @mcp.tool()
    def sor_schema(table: str | None = None) -> str:
        """Return the SOR Postgres schema (tables -> columns with types).

        Args:
            table: A table name to get its columns, or omit for a list of all
                SOR table names. Use this instead of querying information_schema.
        """
        return sor_query.sor_schema_impl(table)
```

Because the inner function is named `sor_query` (shadowing the module), bind the
impl to a module-level alias once, right after the `import sor_query` line:

```python
sor_query_module_sor_query_impl = sor_query.sor_query_impl
```

- [ ] **Step 5: Verify `@mcp.tool(description=...)` is supported**

Run: `python -c "from mcp.server.fastmcp import FastMCP; import inspect; print('description' in inspect.signature(FastMCP.tool).parameters)"`
Expected: `True`.
If `False`: instead of `description=`, set the docstring before registration — define the function, then `sor_query.__doc__ = sor_query.build_sor_query_description()` is not possible post-decoration, so use the documented FastMCP add-tool API: `mcp.add_tool(fn, name="sor_query", description=sor_query.build_sor_query_description())`. Adjust and re-run Step 6.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_sor_query.py -k "registered or dispatches" -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Run the whole unit file**

Run: `pytest tests/test_sor_query.py -v`
Expected: PASS (all tests).

- [ ] **Step 8: Commit**

```bash
git add mcp_server.py tests/test_sor_query.py
git commit -m "feat(sor_query): register sor_query + sor_schema MCP tools"
```

---

## Task 8: Integration tests against a real Postgres (read-only, joins, aggregation, timeout, schema refresh)

**Files:**
- Create: `tests/test_sor_query_integration.py`

These run only when `SOR_TEST_DSN` points at a disposable Postgres with a couple of
CamelCase tables. They verify the guarantees unit tests mock away.

- [ ] **Step 1: Write the integration tests**

```python
# tests/test_sor_query_integration.py
import os
import pytest

import sor_query

DSN = os.environ.get("SOR_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="SOR_TEST_DSN not set")


@pytest.fixture()
def seeded(monkeypatch):
    import psycopg
    with psycopg.connect(DSN) as conn:
        conn.execute('DROP TABLE IF EXISTS "Buildings"')
        conn.execute('CREATE TABLE "Buildings" (id int primary key, "Nick_Name" text)')
        conn.execute('INSERT INTO "Buildings" VALUES (1, \'Carriage\'), (2, \'Linden\')')
        conn.commit()
    monkeypatch.setattr(sor_query, "resolve_sor_dsn", lambda *a, **k: DSN)
    sor_query._CONN = None
    sor_query._SCHEMA_CACHE["data"] = None
    yield


def test_readonly_blocks_writes(seeded):
    out = sor_query.sor_query_impl('SELECT 1; ', )  # validator catches multi/non-select first
    # A write that passes the SELECT check still fails at the DB (read-only txn):
    out = sor_query.sor_query_impl('WITH x AS (DELETE FROM "Buildings" RETURNING 1) SELECT * FROM x')
    assert "error" in out.lower()


def test_aggregation_and_count(seeded):
    out = sor_query.sor_query_impl('SELECT count(*) AS n FROM "Buildings"')
    assert "n" in out.splitlines()[0]
    assert "2" in out


def test_limit_truncation(seeded):
    out = sor_query.sor_query_impl('SELECT * FROM "Buildings"', limit=1)
    assert out.startswith("[1 of >1 rows")


def test_schema_reflects_new_column(seeded):
    sor_query.get_sor_schema(refresh=True)
    import psycopg
    with psycopg.connect(DSN) as conn:
        conn.execute('ALTER TABLE "Buildings" ADD COLUMN "Risk_Score" int')
        conn.commit()
    refreshed = sor_query.get_sor_schema(refresh=True)
    assert any(c == "Risk_Score" for c, _ in refreshed["Buildings"])
```

- [ ] **Step 2: Run (skips cleanly without a DB)**

Run: `pytest tests/test_sor_query_integration.py -v`
Expected without `SOR_TEST_DSN`: all SKIPPED.
Expected with a disposable Postgres in `SOR_TEST_DSN`: PASS (4 tests).

- [ ] **Step 3: Commit**

```bash
git add tests/test_sor_query_integration.py
git commit -m "test(sor_query): integration tests for read-only, agg, limit, schema refresh"
```

---

## Task 9: Final verification + rollout notes

- [ ] **Step 1: Full test run**

Run: `pytest tests/test_sor_query.py tests/test_sor_query_integration.py -v`
Expected: unit PASS; integration SKIP (or PASS with `SOR_TEST_DSN`).

- [ ] **Step 2: Smoke-test the live tool via mcporter** (after the MCP server picks up the new code — restart it per RAG-in-a-Box's normal process)

Run:
```bash
printf '{"sql":"SELECT \"Status\", count(*) AS n FROM \"Collection Tickets\" GROUP BY \"Status\" ORDER BY n DESC","limit":20}' > /tmp/sq.json
npx mcporter call doc-organizer.sor_query --args "$(cat /tmp/sq.json)" --output json
npx mcporter call doc-organizer.sor_schema --args '{"table":"Contacts"}' --output json
```
Expected: a small TSV of statuses+counts; a column list for Contacts. No `information_schema` probe needed.

- [ ] **Step 3: Provisioning note (ops, optional hardening)**

The code enforces read-only via `default_transaction_read_only=on`. For defense in
depth, ops can later point `SOR_DSN` (for this tool) at a dedicated `sor_readonly`
role with only `SELECT` grants. Not required for v1 correctness; record as a
follow-up if desired.

- [ ] **Step 4: Verification against the original problem**

After a usage window, re-run the LiteLLM spend-log analysis (the LiteLLM-Setup
agent owns this): expect SOR-related `execute_code` turn counts to drop, average
result-byte size to fall, and `information_schema` probe frequency for core tables
to approach zero.

- [ ] **Step 5: Final commit (if any docs/cleanup)**

```bash
git add -A
git commit -m "docs(sor_query): rollout + verification notes"
```

---

## Notes for the implementer

- **Why string returns (not the `{"error": ...}` dict `file_search` uses):** `sor_query` returns tabular text, so success is TSV/JSON text and errors are `ERROR: ...` / `SOR query error: ...` strings — consistent with a text-returning tool and with the spec's examples. This is a deliberate, documented divergence.
- **Why always-wrap for LIMIT:** parsing SQL to detect an existing `LIMIT` is fragile; an outer `SELECT * FROM (<sql>) LIMIT n+1` is robust, supports CTEs/aggregates/ORDER BY, and the `n+1` fetch gives truncation detection for free.
- **Connection lifetime:** one lazy module-level connection, recovered on close/`INERROR` exactly like `PostgresSource`. The read-only + timeout options are libpq-level so they survive reconnects and need no per-query `SET`.
- **DRY:** `_q()` (identifier quoting) is shared by schema formatting; serialization/validation/wrapping are independent pure functions reused by both impls.
