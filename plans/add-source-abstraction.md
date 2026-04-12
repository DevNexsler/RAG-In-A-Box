# Add Source Protocol Abstraction (filesystem + postgres) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a `Source` protocol in `sources/` so RAG-in-a-Box can index multiple heterogeneous data sources (filesystem today, PostgreSQL for Comm-Data-Store next, Slack/Notion/Gmail later) through one uniform contract, with namespaced doc_ids that share a single LanceDB `chunks` table.

**Architecture:** Define a tiny `Source` protocol with two methods: `scan() → Iterator[SourceRecord]` and `extract(record) → ExtractionResult`. Refactor the filesystem scan path behind `FilesystemSource` (zero behavior change), then add `PostgresSource`. The flow iterates all configured sources, namespaces each record's `doc_id` as `"{source_name}::{doc_id}"`, and calls the owning source's `extract()`. Everything downstream (chunking, enrichment, embedding, LanceDB, hybrid search, MCP tools) is already source-agnostic and stays unchanged.

**Tech Stack:** Python 3.13, `psycopg[binary]==3.3.3` (binary wheel, no native build), existing LanceDB + LlamaIndex + Prefect pipeline, SQLite via stdlib for the `DocIDStore` migration, pytest.

---

## Pre-flight Verified Assumptions (do not re-check)

These were de-risked before the plan was written. Treat as true:

- `psycopg[binary]==3.3.3` installs cleanly on `python:3.13-slim` (doc-organizer's Dockerfile base). Binary wheel; no `pg_config`, no compiler, no build tools needed.
- Network path from a container on `rag-in-a-box_default` with `extra_hosts: - "host.docker.internal:host-gateway"` can reach `comm-data-store-postgres-1` at `host.docker.internal:5433`. Verified by running `psycopg.connect("postgresql://comm_data_store:change-me@host.docker.internal:5433/comm_data_store")` and pulling 636 messages + 44 transcripts.
- Comm-Data-Store lives at `/home/danpark/projects/Comm-Data-Store/` and uses the env password `change-me` in dev; the running container is `comm-data-store-postgres-1`. DB + user are both named `comm_data_store`. Published on host port 5433.
- Comm-Data-Store schema relevant to indexing: `messages(id, source, source_message_id, channel_id, sender_participant_id, sent_at, body)`, `transcripts(id, call_id, transcript_text, created_at)`, `channels(id, name, …)`, `participants(id, display_name, …)`, `calls(id, source, source_call_id, …)`. `(source, source_message_id)` is `UNIQUE` on `messages` and is the natural doc_id key.
- LanceDB metadata is schema-flexible — new string sub-fields are added via the existing `_evolve_metadata_schema()` path at `lancedb_store.py:97`, so a new `source_name` metadata field needs no migration, just writes.
- The current DocIDStore schema (`doc_id_store.py:107-145`) has one primary table `doc_registry(doc_id TEXT PK, rel_path TEXT NOT NULL, created REAL NOT NULL)` plus `counter`, `audit_log`, `retired_ids`. We will add `source_name` to `doc_registry` via `ALTER TABLE`.

## Locked Design Decisions (do not re-litigate)

1. **`Source` protocol lives in `sources/base.py`** with one `SourceRecord` dataclass and one `Source` Protocol class. `scan()` returns an iterator (streamable, not a list), `extract()` returns the existing `ExtractionResult` from `extractors.py` so the downstream pipeline needs zero changes.
2. **`DocIDStore.doc_registry` grows a nullable `source_name TEXT` column**, backfilled to `'documents'` for existing rows. Nullable → `NOT NULL` in a second step after backfill, in the same migration. One SQLite file keeps tracking all sources.
3. **Single LanceDB `chunks` table across all sources.** Doc_ids are globally namespaced as `"{source_name}::{doc_id}"` before upsert (e.g. `"documents::note1@00001@.md"` and `"comm_messages::quo/abc123"`). A new `source_name` metadata field is written on every chunk for `where` filtering at query time.
4. **Backward-compat shim in `core/config.py`**: if `config.yaml` uses the old `documents_root:` shape, synthesize an equivalent `sources: [{type: filesystem, name: "documents", root: ...}]` at load time. Both shapes work; mixing them in one file is a hard error.
5. **`psycopg[binary]==3.3.3`** goes into `requirements.txt`. No Dockerfile changes.
6. **`COMM_DATA_STORE_DSN`** env var threads from `.env` → `docker-compose.yml` → container, same pattern as `OPENROUTER_API_KEY`.
7. **`FilesystemSource` owns the existing `DocIDStore` rename/collision/retirement machinery.** `PostgresSource` registers its doc_ids in the same table but with `rel_path=NULL` — it never renames or collides, so the filesystem-only code paths simply never fire for PG rows.
8. **MCP `file_search` tool gains an optional `source_name` filter** alongside the existing `source_type` filter. `_VALID_SOURCE_NAMES` is derived at request time from `DocIDStore.distinct_source_names()` instead of a hardcoded set.
9. **Comm-Data-Store grows `updated_at` columns upstream on `messages` and `transcripts`**, not a `COALESCE(updated_at, sent_at)` expression in the query. This is more durable for any future Postgres source we add — every PostgresSource TableSpec can count on an `updated_at` column existing rather than each new table owner figuring out their own fallback. Migration lives in `Comm-Data-Store/migrations/004_updated_at_on_messages_and_transcripts.sql`, includes a backfill from `sent_at`/`created_at` so existing rows don't stampede the indexer on first run, and installs a `BEFORE UPDATE` trigger so future in-place edits bump `updated_at` automatically.

## Non-Goals

- No Slack/Notion/Gmail source — we're only building the abstraction + filesystem + postgres. Future sources land in a separate PR using the same protocol.
- No changes to the reranker, embeddings, chunking, or enrichment pipelines. All source-agnostic.
- No changes to `api_server.py` (REST upload/download still filesystem-only).
- No schema change to the Comm-Data-Store Postgres DB. We read it, we don't write to it.
- No migration of existing LanceDB data. The new `source_name` field gets written on new chunks only; existing chunks are backfilled via the `_evolve_metadata_schema()` path which defaults missing fields to empty string (searches on `source_name = 'documents'` will miss them until they're re-indexed, which we accept — re-indexing is triggered by mtime drift on next `file_index_update` anyway).

---

## File Structure

Changes organized by responsibility, not layer. Files that change together live together.

### New files

| File | Responsibility |
|---|---|
| `sources/__init__.py` | `build_source(config_dict)` factory/dispatcher. ~30 LOC. |
| `sources/base.py` | `SourceRecord` dataclass + `Source` Protocol. Pure types, no logic. ~60 LOC. |
| `sources/filesystem.py` | `FilesystemSource` — owns `DocIDStore` and wraps the existing scan + `extract_text` dispatch. Code mostly lifted from `flow_index_vault.py::scan_vault_task` (~226-339) + extract dispatch in the flow body. ~220 LOC after refactor. |
| `sources/postgres.py` | `PostgresSource` — psycopg connection pool, streaming server-side cursor scan, row→SourceRecord mapping via a declarative `TableSpec`. ~150 LOC. |
| `tests/sources/__init__.py` | Empty — package marker. |
| `tests/sources/test_base.py` | Contract tests for SourceRecord/Source. ~50 LOC. |
| `tests/sources/test_filesystem_source.py` | Parity test against the legacy `scan_vault_task` + extract path. ~120 LOC. |
| `tests/sources/test_postgres_source.py` | Unit tests using a real connection to `comm-data-store-postgres-1` (marked `@pytest.mark.live`, skipped without the DSN). ~180 LOC. |
| `tests/test_multi_source_flow.int.test.py` | Integration test: flow runs with two sources, namespacing works, source-scoped deletes work, backward-compat shim works. ~200 LOC. |
| `plans/add-source-abstraction.md` | This file. |

### Modified files

| File | Change |
|---|---|
| `requirements.txt` | Add `psycopg[binary]>=3.3,<4`. |
| `core/config.py` | Backward-compat shim: expand old `documents_root` into `sources:` list at load time. Validate `sources:` shape. |
| `doc_id_store.py` | Schema migration: add `source_name` column, backfill to `'documents'`, add `distinct_source_names()` method. |
| `flow_index_vault.py` | Replace the single `scan_vault_task(vault_root, ...)` call with a multi-source loop. Remove the now-dead inline extract-dispatch and delegate to `source.extract()`. Keep `scan_vault_task` as a thin wrapper that `FilesystemSource.scan()` calls internally, so the helper functions like `_matches_any` stay where they are. |
| `lancedb_store.py` | Add `source_name` to the set of fields `_build_where_clause` accepts (`:157-173` area). |
| `mcp_server.py` | Add `source_name: str \| None = None` parameter to `_file_search_impl` and the tool schema. Derive `_VALID_SOURCE_NAMES` at call time from the registry. |
| `docker-compose.yml` | Add `COMM_DATA_STORE_DSN=${COMM_DATA_STORE_DSN}` to the `environment:` list. |
| `config.yaml` | After deploy: add `sources:` block. Old `documents_root:` line stays for one deploy cycle as a fallback, then removed in task 12. |
| `.env` | Add `COMM_DATA_STORE_DSN=postgresql://comm_data_store:change-me@host.docker.internal:5433/comm_data_store` (dev). Document in comment. |
| `config.yaml.example`, `config.local.yaml.example`, `config.vps.yaml.example` | Add commented-out `sources:` example with both filesystem and postgres entries. |
| `CLAUDE.md` or `README.md` | One-paragraph addition: "Adding a new data source" pointing at `sources/base.py` and the existing implementations as reference. |

---

## Execution Order Overview

Phases build on each other. **Each task must leave the repo in a state where `pytest -q -m "not live"` passes** — incomplete tasks are never committed.

1. **Foundation** (Tasks 1–3): types, schema migration, config shim. No behavior change yet.
2. **Filesystem parity** (Task 4): refactor current scan path behind `FilesystemSource`. Regression tests prove zero behavior drift.
3. **Postgres prerequisites** (Tasks 5, 5.5): dependency, upstream `updated_at` migration on Comm-Data-Store.
4. **Postgres support** (Tasks 6–7): `PostgresSource`, dispatcher.
5. **Flow integration** (Tasks 8–9): multi-source loop in the flow, LanceDB source_name metadata.
6. **Search surface** (Task 10): MCP tool update, registry-derived validation.
7. **Deployment** (Tasks 11–12): container env wiring, enable comm-data-store in config, live smoke.

**Cross-repo note:** Task 5.5 modifies the `Comm-Data-Store` repo at `/home/danpark/projects/Comm-Data-Store/`. All other tasks modify `RAG-in-a-Box`. The subagent executing Task 5.5 must `cd` into Comm-Data-Store and make a commit there; that commit is independent of the RAG-in-a-Box commits.

---

## Task 1: Foundation — `Source` protocol and `SourceRecord` dataclass

**Why:** Everything else depends on these types. No logic, pure contracts, easy to review.

**Files:**
- Create: `sources/__init__.py` (empty for now — dispatcher added in Task 7)
- Create: `sources/base.py`
- Create: `tests/sources/__init__.py` (empty)
- Create: `tests/sources/test_base.py`

### Steps

- [ ] **Step 1.1: Write the failing test**

Create `tests/sources/test_base.py`:
```python
"""Contract tests for the Source protocol and SourceRecord dataclass."""

from dataclasses import FrozenInstanceError
from typing import Iterator

import pytest

from sources.base import Source, SourceRecord
from extractors import ExtractionResult


def test_source_record_is_frozen():
    """SourceRecord is immutable so the dispatch loop can't mutate it by accident."""
    r = SourceRecord(
        doc_id="x",
        source_type="md",
        natural_key="x.md",
        mtime=1.0,
        size=10,
        metadata={"k": "v"},
    )
    with pytest.raises(FrozenInstanceError):
        r.doc_id = "y"  # type: ignore[misc]


def test_source_record_metadata_defaults_to_empty_dict():
    r = SourceRecord(doc_id="x", source_type="md", natural_key="x.md", mtime=1.0, size=10)
    assert r.metadata == {}


def test_source_protocol_is_structural():
    """Any class with scan/extract/name/close satisfies the protocol — no subclassing required."""

    class Stub:
        name = "stub"

        def scan(self) -> Iterator[SourceRecord]:
            yield SourceRecord(doc_id="a", source_type="md", natural_key="a.md", mtime=1.0, size=5)

        def extract(self, record: SourceRecord) -> ExtractionResult:
            return ExtractionResult.from_text("hello")

        def close(self) -> None:
            pass

    s: Source = Stub()  # Would fail type-check if Protocol wasn't structural
    assert s.name == "stub"
    records = list(s.scan())
    assert len(records) == 1
    assert records[0].doc_id == "a"
```

- [ ] **Step 1.2: Run the test to verify it fails**

```bash
PYTHONPATH=. pytest tests/sources/test_base.py -v
```

Expected: `ModuleNotFoundError: No module named 'sources'`.

- [ ] **Step 1.3: Create the empty package**

```bash
mkdir -p sources tests/sources
touch sources/__init__.py tests/sources/__init__.py
```

- [ ] **Step 1.4: Write the minimal implementation**

Create `sources/base.py`:
```python
"""Source protocol: uniform contract for heterogeneous data sources.

All sources (filesystem, postgres, slack, ...) implement this so the indexing
flow can iterate them uniformly. Downstream (chunk, embed, store, search) is
already source-agnostic and never sees this layer.
"""

from dataclasses import dataclass, field
from typing import Iterator, Protocol, runtime_checkable

from extractors import ExtractionResult


@dataclass(frozen=True)
class SourceRecord:
    """One indexable unit emitted by a Source. Shape is what diff_index_task
    and the flow's per-record loop expect.

    doc_id is unique *within* the source. The flow namespaces it globally
    by prefixing "{source_name}::" before upserting into LanceDB.
    """
    doc_id: str
    source_type: str
    natural_key: str
    mtime: float
    size: int
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class Source(Protocol):
    """Structural protocol — any class with these four members is a Source."""

    name: str

    def scan(self) -> Iterator[SourceRecord]:
        """Yield every indexable record in this source. Should be streaming
        (not load everything into memory) so large sources don't OOM."""
        ...

    def extract(self, record: SourceRecord) -> ExtractionResult:
        """Convert a record to its extractable text + frontmatter. May do I/O
        (read file from disk, fetch row from DB) or may use cached data on
        the record if scan() already populated it."""
        ...

    def close(self) -> None:
        """Release any connections/handles. Called once after the last scan."""
        ...
```

- [ ] **Step 1.5: Run the test to verify it passes**

```bash
PYTHONPATH=. pytest tests/sources/test_base.py -v
```

Expected: 3 passed.

- [ ] **Step 1.6: Run the full non-live suite to confirm nothing broke**

```bash
PYTHONPATH=. pytest -q -m "not live"
```

Expected: 530+ passed (should match pre-change count + 3).

- [ ] **Step 1.7: Commit**

```bash
git add sources/__init__.py sources/base.py tests/sources/__init__.py tests/sources/test_base.py
git commit -m "Add Source protocol and SourceRecord dataclass

Introduce sources/base.py defining the uniform contract that all
data sources (filesystem, postgres, future Slack/Notion/etc.) will
implement. Pure types — no logic, no behavior change to any
existing code path."
```

---

## Task 2: DocIDStore migration — add `source_name` column

**Why:** Before any source other than filesystem can register doc_ids, the registry needs to know *which* source each row belongs to. Additive migration; existing rows backfilled to `'documents'`.

**Files:**
- Modify: `doc_id_store.py` (schema init + new method)
- Modify: `tests/test_doc_id_store.py`

### Steps

- [ ] **Step 2.1: Read the current test file to find the right place**

```bash
grep -n "class Test\|def test_" tests/test_doc_id_store.py | head -20
```

Note the class structure; add the new test class at the bottom.

- [ ] **Step 2.2: Write the failing test**

Append to `tests/test_doc_id_store.py`:
```python
class TestSourceNameMigration:
    """Schema migration adds source_name column, backfills existing rows."""

    def test_new_store_has_source_name_column(self, tmp_path):
        from doc_id_store import DocIDStore

        store = DocIDStore(tmp_path / "reg.db")
        store.register("abc", "note.md")

        # Column exists and defaults to 'documents' for rows registered
        # without explicit source_name (backward compat for Task 4 which
        # hasn't updated FilesystemSource yet).
        cur = store._conn.execute("SELECT source_name FROM doc_registry WHERE doc_id = 'abc'")
        assert cur.fetchone()[0] == "documents"

    def test_legacy_store_gets_backfilled_on_open(self, tmp_path):
        """A pre-migration DB (no source_name column) opens cleanly and
        existing rows get backfilled to 'documents'."""
        import sqlite3

        db = tmp_path / "legacy.db"
        # Simulate a pre-migration DB: create doc_registry *without* source_name.
        conn = sqlite3.connect(db)
        conn.executescript("""
            CREATE TABLE doc_registry (
                doc_id TEXT PRIMARY KEY,
                rel_path TEXT NOT NULL,
                created REAL NOT NULL
            );
            CREATE TABLE counter (id INTEGER PRIMARY KEY CHECK (id = 1), value INTEGER NOT NULL DEFAULT 0);
            INSERT INTO counter VALUES (1, 0);
            INSERT INTO doc_registry VALUES ('old1', 'old.md', 0.0);
        """)
        conn.commit()
        conn.close()

        # Opening via DocIDStore should run the migration
        from doc_id_store import DocIDStore
        store = DocIDStore(db)

        cur = store._conn.execute("SELECT source_name FROM doc_registry WHERE doc_id = 'old1'")
        assert cur.fetchone()[0] == "documents"

    def test_register_accepts_source_name(self, tmp_path):
        """New register() signature accepts source_name kwarg."""
        from doc_id_store import DocIDStore

        store = DocIDStore(tmp_path / "reg.db")
        store.register("pg1", "quo/abc123", source_name="comm_messages")

        cur = store._conn.execute(
            "SELECT rel_path, source_name FROM doc_registry WHERE doc_id = 'pg1'"
        )
        row = cur.fetchone()
        assert row[0] == "quo/abc123"
        assert row[1] == "comm_messages"

    def test_distinct_source_names(self, tmp_path):
        """Helper for mcp_server to derive _VALID_SOURCE_NAMES at request time."""
        from doc_id_store import DocIDStore

        store = DocIDStore(tmp_path / "reg.db")
        store.register("a", "x.md", source_name="documents")
        store.register("b", "y.md", source_name="documents")
        store.register("c", "quo/123", source_name="comm_messages")

        assert store.distinct_source_names() == {"documents", "comm_messages"}
```

- [ ] **Step 2.3: Run the failing test**

```bash
PYTHONPATH=. pytest tests/test_doc_id_store.py::TestSourceNameMigration -v
```

Expected: 4 failures — `source_name` column does not exist, `register()` has no `source_name` kwarg, `distinct_source_names` does not exist.

- [ ] **Step 2.4: Update `_init_schema` to add the column idempotently**

In `doc_id_store.py`, modify `_init_schema` (line 107). Replace the `doc_registry` `CREATE TABLE` block with:
```python
        c.execute("""
            CREATE TABLE IF NOT EXISTS doc_registry (
                doc_id      TEXT PRIMARY KEY,
                rel_path    TEXT NOT NULL,
                created     REAL NOT NULL,
                source_name TEXT NOT NULL DEFAULT 'documents'
            )
        """)
        # Migration: add source_name column to pre-existing tables.
        # SQLite ALTER TABLE ADD COLUMN is a no-op if the column already exists,
        # but raises OperationalError, so we detect and swallow.
        try:
            c.execute("ALTER TABLE doc_registry ADD COLUMN source_name TEXT NOT NULL DEFAULT 'documents'")
        except Exception:
            pass  # Column already exists (fresh DB from the CREATE above, or already migrated)
```

- [ ] **Step 2.5: Update `register` to accept `source_name`**

Find the current `register` method (around line 156). Change its signature to use a None sentinel:

    def register(
        self,
        doc_id: str,
        rel_path: str,
        *,
        event: str = "",
        detail: str = "",
        source_name: str | None = None,
    ) -> None:

When `source_name is None` (legacy callers that don't know about sources), the INSERT uses a literal `'documents'` default for new rows and the `ON CONFLICT DO UPDATE SET` clause updates ONLY `rel_path` (preserving the existing `source_name` on the row). When `source_name` is an explicit string, both `rel_path` and `source_name` are updated on conflict. Branch like:

    if source_name is None:
        c.execute(
            "INSERT INTO doc_registry (doc_id, rel_path, created, source_name) "
            "VALUES (?, ?, ?, 'documents') "
            "ON CONFLICT(doc_id) DO UPDATE SET rel_path=excluded.rel_path",
            (doc_id, rel_path, time.time()),
        )
    else:
        c.execute(
            "INSERT INTO doc_registry (doc_id, rel_path, created, source_name) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(doc_id) DO UPDATE SET "
            "    rel_path=excluded.rel_path, "
            "    source_name=excluded.source_name",
            (doc_id, rel_path, time.time(), source_name),
        )

The `None` sentinel is load-bearing: it means "caller didn't specify, don't touch existing source_name". `FilesystemSource` and `PostgresSource` always pass their instance name explicitly, so they take the second branch. Legacy callers in `scan_vault_task` that don't pass source_name take the first branch, which preserves any source_name that was already on the row.

Preserve all existing audit-log logic inside the method.

- [ ] **Step 2.6: Add `distinct_source_names` method**

Add to `doc_id_store.py` near `all_mappings` (line ~245):
```python
    def distinct_source_names(self) -> set[str]:
        """Return the set of source_name values currently in the registry.
        Used by mcp_server to validate the source_name filter parameter."""
        cur = self._conn.execute("SELECT DISTINCT source_name FROM doc_registry")
        return {row[0] for row in cur.fetchall()}
```

- [ ] **Step 2.7: Run the new tests to verify they pass**

```bash
PYTHONPATH=. pytest tests/test_doc_id_store.py::TestSourceNameMigration -v
```

Expected: 4 passed.

- [ ] **Step 2.8: Run all of `test_doc_id_store.py` to catch regressions**

```bash
PYTHONPATH=. pytest tests/test_doc_id_store.py -v
```

Expected: all passed (the pre-existing tests didn't touch `source_name` and should still work).

- [ ] **Step 2.9: Run the full non-live suite**

```bash
PYTHONPATH=. pytest -q -m "not live"
```

Expected: 530+ passed (no regressions).

- [ ] **Step 2.10: Commit**

```bash
git add doc_id_store.py tests/test_doc_id_store.py
git commit -m "Migrate DocIDStore schema: add source_name column

doc_registry grows a source_name TEXT NOT NULL DEFAULT 'documents'
column. Additive migration via ALTER TABLE, idempotent, runs on every
_init_schema so pre-existing DBs are backfilled transparently.

register() now accepts a source_name kwarg (defaults to 'documents'
so existing FilesystemSource callers keep working until Task 4
passes the explicit value).

Adds distinct_source_names() for mcp_server's runtime validation."
```

---

## Task 3: Config backward-compat shim

**Why:** The plan adds a new `sources:` top-level key to `config.yaml`. Old configs with `documents_root:` must keep working through Task 11. One shape wins at load time (new), and the old shape is expanded into it.

**Files:**
- Modify: `core/config.py`
- Create: `tests/test_config_sources.py`

### Steps

- [ ] **Step 3.1: Write the failing test**

Create `tests/test_config_sources.py`:
```python
"""Tests for the sources: backward-compat shim in core/config.py."""

from pathlib import Path

import pytest
import yaml

from core.config import load_config


def _write_config(tmp_path: Path, data: dict) -> Path:
    """Write a YAML config file AND create any referenced paths so load_config
    validation doesn't reject it."""
    p = tmp_path / "test_config.yaml"
    # Ensure documents_root / vault_root / source roots exist on disk
    roots = []
    if "documents_root" in data:
        roots.append(data["documents_root"])
    if "vault_root" in data:
        roots.append(data["vault_root"])
    for src in data.get("sources", []):
        if src.get("type") == "filesystem" and "root" in src:
            roots.append(src["root"])
    for r in roots:
        Path(r).mkdir(parents=True, exist_ok=True)
    data.setdefault("index_root", str(tmp_path / "index"))
    Path(data["index_root"]).mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data))
    return p


def test_old_style_config_synthesizes_single_filesystem_source(tmp_path):
    """documents_root: /path expands to sources: [{type: filesystem, name: documents, root: /path}]."""
    cfg_path = _write_config(tmp_path, {
        "documents_root": str(tmp_path / "vault"),
    })
    cfg = load_config(str(cfg_path))
    assert "sources" in cfg
    assert len(cfg["sources"]) == 1
    src = cfg["sources"][0]
    assert src["type"] == "filesystem"
    assert src["name"] == "documents"
    assert src["root"] == str(tmp_path / "vault")


def test_new_style_config_loads_sources_as_is(tmp_path):
    cfg_path = _write_config(tmp_path, {
        "sources": [
            {"type": "filesystem", "name": "docs", "root": str(tmp_path / "vault")},
            {"type": "postgres", "name": "comm", "dsn": "postgresql://...", "tables": []},
        ],
    })
    cfg = load_config(str(cfg_path))
    assert len(cfg["sources"]) == 2
    assert cfg["sources"][0]["name"] == "docs"
    assert cfg["sources"][1]["type"] == "postgres"


def test_mixing_old_and_new_style_is_an_error(tmp_path):
    cfg_path = _write_config(tmp_path, {
        "documents_root": str(tmp_path / "vault"),
        "sources": [{"type": "filesystem", "name": "docs", "root": str(tmp_path / "vault")}],
    })
    with pytest.raises(ValueError, match="Cannot use both.*documents_root.*sources"):
        load_config(str(cfg_path))


def test_backward_compat_preserves_scan_include_exclude(tmp_path):
    """scan.include/exclude at the top level should flow into the synthesized source."""
    cfg_path = _write_config(tmp_path, {
        "documents_root": str(tmp_path / "vault"),
        "scan": {
            "include": ["**/*.md", "**/*.pdf"],
            "exclude": ["**/.git/**"],
        },
    })
    cfg = load_config(str(cfg_path))
    src = cfg["sources"][0]
    assert src["scan"]["include"] == ["**/*.md", "**/*.pdf"]
    assert src["scan"]["exclude"] == ["**/.git/**"]
```

- [ ] **Step 3.2: Run the failing test**

```bash
PYTHONPATH=. pytest tests/test_config_sources.py -v
```

Expected: failures — `sources` key not present after load.

- [ ] **Step 3.3: Implement the shim in `core/config.py`**

After the existing validation block in `load_config` (around line 61, right after `docs_path` is validated), insert:
```python
    # --- Sources shim (backward compat with documents_root:) ---
    has_sources = "sources" in raw
    has_legacy = any(k in raw for k in ("documents_root", "vault_root")) and not has_sources
    has_both = "sources" in raw and any(k in raw for k in ("documents_root", "vault_root"))

    if has_both:
        # Disallow mixing; force the author to choose one shape.
        raise ValueError(
            "Cannot use both 'documents_root' and 'sources' in the same config. "
            "Pick one: either keep 'documents_root' (legacy single-source) or "
            "switch to 'sources:' (multi-source)."
        )

    if not has_sources:
        # Synthesize a single filesystem source from the legacy keys.
        raw["sources"] = [{
            "type": "filesystem",
            "name": "documents",
            "root": raw["documents_root"],
            "scan": raw.get("scan", {}),
        }]
    else:
        # Validate new-style shape
        if not isinstance(raw["sources"], list) or not raw["sources"]:
            raise ValueError("'sources' must be a non-empty list")
        seen_names: set[str] = set()
        for i, src in enumerate(raw["sources"]):
            if not isinstance(src, dict):
                raise ValueError(f"sources[{i}] must be a mapping")
            if "type" not in src:
                raise ValueError(f"sources[{i}] missing required key 'type'")
            if "name" not in src:
                raise ValueError(f"sources[{i}] missing required key 'name'")
            if src["name"] in seen_names:
                raise ValueError(f"duplicate source name: {src['name']!r}")
            seen_names.add(src["name"])
```

Leave `documents_root` / `vault_root` in `raw` so legacy callers that read those keys directly (there are a couple in `flow_index_vault.py` still) keep working until Task 8 refactors them out.

- [ ] **Step 3.4: Run the new tests**

```bash
PYTHONPATH=. pytest tests/test_config_sources.py -v
```

Expected: 4 passed.

- [ ] **Step 3.5: Run the full non-live suite**

```bash
PYTHONPATH=. pytest -q -m "not live"
```

Expected: no regressions. The existing `test_config.py` and `test_config_edge_cases` cases all still use `documents_root:`, so they'll flow through the shim and synthesize a `sources:` list — nothing they assert should break.

- [ ] **Step 3.6: Commit**

```bash
git add core/config.py tests/test_config_sources.py
git commit -m "Add config shim: synthesize sources: from documents_root:

load_config() now always returns a 'sources' list on the config dict.
Old-style configs with 'documents_root' expand to a single filesystem
source named 'documents'. New-style configs with 'sources:' are
validated (list, required 'type' and 'name' keys, unique names).

Mixing both shapes is a ValueError. Legacy documents_root/vault_root
keys are preserved in the dict for downstream code that hasn't been
refactored yet (flow_index_vault.py does that in Task 8)."
```

---

## Task 4: Refactor filesystem scan behind `FilesystemSource`

**Why:** Move the existing scan + extract logic out of the flow and into a `FilesystemSource` class that implements the Source protocol. **Zero behavior change** — this is pure refactor, proven by a parity test.

**Files:**
- Create: `sources/filesystem.py`
- Create: `tests/sources/test_filesystem_source.py`
- Modify: `flow_index_vault.py` (no functional change yet — still called from the same place; just re-routes through the new class)

### Steps

- [ ] **Step 4.1: Write the parity test**

Create `tests/sources/test_filesystem_source.py`:
```python
"""Parity test: FilesystemSource.scan() + .extract() must produce the
same records and extraction output as the legacy scan_vault_task +
extract_text path in flow_index_vault.py."""

import shutil
from pathlib import Path

import pytest


@pytest.fixture
def tmp_vault(tmp_path: Path) -> Path:
    """Copy the real test_vault into a tmp dir so we don't mutate tracked files."""
    src = Path(__file__).parent.parent.parent / "test_vault"
    dst = tmp_path / "vault"
    shutil.copytree(src, dst)
    return dst


def test_scan_yields_same_records_as_legacy(tmp_vault):
    """Given identical config, FilesystemSource.scan() yields records with
    the same doc_id/mtime/size/source_type/metadata as scan_vault_task."""
    from doc_id_store import DocIDStore
    from flow_index_vault import scan_vault_task, _RUNTIME
    from sources.filesystem import FilesystemSource

    include = ["**/*.md", "**/*.pdf", "**/*.png"]
    exclude = []

    # Two separate registries so the two scans don't interfere with each other
    reg_legacy = DocIDStore(tmp_vault.parent / "legacy.db")
    reg_new = DocIDStore(tmp_vault.parent / "new.db")

    # Legacy path needs _RUNTIME populated with the registry
    _RUNTIME.clear()
    _RUNTIME["doc_id_store"] = reg_legacy
    legacy_records = scan_vault_task(tmp_vault, include, exclude)
    _RUNTIME.clear()

    # Reset the vault so rename side effects from the legacy scan don't bias the new one
    src = Path(__file__).parent.parent.parent / "test_vault"
    shutil.rmtree(tmp_vault)
    shutil.copytree(src, tmp_vault)

    src_new = FilesystemSource(
        name="documents",
        root=tmp_vault,
        scan_config={"include": include, "exclude": exclude},
        registry=reg_new,
    )
    new_records = list(src_new.scan())

    # Records are equivalent modulo the exact @XXXXX@ suffix (counters are
    # independent between the two registries). Compare normalized doc_ids.
    from doc_id_store import strip_id_from_filename

    def normalize(records):
        return sorted([
            (
                r["source_type"] if isinstance(r, dict) else r.source_type,
                strip_id_from_filename(
                    Path(r["rel_path"] if isinstance(r, dict) else r.natural_key).name
                ),
                r["size"] if isinstance(r, dict) else r.size,
            )
            for r in records
        ])

    assert normalize(legacy_records) == normalize(new_records)
    assert len(new_records) == len(legacy_records)


def test_extract_matches_legacy_extract_text(tmp_vault):
    """FilesystemSource.extract(record) returns the same text as
    the legacy extract_text() call with the same arguments."""
    from extractors import extract_text
    from doc_id_store import DocIDStore
    from sources.filesystem import FilesystemSource

    reg = DocIDStore(tmp_vault.parent / "reg.db")
    src = FilesystemSource(
        name="documents",
        root=tmp_vault,
        scan_config={"include": ["**/*.md"], "exclude": []},
        registry=reg,
    )

    records = list(src.scan())
    md_record = next(r for r in records if r.source_type == "md")

    new_result = src.extract(md_record)

    # Legacy equivalent
    legacy_result = extract_text(
        file_path=str(md_record.metadata["abs_path"]),
        ext="md",
        ocr_provider=None,
        pdf_strategy="text_then_ocr",
        min_text_chars=200,
        ocr_page_limit=200,
    )

    assert new_result.full_text == legacy_result.full_text
```

- [ ] **Step 4.2: Run the failing test**

```bash
PYTHONPATH=. pytest tests/sources/test_filesystem_source.py -v
```

Expected: `ModuleNotFoundError: No module named 'sources.filesystem'`.

- [ ] **Step 4.3: Create `sources/filesystem.py`**

This file is mostly a reshuffle of code already in `flow_index_vault.py`. Create `sources/filesystem.py`:
```python
"""FilesystemSource: scans a directory tree using scan.include/exclude globs
and extracts text via the existing extract_text() dispatcher.

This is a refactor of the legacy scan_vault_task + extract dispatch that
previously lived inline in flow_index_vault.py — zero behavior change,
proven by tests/sources/test_filesystem_source.py.
"""

from dataclasses import replace
from pathlib import Path
from typing import Iterator

from sources.base import Source, SourceRecord
from doc_id_store import DocIDStore
from extractors import ExtractionResult, extract_text


class FilesystemSource:
    """Indexes files matching glob patterns under a directory root."""

    def __init__(
        self,
        name: str,
        root: str | Path,
        scan_config: dict,
        registry: DocIDStore,
        pdf_config: dict | None = None,
    ):
        self.name = name
        self._root = Path(root)
        self._include = scan_config.get("include", ["**/*.md"])
        self._exclude = scan_config.get("exclude", [])
        self._registry = registry
        self._pdf_config = pdf_config or {}
        self._ocr_provider = None  # Set by flow after instantiation; see Task 8

    def set_ocr_provider(self, provider):
        """Injected by flow_index_vault after config is loaded. OCR is a
        flow-level concern, not a source-level one — filesystem doesn't
        know how to build an OCR provider and shouldn't have to."""
        self._ocr_provider = provider

    def scan(self) -> Iterator[SourceRecord]:
        """Delegate to the legacy scan_vault_task which owns the walk +
        rename + registry logic. Yield SourceRecord per returned dict."""
        from flow_index_vault import scan_vault_task, _RUNTIME

        # scan_vault_task reads _RUNTIME["doc_id_store"] — preserve that
        # contract for now. Task 8 will change scan_vault_task to accept
        # the registry as an argument.
        prev = _RUNTIME.get("doc_id_store")
        _RUNTIME["doc_id_store"] = self._registry
        try:
            records = scan_vault_task(self._root, self._include, self._exclude)
        finally:
            if prev is None:
                _RUNTIME.pop("doc_id_store", None)
            else:
                _RUNTIME["doc_id_store"] = prev

        # Backfill source_name on rows this scan just registered.
        # (Task 2's default handled new inserts; explicit update here makes
        # the source_name accurate if the row was pre-existing.)
        for r in records:
            self._registry._conn.execute(
                "UPDATE doc_registry SET source_name = ? WHERE doc_id = ?",
                (self.name, r["doc_id"]),
            )
        self._registry._conn.commit()

        for r in records:
            yield SourceRecord(
                doc_id=r["doc_id"],
                source_type=r.get("source_type_guess") or r["ext"],  # final mapping happens in flow via _SOURCE_TYPE_MAP
                natural_key=r["rel_path"],
                mtime=r["mtime"],
                size=r["size"],
                metadata={
                    "ext": r["ext"],
                    "abs_path": r["abs_path"],
                    "rel_path": r["rel_path"],
                },
            )

    def extract(self, record: SourceRecord) -> ExtractionResult:
        return extract_text(
            file_path=record.metadata["abs_path"],
            ext=record.metadata["ext"],
            ocr_provider=self._ocr_provider,
            pdf_strategy=self._pdf_config.get("strategy", "text_then_ocr"),
            min_text_chars=self._pdf_config.get("min_text_chars_before_ocr", 200),
            ocr_page_limit=self._pdf_config.get("ocr_page_limit", 200),
        )

    def close(self) -> None:
        pass  # Filesystem has no handle to close
```

- [ ] **Step 4.4: Run the test**

```bash
PYTHONPATH=. pytest tests/sources/test_filesystem_source.py -v
```

Expected: 2 passed.

- [ ] **Step 4.5: Run the full non-live suite**

```bash
PYTHONPATH=. pytest -q -m "not live"
```

Expected: no regressions. The flow still calls `scan_vault_task` directly, so nothing is broken yet — we've just added a new wrapper that tests confirm is equivalent.

- [ ] **Step 4.6: Commit**

```bash
git add sources/filesystem.py tests/sources/test_filesystem_source.py
git commit -m "Add FilesystemSource wrapping existing scan_vault_task

Refactor of the legacy filesystem scan path into a Source-protocol
implementation. Zero behavior change — delegates to scan_vault_task
internally and proven via parity test against the legacy flow.

The flow still calls scan_vault_task directly; Task 8 rewires it to
go through FilesystemSource.scan() instead."
```

---

## Task 5: Add `psycopg[binary]` to requirements

**Why:** Quick prerequisite for Task 6. Kept as its own task so if the CI Docker build fails we know exactly which change broke it.

**Files:**
- Modify: `requirements.txt`

### Steps

- [ ] **Step 5.1: Add the dependency line**

Append to `requirements.txt` in the appropriate section (near other HTTP/DB clients):
```
# PostgreSQL client (for sources.postgres.PostgresSource)
# Binary wheel: no pg_config, no build tools, works on python:3.13-slim.
psycopg[binary]>=3.3,<4
```

- [ ] **Step 5.2: Install locally and smoke-test the import**

```bash
pip install 'psycopg[binary]>=3.3,<4'
python3 -c "import psycopg; print('psycopg', psycopg.__version__)"
```

Expected: `psycopg 3.3.x` (or newer 3.x).

- [ ] **Step 5.3: Verify Docker build still works**

```bash
docker compose build doc-organizer 2>&1 | tail -20
```

Expected: `Image doc-organizer:latest Built` with `psycopg` in the installed packages line.

- [ ] **Step 5.4: Commit**

```bash
git add requirements.txt
git commit -m "Add psycopg[binary] dependency for PostgresSource"
```

---

## Task 5.5: Add `updated_at` columns upstream in Comm-Data-Store

**Why:** `PostgresSource` uses an `mtime_column` for incremental diffing. `sent_at` only captures the original timestamp — if a message is ever edited in place, the indexer would miss the change. Using a dedicated `updated_at` column maintained by a trigger is durable: every future Postgres source we add can assume an `updated_at` column exists, and correctness doesn't depend on each table owner reinventing a fallback.

This task touches a **different repository** (`/home/danpark/projects/Comm-Data-Store/`) and ends with a commit in that repo, not in RAG-in-a-Box.

**Files (in the `Comm-Data-Store` repo):**
- Create: `migrations/004_updated_at_on_messages_and_transcripts.sql`

### Steps

- [ ] **Step 5.5.1: Check current state of the Comm-Data-Store repo**

```bash
cd /home/danpark/projects/Comm-Data-Store
git status
git log --oneline -5
ls migrations/
```

Expected: clean working tree, recent migrations numbered 001–003 (two 003s — pre-existing numbering collision, leave it alone), no `004_*` file yet.

- [ ] **Step 5.5.2: Write the migration file**

Create `/home/danpark/projects/Comm-Data-Store/migrations/004_updated_at_on_messages_and_transcripts.sql`:
```sql
-- 004_updated_at_on_messages_and_transcripts.sql
--
-- Adds updated_at tracking to messages and transcripts so downstream
-- consumers (e.g. RAG-in-a-Box PostgresSource incremental indexing)
-- can diff on a column that reflects both inserts and in-place edits.
--
-- Strategy:
--   1. Add updated_at as nullable
--   2. Backfill from sent_at / created_at so historical rows aren't
--      re-processed as "new" when a consumer first runs after migration
--   3. Set NOT NULL with DEFAULT now() for future inserts
--   4. Install a BEFORE UPDATE trigger to bump updated_at on any edit

-- Shared trigger function (idempotent — uses CREATE OR REPLACE)
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- --- messages ---
ALTER TABLE messages ADD COLUMN IF NOT EXISTS updated_at timestamptz;
UPDATE messages SET updated_at = sent_at WHERE updated_at IS NULL;
ALTER TABLE messages ALTER COLUMN updated_at SET NOT NULL;
ALTER TABLE messages ALTER COLUMN updated_at SET DEFAULT now();

DROP TRIGGER IF EXISTS trg_messages_updated_at ON messages;
CREATE TRIGGER trg_messages_updated_at
    BEFORE UPDATE ON messages
    FOR EACH ROW
    EXECUTE FUNCTION set_updated_at();

-- Index for efficient incremental scans (ORDER BY updated_at)
CREATE INDEX IF NOT EXISTS idx_messages_updated_at ON messages (updated_at);

-- --- transcripts ---
ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS updated_at timestamptz;
UPDATE transcripts SET updated_at = created_at WHERE updated_at IS NULL;
ALTER TABLE transcripts ALTER COLUMN updated_at SET NOT NULL;
ALTER TABLE transcripts ALTER COLUMN updated_at SET DEFAULT now();

DROP TRIGGER IF EXISTS trg_transcripts_updated_at ON transcripts;
CREATE TRIGGER trg_transcripts_updated_at
    BEFORE UPDATE ON transcripts
    FOR EACH ROW
    EXECUTE FUNCTION set_updated_at();

CREATE INDEX IF NOT EXISTS idx_transcripts_updated_at ON transcripts (updated_at);
```

- [ ] **Step 5.5.3: Apply the migration to the live dev database**

The Comm-Data-Store repo has a helper:
```bash
cd /home/danpark/projects/Comm-Data-Store
./scripts/apply-migrations.sh
```

If the helper script doesn't exist or doesn't pick up 004 automatically, apply manually:
```bash
docker exec -i comm-data-store-postgres-1 psql -U comm_data_store -d comm_data_store \
    < /home/danpark/projects/Comm-Data-Store/migrations/004_updated_at_on_messages_and_transcripts.sql
```

Expected: `ALTER TABLE` × 4, `UPDATE 636` (or whatever the current message count is), `UPDATE 44` for transcripts, `CREATE TRIGGER` × 2, `CREATE INDEX` × 2, no errors.

- [ ] **Step 5.5.4: Verify the schema**

```bash
docker exec comm-data-store-postgres-1 psql -U comm_data_store -d comm_data_store -c "\d messages"
docker exec comm-data-store-postgres-1 psql -U comm_data_store -d comm_data_store -c "\d transcripts"
```

Expected: both tables list `updated_at | timestamp with time zone | not null default now()` as a column, and a trigger `trg_*_updated_at BEFORE UPDATE ...` is shown at the bottom.

- [ ] **Step 5.5.5: Verify backfill worked — existing rows have updated_at = sent_at**

```bash
docker exec comm-data-store-postgres-1 psql -U comm_data_store -d comm_data_store -c "
SELECT count(*) AS backfilled_correctly
FROM messages WHERE updated_at = sent_at;
"
```

Expected: count matches the total row count (all existing rows backfilled). New rows inserted after the migration will have `updated_at = now() >= sent_at`, which is still correct for diff purposes (newer or equal).

- [ ] **Step 5.5.6: Verify the trigger fires on UPDATE**

```bash
docker exec comm-data-store-postgres-1 psql -U comm_data_store -d comm_data_store -c "
BEGIN;
-- Pick any existing row
WITH t AS (SELECT id FROM messages LIMIT 1)
UPDATE messages SET body = body || '' WHERE id = (SELECT id FROM t)
RETURNING id, sent_at, updated_at, (updated_at > sent_at) AS trigger_fired;
ROLLBACK;
"
```

Expected: `trigger_fired` column is `t` (true) — the trigger bumped `updated_at` to `now()` which is after `sent_at`. ROLLBACK ensures we don't actually mutate the row.

- [ ] **Step 5.5.7: Commit in the Comm-Data-Store repo**

```bash
cd /home/danpark/projects/Comm-Data-Store
git add migrations/004_updated_at_on_messages_and_transcripts.sql
git commit -m "Add updated_at tracking to messages and transcripts

Adds nullable updated_at column, backfills from sent_at / created_at
on existing rows, then sets NOT NULL with DEFAULT now() for new rows.
Installs a BEFORE UPDATE trigger (set_updated_at()) that bumps the
column on any in-place edit.

Added so downstream consumers (RAG-in-a-Box PostgresSource) can diff
incrementally on updated_at and catch both inserts and edits — sent_at
alone only catches the original timestamp and misses later edits.
Also adds an index on updated_at for efficient ORDER BY scans."
```

- [ ] **Step 5.5.8: Push the Comm-Data-Store commit (optional, ask first)**

Do not auto-push. Surface to the human reviewer after the plan is otherwise complete. The RAG-in-a-Box tasks (6+) reference the live migrated DB, so as long as Step 5.5.3 applied successfully they can proceed even if the upstream commit isn't pushed yet.

- [ ] **Step 5.5.9: Return to the RAG-in-a-Box repo for the remaining tasks**

```bash
cd /home/danpark/projects/RAG-in-a-Box
git status
```

Expected: clean. No changes here — Task 5.5 only touched the other repo.

---

## Task 6: Implement `PostgresSource`

**Why:** The actual native Postgres reader. This is the biggest functional addition in the plan.

**Files:**
- Create: `sources/postgres.py`
- Create: `tests/sources/test_postgres_source.py` (live, skipped without DSN)

### Steps

- [ ] **Step 6.1: Write the failing unit tests**

Create `tests/sources/test_postgres_source.py`:
```python
"""Live tests for PostgresSource against comm-data-store-postgres-1.

Skipped unless COMM_DATA_STORE_DSN is set. Can be run locally with:
    COMM_DATA_STORE_DSN=postgresql://comm_data_store:change-me@localhost:5433/comm_data_store \\
        pytest tests/sources/test_postgres_source.py -v
"""

import os

import pytest

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("COMM_DATA_STORE_DSN"),
        reason="COMM_DATA_STORE_DSN not set — skipping live postgres tests",
    ),
]


@pytest.fixture
def pg_source():
    from sources.postgres import PostgresSource, TableSpec

    specs = [
        TableSpec(
            source_type="pg_message",
            query="""
                SELECT
                    m.source,
                    m.source_message_id,
                    m.sent_at,
                    m.updated_at,
                    m.body AS _text,
                    c.name AS channel,
                    p.display_name AS sender
                FROM messages m
                LEFT JOIN channels c ON c.id = m.channel_id
                LEFT JOIN participants p ON p.id = m.sender_participant_id
                WHERE m.body IS NOT NULL AND m.body <> ''
                ORDER BY m.id
                LIMIT 20
            """,
            id_template="{source}/{source_message_id}",
            text_column="_text",
            mtime_column="updated_at",
            metadata_columns=["channel", "sender", "source", "sent_at"],
        ),
    ]
    src = PostgresSource(
        name="comm_messages",
        dsn=os.environ["COMM_DATA_STORE_DSN"],
        tables=specs,
    )
    yield src
    src.close()


def test_scan_yields_records(pg_source):
    records = list(pg_source.scan())
    assert len(records) > 0
    assert len(records) <= 20  # respects LIMIT


def test_record_shape(pg_source):
    records = list(pg_source.scan())
    r = records[0]
    assert r.source_type == "pg_message"
    assert "/" in r.doc_id  # "{source}/{source_message_id}"
    assert r.mtime > 0
    assert r.size > 0
    assert "channel" in r.metadata
    assert "sender" in r.metadata
    assert "_text" in r.metadata  # cached text for zero-IO extract


def test_extract_returns_nonempty_text(pg_source):
    records = list(pg_source.scan())
    result = pg_source.extract(records[0])
    assert len(result.full_text) > 0
    # Frontmatter includes channel and sender
    assert "channel" in result.frontmatter or "sender" in result.frontmatter


def test_natural_key_matches_doc_id(pg_source):
    """For Postgres, natural_key and doc_id are the same (no rename layer)."""
    records = list(pg_source.scan())
    for r in records:
        assert r.doc_id == r.natural_key


def test_close_releases_connection(pg_source):
    pg_source.close()
    # Second close is a no-op
    pg_source.close()
```

- [ ] **Step 6.2: Run the failing test**

```bash
export COMM_DATA_STORE_DSN="postgresql://comm_data_store:change-me@localhost:5433/comm_data_store"
PYTHONPATH=. pytest tests/sources/test_postgres_source.py -v
```

Expected: `ModuleNotFoundError: No module named 'sources.postgres'`.

- [ ] **Step 6.3: Implement `sources/postgres.py`**

Create:
```python
"""PostgresSource: read indexable text from PostgreSQL tables.

Each TableSpec declares one SELECT query, the column holding the text,
the column to use as mtime, and a natural-key template. The source opens
one connection per scan, uses a server-side cursor to stream rows so
large tables don't OOM, and yields a SourceRecord per row.
"""

from dataclasses import dataclass, field
from typing import Iterator

import psycopg
from psycopg.rows import dict_row

from sources.base import Source, SourceRecord
from extractors import ExtractionResult


@dataclass
class TableSpec:
    source_type: str            # "pg_message", "pg_transcript", ...
    query: str                  # SELECT with all columns referenced below
    id_template: str            # Python .format() template, e.g. "{source}/{source_message_id}"
    text_column: str            # Name of the column holding the indexable text
    mtime_column: str           # Column holding timestamptz for diff mtime
    metadata_columns: list[str] = field(default_factory=list)


class PostgresSource:
    def __init__(self, name: str, dsn: str, tables: list[TableSpec]):
        self.name = name
        self._dsn = dsn
        self._tables = tables
        self._conn: psycopg.Connection | None = None

    def _get_conn(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self._dsn, row_factory=dict_row)
        return self._conn

    def scan(self) -> Iterator[SourceRecord]:
        conn = self._get_conn()
        for spec in self._tables:
            # Server-side cursor for streaming; named cursors stream in
            # batches of itersize rather than fetching everything.
            with conn.cursor(name=f"scan_{self.name}_{spec.source_type}") as cur:
                cur.itersize = 500
                cur.execute(spec.query)
                for row in cur:
                    doc_id = spec.id_template.format(**row)
                    text = row.get(spec.text_column) or ""
                    mtime_val = row[spec.mtime_column]
                    mtime = mtime_val.timestamp() if mtime_val else 0.0
                    metadata = {c: row[c] for c in spec.metadata_columns if c in row}
                    metadata["_text"] = text  # cache for zero-IO extract
                    yield SourceRecord(
                        doc_id=doc_id,
                        source_type=spec.source_type,
                        natural_key=doc_id,
                        mtime=mtime,
                        size=len(text.encode("utf-8")),
                        metadata=metadata,
                    )

    def extract(self, record: SourceRecord) -> ExtractionResult:
        # Text was cached by scan() — no second round-trip needed.
        text = record.metadata.get("_text", "")
        frontmatter = {k: v for k, v in record.metadata.items() if k != "_text"}
        return ExtractionResult.from_text(text, frontmatter=frontmatter)

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        self._conn = None
```

- [ ] **Step 6.4: Run the tests**

```bash
export COMM_DATA_STORE_DSN="postgresql://comm_data_store:change-me@localhost:5433/comm_data_store"
PYTHONPATH=. pytest tests/sources/test_postgres_source.py -v
```

Expected: 5 passed.

- [ ] **Step 6.5: Run the full non-live suite (no Postgres needed)**

```bash
unset COMM_DATA_STORE_DSN
PYTHONPATH=. pytest -q -m "not live"
```

Expected: no regressions. Postgres tests skip cleanly when DSN is unset.

- [ ] **Step 6.6: Commit**

```bash
git add sources/postgres.py tests/sources/test_postgres_source.py
git commit -m "Add PostgresSource for native SQL-backed indexing

PostgresSource implements Source protocol. Opens one psycopg3
connection per scan(), uses a named server-side cursor streaming
500 rows at a time, and yields one SourceRecord per row.

Text is cached on record.metadata during scan so extract() is a
zero-I/O local operation.

TableSpec declares: query, id_template, text_column, mtime_column,
metadata_columns. Multiple specs per source (e.g. messages +
transcripts) share one connection.

Tests hit comm-data-store-postgres-1 via COMM_DATA_STORE_DSN,
skipped when env var absent."
```

---

## Task 7: Source dispatcher — `sources/__init__.py`

**Why:** Single entry point the flow calls to build sources from config dicts.

**Files:**
- Modify: `sources/__init__.py`
- Create: `tests/sources/test_dispatcher.py`

### Steps

- [ ] **Step 7.1: Write the failing test**

Create `tests/sources/test_dispatcher.py`:
```python
"""Tests for the sources.build_source dispatcher."""

import pytest

from sources import build_source
from sources.filesystem import FilesystemSource
from sources.postgres import PostgresSource


def test_dispatch_filesystem(tmp_path):
    from doc_id_store import DocIDStore

    reg = DocIDStore(tmp_path / "r.db")
    (tmp_path / "vault").mkdir()
    src = build_source({
        "type": "filesystem",
        "name": "docs",
        "root": str(tmp_path / "vault"),
        "scan": {"include": ["**/*.md"], "exclude": []},
    }, registry=reg)
    assert isinstance(src, FilesystemSource)
    assert src.name == "docs"


def test_dispatch_postgres():
    src = build_source({
        "type": "postgres",
        "name": "comm",
        "dsn": "postgresql://user:pass@host:5432/db",
        "tables": [],
    })
    assert isinstance(src, PostgresSource)
    assert src.name == "comm"


def test_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown source type"):
        build_source({"type": "slack", "name": "x"})


def test_missing_name_raises():
    with pytest.raises(KeyError):
        build_source({"type": "filesystem", "root": "/tmp"})
```

- [ ] **Step 7.2: Run the failing test**

```bash
PYTHONPATH=. pytest tests/sources/test_dispatcher.py -v
```

Expected: `ImportError: cannot import name 'build_source' from 'sources'`.

- [ ] **Step 7.3: Implement the dispatcher**

Replace the empty `sources/__init__.py` with:
```python
"""Source protocol package. See sources/base.py for the contract."""

from sources.base import Source, SourceRecord
from sources.filesystem import FilesystemSource
from sources.postgres import PostgresSource, TableSpec

__all__ = [
    "Source",
    "SourceRecord",
    "FilesystemSource",
    "PostgresSource",
    "TableSpec",
    "build_source",
]


def build_source(config: dict, *, registry=None, pdf_config=None) -> Source:
    """Build a Source from a config dict (one entry of the sources: list).

    Args:
        config: {type, name, ...type-specific keys}
        registry: DocIDStore to pass to filesystem sources. Required for
            type='filesystem', ignored for type='postgres'.
        pdf_config: PDF extraction config to pass to filesystem sources.

    Raises:
        ValueError: Unknown source type.
        KeyError: Missing required config key.
    """
    name = config["name"]
    src_type = config.get("type")

    if src_type == "filesystem":
        if registry is None:
            raise ValueError(f"filesystem source '{name}' requires a DocIDStore registry")
        return FilesystemSource(
            name=name,
            root=config["root"],
            scan_config=config.get("scan", {}),
            registry=registry,
            pdf_config=pdf_config,
        )

    if src_type == "postgres":
        tables = [TableSpec(**t) for t in config.get("tables", [])]
        return PostgresSource(
            name=name,
            dsn=config["dsn"],
            tables=tables,
        )

    raise ValueError(f"Unknown source type: {src_type!r}")
```

- [ ] **Step 7.4: Run the tests**

```bash
PYTHONPATH=. pytest tests/sources/test_dispatcher.py -v
```

Expected: 4 passed.

- [ ] **Step 7.5: Run the full non-live suite**

```bash
PYTHONPATH=. pytest -q -m "not live"
```

Expected: no regressions.

- [ ] **Step 7.6: Commit**

```bash
git add sources/__init__.py tests/sources/test_dispatcher.py
git commit -m "Add sources.build_source dispatcher

One entry point flow_index_vault.py will call to build Sources from
the config['sources'] list. Dispatches on type, raises ValueError
for unknowns, and enforces that filesystem sources receive a registry."
```

---

## Task 8: Flow refactor — multi-source loop with namespaced doc_ids

**Why:** Wire it all together. This is the riskiest task — it rewrites the main scan/diff/extract loop in `flow_index_vault.py`. Keep existing tests green at every step.

**Files:**
- Modify: `flow_index_vault.py`
- Create: `tests/test_multi_source_flow.int.test.py`

### Steps

- [ ] **Step 8.1: Write the multi-source integration test**

Create `tests/test_multi_source_flow.int.test.py`:
```python
"""Integration test: flow_index_vault handles multiple sources correctly.

Covers: namespaced doc_ids prevent collisions, source-scoped deletes don't
affect other sources, existing filesystem tests still pass through the
backward-compat shim.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

_has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
_has_deepinfra = bool(os.environ.get("DEEPINFRA_API_KEY"))

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (_has_openrouter and _has_deepinfra),
        reason="OPENROUTER_API_KEY + DEEPINFRA_API_KEY required",
    ),
]


class TestMultiSourceFlow:
    """Run index_vault_flow with a config that has two filesystem sources
    (so we can test without a Postgres dependency) and verify namespacing."""

    @pytest.fixture(scope="class")
    def two_source_result(self):
        """Create two vault dirs, index both, return the lancedb store + config."""
        import yaml
        from core.config import load_config
        from lancedb_store import LanceDBStore
        from doc_id_store import DocIDStore

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            vault_a = tmp / "vault_a"
            vault_b = tmp / "vault_b"
            src = Path(__file__).parent.parent / "test_vault"
            shutil.copytree(src, vault_a)
            shutil.copytree(src, vault_b)
            index_root = tmp / "index"
            index_root.mkdir()

            # Build config with TWO filesystem sources
            config = {
                "sources": [
                    {"type": "filesystem", "name": "alpha", "root": str(vault_a),
                     "scan": {"include": ["**/*.md"], "exclude": []}},
                    {"type": "filesystem", "name": "beta", "root": str(vault_b),
                     "scan": {"include": ["**/*.md"], "exclude": []}},
                ],
                "index_root": str(index_root),
                "documents_root": str(vault_a),  # Legacy key retained for backward compat
                "chunking": {"max_chars": 1800, "overlap": 200, "semantic": {"enabled": False}},
                "embeddings": load_config()["embeddings"],
                "enrichment": {"enabled": False},  # Skip enrichment to keep test fast
                "search": {"mode": "hybrid", "vector_top_k": 10, "keyword_top_k": 10, "final_top_k": 5, "rrf_k": 60},
                "lancedb": {"table": "chunks"},
                "ocr": {"enabled": False},
                "pdf": {"strategy": "text_then_ocr", "ocr_page_limit": 200, "min_text_chars_before_ocr": 200},
                "mcp": {"host": "127.0.0.1", "port": 7788},
                "logging": {"level": "WARNING"},
            }

            config_path = tmp / "config.yaml"
            config_path.write_text(yaml.safe_dump(config))

            saved_env = {}
            for key in ("PREFECT_API_URL", "PREFECT_SERVER_ALLOW_EPHEMERAL_MODE", "DOCUMENTS_ROOT", "INDEX_ROOT"):
                saved_env[key] = os.environ.get(key)
            os.environ["PREFECT_API_URL"] = ""
            os.environ["PREFECT_SERVER_ALLOW_EPHEMERAL_MODE"] = "true"
            os.environ.pop("DOCUMENTS_ROOT", None)
            os.environ.pop("INDEX_ROOT", None)

            try:
                from prefect.settings.models.root import Settings as PrefectSettings
                import prefect.context
                prefect.context.get_settings_context().settings = PrefectSettings()
            except Exception:
                pass
            try:
                from flow_index_vault import index_vault_flow
                index_vault_flow(str(config_path))
            finally:
                for k, v in saved_env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

            store = LanceDBStore(str(index_root), "chunks")
            registry = DocIDStore(index_root / "doc_registry.db")
            yield {"store": store, "registry": registry, "vault_a": vault_a, "vault_b": vault_b, "index_root": index_root}

    def test_both_sources_indexed(self, two_source_result):
        """Every doc_id should be namespaced — alpha::... and beta::..."""
        doc_ids = two_source_result["store"].list_doc_ids()
        alpha_ids = [d for d in doc_ids if d.startswith("alpha::")]
        beta_ids = [d for d in doc_ids if d.startswith("beta::")]
        assert len(alpha_ids) >= 3
        assert len(beta_ids) >= 3

    def test_no_cross_source_collisions(self, two_source_result):
        """Same filename under two sources produces two distinct doc_ids."""
        doc_ids = two_source_result["store"].list_doc_ids()
        # note1.md appears in both vaults; with namespacing they don't collide
        assert any(d.startswith("alpha::") and "note1" in d for d in doc_ids)
        assert any(d.startswith("beta::") and "note1" in d for d in doc_ids)

    def test_registry_tracks_source_name(self, two_source_result):
        """DocIDStore has source_name populated for both sources."""
        registry = two_source_result["registry"]
        assert "alpha" in registry.distinct_source_names()
        assert "beta" in registry.distinct_source_names()
```

- [ ] **Step 8.2: Run the failing test**

```bash
PYTHONPATH=. pytest tests/test_multi_source_flow.int.test.py -v
```

Expected: failure (most likely an assertion that `alpha::` prefix is missing because the flow hasn't been refactored yet).

- [ ] **Step 8.3: Refactor `index_vault_flow`**

Open `flow_index_vault.py`. Find `index_vault_flow` at line 720. Replace the scan invocation area (around lines 726–870) so the body iterates sources. Key transformations:

Near the top of the function, after `config = load_config(config_path)`, build the sources list:
```python
    from sources import build_source

    # Build all configured sources. The config loader has already synthesized
    # a single filesystem source from legacy documents_root: if needed.
    sources_cfg = config.get("sources", [])
    if not sources_cfg:
        raise ValueError("No sources configured. Set 'sources:' or legacy 'documents_root:' in config.yaml")

    pdf_cfg = config.get("pdf", {})
    all_sources = [
        build_source(s, registry=doc_id_store, pdf_config=pdf_cfg)
        for s in sources_cfg
    ]
```

(Note: `doc_id_store` must be built before this point — it already is at line 738.)

For each filesystem source, inject the OCR provider after it's built:
```python
    for src in all_sources:
        if hasattr(src, "set_ocr_provider"):
            src.set_ocr_provider(ocr_provider)
```

Replace the single-scan call (line 860) with a per-source loop that namespaces records:
```python
    all_records: list[dict] = []
    for src in all_sources:
        for rec in src.scan():
            namespaced_id = f"{src.name}::{rec.doc_id}"
            all_records.append({
                "doc_id": namespaced_id,
                "rel_path": rec.natural_key,
                "abs_path": rec.metadata.get("abs_path", rec.natural_key),
                "mtime": rec.mtime,
                "size": rec.size,
                "ext": rec.metadata.get("ext", ""),
                "source_type": rec.source_type,
                "source_name": src.name,
                "_record": rec,  # Keep the original for extract()
            })
```

The downstream `diff_index_task` call gets `all_records` instead of `scanned` — it doesn't care about the namespacing, it just diffs by doc_id + mtime. When the extract step runs, use the stored `_record`:
```python
    by_name = {s.name: s for s in all_sources}
    # ... in the extract loop, for each record in new + updated:
    src = by_name[record["source_name"]]
    result = src.extract(record["_record"])
    # ... rest of the chunk/embed/upsert path unchanged
```

Important: source_name MUST be added to every TextNode's metadata at upsert time. Look for `metadata=` dict construction in the flow (around line 435 and line 462) and add:
```python
"source_name": record["source_name"],
```

- [ ] **Step 8.4: Run `test_integration.py::TestFullPipelineWithEnrichment` to catch single-source regressions**

```bash
PYTHONPATH=. DOCUMENTS_ROOT=/home/danpark/projects/RAG-in-a-Box/test_vault INDEX_ROOT=/tmp/rag-test-index pytest tests/test_integration.py::TestFullPipelineWithEnrichment -v
```

Expected: all 9 tests pass — the backward-compat shim feeds the single-source path through the new multi-source loop with one source named "documents".

- [ ] **Step 8.5: Run the new multi-source test**

```bash
PYTHONPATH=. pytest tests/test_multi_source_flow.int.test.py -v
```

Expected: 3 passed.

- [ ] **Step 8.6: Run the full live suite**

```bash
PYTHONPATH=. DOCUMENTS_ROOT=/home/danpark/projects/RAG-in-a-Box/test_vault INDEX_ROOT=/tmp/rag-test-index pytest -m "live" -q --tb=line
```

Expected: 73 passed (70 previous + 3 new), 0 skipped, 0 failed. Coverage does not regress.

- [ ] **Step 8.7: Run the full non-live suite**

```bash
PYTHONPATH=. pytest -q -m "not live"
```

Expected: 530+ passed (plus new tests from tasks 1-7).

- [ ] **Step 8.8: Commit**

```bash
git add flow_index_vault.py tests/test_multi_source_flow.int.test.py
git commit -m "Refactor index_vault_flow to iterate over Sources

The flow now builds a list of Sources from config['sources'] and
loops over them. Each record's doc_id is namespaced as
'{source_name}::{doc_id}' before diff/upsert, so two sources can't
collide even if they happen to produce the same underlying ID.

source_name is written to every chunk's metadata at upsert time for
LanceDB where-clause filtering.

Legacy single-source configs flow through unchanged: the config shim
(Task 3) synthesizes a single source named 'documents', and the
TestFullPipelineWithEnrichment integration suite proves behavior
parity."
```

---

## Task 9: LanceDB `source_name` filter support

**Why:** With `source_name` now written to every chunk's metadata, hybrid search needs to accept it as a filter so clients can say "search comm_messages only."

**Files:**
- Modify: `lancedb_store.py` (the `_build_where_clause` method around line 146–175)
- Modify: `tests/test_store.py` or `tests/test_search.py` — add a filter test

### Steps

- [ ] **Step 9.1: Write the failing test**

Append to `tests/test_store.py`:
```python
class TestSourceNameFilter:
    """Search honors source_name metadata filter."""

    def test_where_clause_includes_source_name(self, tmp_path):
        from lancedb_store import LanceDBStore

        store = LanceDBStore(str(tmp_path / "idx"), "chunks")
        where = store._build_where_clause(source_name="comm_messages")
        assert "source_name" in where
        assert "comm_messages" in where
```

- [ ] **Step 9.2: Run the failing test**

```bash
PYTHONPATH=. pytest tests/test_store.py::TestSourceNameFilter -v
```

Expected: `TypeError: _build_where_clause() got unexpected keyword argument 'source_name'`.

- [ ] **Step 9.3: Add `source_name` to `_build_where_clause`**

In `lancedb_store.py`, find the signature around line 146. Add `source_name: str | None = None` to the kwargs. In the body (around line 165 where other metadata filters are built), add:
```python
        if source_name:
            parts.append(f"lower(metadata.source_name) = '{self._sql_escape(source_name.lower())}'")
```

Also update the outer `vector_search` / `keyword_search` / `hybrid_search` methods (wherever they forward kwargs to `_build_where_clause`) to pass `source_name` through.

- [ ] **Step 9.4: Run the test**

```bash
PYTHONPATH=. pytest tests/test_store.py::TestSourceNameFilter -v
```

Expected: 1 passed.

- [ ] **Step 9.5: Run the full non-live suite**

```bash
PYTHONPATH=. pytest -q -m "not live"
```

Expected: no regressions.

- [ ] **Step 9.6: Commit**

```bash
git add lancedb_store.py tests/test_store.py
git commit -m "Add source_name filter to LanceDBStore._build_where_clause

Hybrid search can now scope results to a single source by passing
source_name='comm_messages' (or any other registered source). Uses
the same case-insensitive metadata match pattern as source_type."
```

---

## Task 10: MCP `file_search` source_name parameter

**Why:** Expose the filter to MCP clients so they can run "search in comm_messages only" queries.

**Files:**
- Modify: `mcp_server.py`
- Modify: `tests/test_mcp_handlers.int.test.py` or `tests/test_mcp_contract.py`

### Steps

- [ ] **Step 10.1: Write the failing test**

Append to `tests/test_mcp_handlers.int.test.py` (the file matches the naming convention there — adapt if different):
```python
class TestSourceNameFilter:
    def test_file_search_accepts_source_name(self, monkeypatch, tmp_path):
        """_file_search_impl accepts a source_name kwarg and validates it."""
        import mcp_server

        # Setup: point _cache at a minimal stub so the call reaches the filter check
        # (Adapt to whatever fixture test_mcp_handlers uses for the cache)
        # ... existing fixture setup ...

        # Valid source_name: no validation error
        response = mcp_server._file_search_impl("test query", source_name="documents")
        assert response.get("error") is not True

    def test_file_search_rejects_unknown_source_name(self, monkeypatch, tmp_path):
        import mcp_server
        # ... fixture setup ...

        response = mcp_server._file_search_impl("test query", source_name="nonexistent")
        assert response["error"] is True
        assert response["code"] == "invalid_source_name"
```

- [ ] **Step 10.2: Run the failing test**

```bash
PYTHONPATH=. pytest tests/test_mcp_handlers.int.test.py::TestSourceNameFilter -v
```

Expected: `TypeError: _file_search_impl() got an unexpected keyword argument 'source_name'`.

- [ ] **Step 10.3: Update `_file_search_impl` signature and add validation**

In `mcp_server.py`, find `_file_search_impl` (around line 152). Add the kwarg:
```python
def _file_search_impl(
    query: str,
    *,
    top_k: int = 10,
    source_type: str | None = None,
    source_name: str | None = None,   # NEW
    # ... existing params ...
) -> dict:
```

After the existing `_validate_source_type` call, add:
```python
    if source_name:
        registry_path = Path(config["index_root"]) / "doc_registry.db"
        if registry_path.exists():
            from doc_id_store import DocIDStore
            reg = DocIDStore(registry_path)
            valid_names = reg.distinct_source_names()
            reg.close()
            if source_name not in valid_names:
                return _error(
                    "invalid_source_name",
                    f"source_name must be one of: {', '.join(sorted(valid_names))}. Got: {source_name!r}.",
                    fix=f"Use file_list_documents to discover which source_names are indexed.",
                )
```

Pass `source_name` through to the store search call:
```python
    hits = store.hybrid_search(
        query=query,
        # ... existing args ...
        source_name=source_name,
    )
```

Finally update the MCP tool schema declaration (search for `"source_type"` in the tool's JSON schema) to add `source_name` as an optional string parameter with a clear description.

- [ ] **Step 10.4: Run the test**

```bash
PYTHONPATH=. pytest tests/test_mcp_handlers.int.test.py::TestSourceNameFilter -v
```

Expected: 2 passed.

- [ ] **Step 10.5: Run `test_mcp_contract.py` to catch schema regressions**

```bash
PYTHONPATH=. pytest tests/test_mcp_contract.py -v
```

Expected: all passed (if a schema assertion requires updating to include source_name, update the fixture expectations).

- [ ] **Step 10.6: Full non-live suite**

```bash
PYTHONPATH=. pytest -q -m "not live"
```

Expected: no regressions.

- [ ] **Step 10.7: Commit**

```bash
git add mcp_server.py tests/test_mcp_handlers.int.test.py tests/test_mcp_contract.py
git commit -m "Add source_name filter to MCP file_search tool

file_search now accepts an optional source_name string. Validation
derives the allowed set at request time from
DocIDStore.distinct_source_names() so new sources auto-appear without
a code edit.

Invalid source_name returns a structured error with a 'fix' hint
pointing at file_list_documents."
```

---

## Task 11: Docker compose + `.env` wiring

**Why:** The container needs `COMM_DATA_STORE_DSN` to reach Postgres at runtime.

**Files:**
- Modify: `docker-compose.yml`
- Modify: `.env` (if using one; otherwise document in README)
- Modify: `config.yaml.example`

### Steps

- [ ] **Step 11.1: Add env var to docker-compose.yml**

Append to the `environment:` list (around line 16):
```yaml
      - COMM_DATA_STORE_DSN=${COMM_DATA_STORE_DSN}
```

- [ ] **Step 11.2: Add env var to `.env`**

Append to `.env`:
```bash
# Comm-Data-Store PostgreSQL connection (for sources.postgres.PostgresSource).
# In dev, password is "change-me" from Comm-Data-Store/.env.example.
# Host must be reachable from the doc-organizer container via host.docker.internal:5433.
COMM_DATA_STORE_DSN=postgresql://comm_data_store:change-me@host.docker.internal:5433/comm_data_store
```

- [ ] **Step 11.3: Verify compose config resolves the variable**

```bash
docker compose config | grep -A1 COMM_DATA_STORE_DSN
```

Expected: `COMM_DATA_STORE_DSN: postgresql://comm_data_store:change-me@host.docker.internal:5433/comm_data_store`.

- [ ] **Step 11.4: Add a commented example to the three `config*.yaml.example` files**

Add at the bottom of each, under an "Optional: multi-source configuration" section:
```yaml
# Multi-source mode (replaces documents_root: when enabled):
#
# sources:
#   - type: filesystem
#     name: documents
#     root: /data/documents
#     scan:
#       include: ["**/*.md", "**/*.pdf", "**/*.docx", ...]
#       exclude: ["99-Archive/**"]
#
#   - type: postgres
#     name: comm_messages
#     dsn: "${COMM_DATA_STORE_DSN}"
#     tables:
#       - source_type: pg_message
#         query: |
#           SELECT m.source, m.source_message_id, m.sent_at, m.updated_at,
#                  m.body AS _text,
#                  c.name AS channel, p.display_name AS sender
#           FROM messages m
#           LEFT JOIN channels c ON c.id = m.channel_id
#           LEFT JOIN participants p ON p.id = m.sender_participant_id
#           WHERE m.body IS NOT NULL AND m.body <> ''
#         id_template: "{source}/{source_message_id}"
#         text_column: _text
#         mtime_column: updated_at
#         metadata_columns: [channel, sender, source, sent_at]
#
#       - source_type: pg_transcript
#         query: |
#           SELECT t.id, t.transcript_text AS _text, t.created_at, t.updated_at, c.source
#           FROM transcripts t JOIN calls c ON c.id = t.call_id
#         id_template: "pg_transcript/{id}"
#         text_column: _text
#         mtime_column: updated_at
#         metadata_columns: [source, created_at]
```

- [ ] **Step 11.5: Commit**

```bash
git add docker-compose.yml .env config.yaml.example config.local.yaml.example config.vps.yaml.example
git commit -m "Wire COMM_DATA_STORE_DSN through compose + .env

Threads the Comm-Data-Store connection string from .env to the
doc-organizer container the same way OPENROUTER_API_KEY is threaded.

Adds a commented-out sources: example to all three config templates
showing filesystem + postgres side by side."
```

**WARNING:** Do not `git add .env` if `.gitignore` excludes it (check first). If excluded, document the variable in README instead and leave `.env` untouched on disk.

---

## Task 12: Enable Comm-Data-Store in production config + live smoke

**Why:** Everything is merged. Time to flip the switch on the real deployment.

**Files:**
- Modify: `config.yaml` (production config, bind-mounted into the running container)

### Steps

- [ ] **Step 12.1: Preview the config change**

Edit `config.yaml` to add the multi-source block. Keep the old `documents_root:` present temporarily — since Task 3 rejects mixing, replace it:
```yaml
# OLD:
# documents_root: "/data/documents"
# scan:
#   include: [...]
#   exclude: [...]

# NEW:
sources:
  - type: filesystem
    name: documents
    root: /data/documents
    scan:
      include: ["**/*.md", "**/*.pdf", "**/*.png", "**/*.jpg", "**/*.jpeg",
                "**/*.docx", "**/*.doc", "**/*.pptx", "**/*.rtf", "**/*.epub",
                "**/*.html", "**/*.htm", "**/*.csv", "**/*.txt", "**/*.xlsx", "**/*.xls"]
      exclude: [".git/**", "**/.DS_Store", "**/node_modules/**", "**/99-Archive/**"]

  - type: postgres
    name: comm_messages
    dsn: "${COMM_DATA_STORE_DSN}"
    tables:
      - source_type: pg_message
        query: |
          SELECT m.source, m.source_message_id, m.sent_at, m.updated_at,
                 m.body AS _text,
                 c.name AS channel, p.display_name AS sender
          FROM messages m
          LEFT JOIN channels c ON c.id = m.channel_id
          LEFT JOIN participants p ON p.id = m.sender_participant_id
          WHERE m.body IS NOT NULL AND m.body <> ''
        id_template: "{source}/{source_message_id}"
        text_column: _text
        mtime_column: updated_at
        metadata_columns: [channel, sender, source, sent_at]

      - source_type: pg_transcript
        query: |
          SELECT t.id, t.transcript_text AS _text, t.created_at, t.updated_at, c.source
          FROM transcripts t JOIN calls c ON c.id = t.call_id
        id_template: "pg_transcript/{id}"
        text_column: _text
        mtime_column: updated_at
        metadata_columns: [source, created_at]
```

Keep all other top-level sections (pdf, ocr, embeddings, chunking, enrichment, search, lancedb, logging, mcp) unchanged.

- [ ] **Step 12.2: Rebuild the container**

```bash
docker compose build doc-organizer 2>&1 | tail -5
```

Expected: `Image doc-organizer:latest Built`.

- [ ] **Step 12.3: Recreate and bring up**

```bash
docker compose up -d doc-organizer
```

- [ ] **Step 12.4: Verify startup logs show both sources**

```bash
docker logs doc-organizer --tail 30
```

Expected: `Auth enabled: API_KEY set (...)`, then `StreamableHTTP session manager started`, `Application startup complete.`, no tracebacks. The flow hasn't run yet so no source-specific logs yet.

- [ ] **Step 12.5: Trigger a full index via MCP**

```bash
# Via curl against the MCP endpoint, OR via the claude-code CLI that's already
# connected. Whichever is easier in your session — the goal is to call
# file_index_update.
```

Expected: indexer runs, logs show scanning of both filesystem and postgres sources, chunks are upserted to LanceDB with source_name metadata, and the run completes without errors.

- [ ] **Step 12.6: Query via MCP to verify comm_messages is searchable**

Run a `file_search` with `source_name="comm_messages"` for a term you know exists in the Comm-Data-Store data (e.g. a client name). Expected: results, all with `source_name: "comm_messages"` in metadata.

- [ ] **Step 12.7: Query without source_name filter (cross-source search)**

Run a `file_search` without any source_name filter. Expected: results mixing filesystem docs and comm_messages docs, ranked by relevance.

- [ ] **Step 12.8: Verify registry has both sources**

```bash
docker exec doc-organizer python -c "
from doc_id_store import DocIDStore
r = DocIDStore('/data/index/doc_registry.db')
print('sources:', r.distinct_source_names())
print('total docs:', r.count())
"
```

Expected: `sources: {'documents', 'comm_messages'}` and a total count > existing docs count.

- [ ] **Step 12.9: Tail logs for 5 minutes, verify no warnings or tracebacks**

```bash
docker logs doc-organizer --since 5m 2>&1 | grep -E "WARNING|ERROR|Traceback"
```

Expected: zero matches.

- [ ] **Step 12.10: Commit config.yaml change**

```bash
git add config.yaml
git commit -m "Enable Comm-Data-Store indexing via postgres source

Switches config.yaml from legacy documents_root: to the sources:
shape, adding a second postgres source targeting
comm-data-store-postgres-1. messages and transcripts are indexed;
calls and channels appear as metadata on the relevant chunks."
```

- [ ] **Step 12.11: Push**

```bash
git push origin main
```

---

## Rollback Plan

If any task reveals a blocking issue:

- **Tasks 1–7**: Pure additions. Safe to revert individual commits without touching anything else.
- **Task 8**: The flow refactor is the only load-bearing change. If it breaks, `git revert` that commit — the legacy single-source path still works via the config shim, and the earlier tasks (Source types, PostgresSource, dispatcher) remain as inert but valid code.
- **Task 12 (deployment)**: If the live rollout fails, revert `config.yaml` to `documents_root:` and `docker compose up -d doc-organizer`. Because the config shim in Task 3 handles both shapes, this is a one-line rollback.
- **DocIDStore migration (Task 2)**: The migration is additive — `ALTER TABLE ADD COLUMN` is not reversible in SQLite, but the new column has a default, so old code that ignores the column keeps working. No rollback needed.

---

## Verification Checklist (run after Task 12)

- [ ] `docker logs doc-organizer` shows `Auth enabled: API_KEY set (...)` and zero tracebacks.
- [ ] `docker exec doc-organizer python -c "from doc_id_store import DocIDStore; print(DocIDStore('/data/index/doc_registry.db').distinct_source_names())"` returns `{'documents', 'comm_messages'}`.
- [ ] `file_search` with `source_name="comm_messages"` returns non-empty results.
- [ ] `file_search` without `source_name` returns results from both sources interleaved.
- [ ] `file_list_documents` shows both `documents/...` and `comm_messages/...` entries.
- [ ] DeepSeek-OCR2 probe at `192.168.68.70:8790/health` still returns 200 (nothing about this change should affect it, but verify).
- [ ] The 3-hour soak-test loop (`b12db931`) keeps reporting OK.
- [ ] `pytest -q` (non-live) still shows 530+ passed, 0 failed.
- [ ] `pytest -q -m "live"` shows 73+ passed (70 pre-existing + 3 new from multi-source test), 0 failed, 0 skipped.

---

## Open Questions / Things to Decide Later

1. **Chunk enrichment over huge Postgres rows.** If a call transcript is 100 kB, enrichment currently summarizes the first `max_input_chars: 20000` characters. Transcripts may benefit from a specialized chunk-first-then-enrich strategy. Revisit after seeing real data shapes.
2. **Secret management for DSN in prod.** `.env` is fine for dev; in production (Render/Fly/Railway) set `COMM_DATA_STORE_DSN` as a platform secret, not in `.env`. Document this in the deployment section of README.
3. **LanceDB backfill of `source_name`.** Existing chunks from before this change have no `source_name` metadata field. The `_evolve_metadata_schema()` path handles schema evolution but defaults missing fields to empty string. Re-indexing via `file_index_update` will naturally backfill as mtimes shift; an explicit backfill task can be added later if needed.
4. **Future: add `slack_messages` source.** When ready, implement `sources/slack.py` following the `PostgresSource` template. Zero changes expected to the flow, dispatcher, or tests infrastructure.
