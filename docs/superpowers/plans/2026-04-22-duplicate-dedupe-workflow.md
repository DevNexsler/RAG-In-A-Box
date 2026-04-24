# Exact Duplicate Dedupe Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent exact duplicate content from creating duplicate indexed documents, while archiving duplicate payloads/records and attaching duplicate provenance to the canonical indexed document.

**Architecture:** Reuse the existing SQLite `doc_registry` as the exact-duplicate ledger by adding size/hash/dedupe columns and indexed lookup. Detect duplicates before expensive extraction/enrichment/embedding work, archive duplicate payloads non-destructively, skip duplicate indexing, and attach duplicate provenance to canonical LanceDB metadata plus an MCP duplicate-check tool.

**Tech Stack:** Python 3, SQLite (`doc_registry.db`), LanceDB, Prefect flow/tasks, filesystem + PostgreSQL sources, MCP server, pytest

---

## File Map

### Existing files to modify

- `doc_id_store.py`
  - Extend `doc_registry` schema for dedupe state.
  - Add indexed lookup helpers, duplicate-marking APIs, and atomic duplicate-claim APIs.
- `flow_index_vault.py`
  - Insert dedupe check before extraction/enrichment/upsert.
  - Handle empty payload early.
  - Archive duplicates and skip indexing.
  - Update canonical metadata/provenance.
- `lancedb_store.py`
  - Add helper to read/update canonical duplicate metadata.
  - Ensure `dup_*` metadata fields persist and surface through read paths.
- `mcp_server.py`
  - Add MCP duplicate-check tool.
- `core/config.py`
  - Load and validate new `dedupe` config block.
- `requirements.txt`
  - Add `blake3` runtime dependency if not already present.
- `config.yaml.example`
  - Document new dedupe config.
- `config.local.yaml.example`
  - Document new dedupe config.
- `config.vps.yaml.example`
  - Document new dedupe config.
- `config_test.yaml.example`
  - Document new dedupe config.

### New files to create

- `core/dedupe.py`
  - Streaming BLAKE3 hashing helpers.
  - Canonical duplicate-decision helpers.
  - Archive path builders.
- `tests/test_dedupe_registry.py`
  - Unit tests for registry schema, lookup, and duplicate marking.
- `tests/test_dedupe_hashing.py`
  - Unit tests for exact hashing and zero-byte behavior.
- `tests/test_dedupe_config.py`
  - Unit tests for dedupe config validation rules.
- `tests/test_dedupe_archival.py`
  - Unit tests for filesystem and PostgreSQL duplicate archival.
- `tests/test_duplicate_check_mcp.py`
  - MCP contract tests for duplicate-check tool.
- `tests/test_duplicate_indexing.int.test.py`
  - Integration tests for first-seen canonical flow and duplicate skip behavior.

## Task 1: Add Dedupe Schema to `doc_registry`

**Files:**
- Modify: `doc_id_store.py`
- Test: `tests/test_dedupe_registry.py`

- [ ] **Step 1: Write failing schema test**

```python
def test_registry_schema_includes_dedupe_columns(tmp_path):
    store = DocIDStore(tmp_path / "doc_registry.db")
    cols = {
        row[1]: row[2]
        for row in store._conn.execute("PRAGMA table_info(doc_registry)").fetchall()
    }
    assert "size_bytes" in cols
    assert "content_hash" in cols
    assert "hash_algo" in cols
    assert "dedupe_status" in cols
    assert "canonical_doc_id" in cols
    assert "archive_path" in cols
    assert "duplicate_reason" in cols
    assert "first_seen_at" in cols
    assert "last_seen_at" in cols
    indexes = store._conn.execute("PRAGMA index_list(doc_registry)").fetchall()
    assert any(row[1] == "idx_doc_registry_size_hash" for row in indexes)
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_dedupe_registry.py::test_registry_schema_includes_dedupe_columns -v`
Expected: FAIL because columns do not exist yet.

- [ ] **Step 3: Add schema migration code**

Implement idempotent `ALTER TABLE` additions in `DocIDStore._init_schema()`.
Also add index creation for `(size_bytes, content_hash)`.
Required columns from spec:
- `size_bytes`
- `content_hash`
- `hash_algo`
- `dedupe_status`
- `canonical_doc_id`
- `archive_path`
- `duplicate_reason`
- `first_seen_at`
- `last_seen_at`

Migration/write-path requirements:
- backfill `first_seen_at` from `created` for existing rows where possible
- every canonical or duplicate identity update must set `last_seen_at`
- duplicate-marking path must preserve original `first_seen_at`

- [ ] **Step 4: Write failing lookup/marking tests**

```python
def test_find_exact_duplicate_returns_first_seen(tmp_path):
    store = DocIDStore(tmp_path / "doc_registry.db")
    store.register("documents::00001", "a.pdf", source_name="documents")
    store.update_dedupe_identity(
        "documents::00001",
        size_bytes=123,
        content_hash=b"\x01" * 32,
        hash_algo="blake3",
        dedupe_status="canonical",
        canonical_doc_id=None,
    )
    hit = store.find_canonical_by_exact_hash(123, b"\x01" * 32)
    assert hit["doc_id"] == "documents::00001"
```

Also cover:
- `first_seen_at` backfilled from `created` for existing rows
- `last_seen_at` updates on canonical/duplicate identity writes
- duplicate-marking preserves original `first_seen_at`

- [ ] **Step 5: Run new tests to verify failure**

Run: `pytest tests/test_dedupe_registry.py -v`
Expected: FAIL because helper APIs do not exist yet.

- [ ] **Step 6: Implement registry APIs**

Add methods with narrow responsibility:
- `update_dedupe_identity(...)`
- `find_canonical_by_exact_hash(...)`
- `mark_duplicate(...)`
- `duplicate_refs_for_canonical(...)`
- `claim_canonical_by_exact_hash(...)`

The registry-side claim path must support atomic first-seen-wins behavior so concurrent ingest cannot create two canonicals for one exact hash.

- [ ] **Step 7: Re-run registry tests**

Run: `pytest tests/test_dedupe_registry.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add doc_id_store.py tests/test_dedupe_registry.py
git commit -m "feat: add registry-backed duplicate metadata"
```

## Task 2: Add Streaming Exact Hashing and Archive Helpers

**Files:**
- Create: `core/dedupe.py`
- Modify: `core/config.py`
- Modify: `requirements.txt`
- Test: `tests/test_dedupe_hashing.py`
- Test: `tests/test_dedupe_config.py`
- Test: `tests/test_dedupe_archival.py`

- [ ] **Step 1: Write failing hash and archival tests**

```python
def test_hash_file_blake3_streaming(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello")
    result = compute_file_identity(p)
    assert result.size_bytes == 5
    assert result.hash_algo == "blake3"
    assert len(result.content_hash) == 32

def test_zero_byte_file_flagged(tmp_path):
    p = tmp_path / "empty.pdf"
    p.write_bytes(b"")
    result = compute_file_identity(p)
    assert result.size_bytes == 0

def test_empty_text_payload_flagged():
    result = compute_text_identity("")
    assert result.size_bytes == 0

def test_postgres_duplicate_snapshot_written_as_json(tmp_path):
    archive = tmp_path / "archive"
    result = archive_duplicate_record(...)
    assert result.archive_path.endswith(
        "/postgres/comm_messages/documents::00001/message-123.json"
    )
    payload = json.loads(Path(result.archive_path).read_text())
    assert payload["source_name"] == "comm_messages"

def test_filesystem_archive_path_keeps_timestamp_and_rel_path(tmp_path):
    source_file = tmp_path / "1-Projects" / "a.pdf"
    result = archive_duplicate_file(...)
    assert "/filesystem/documents::00001/" in result.archive_path
    assert Path(result.archive_path).name.endswith("__1-Projects%2Fa.pdf")
    assert source_file.exists()

def test_dedupe_config_rejects_non_exact_mode():
    with pytest.raises(ValueError):
        load_app_config({"dedupe": {"enabled": True, "mode": "fuzzy"}})

def test_dedupe_config_rejects_non_blake3_hash_algo():
    with pytest.raises(ValueError):
        load_app_config({"dedupe": {"enabled": True, "hash_algo": "sha256"}})

def test_dedupe_config_rejects_disabled_canonical_metadata():
    with pytest.raises(ValueError):
        load_app_config({"dedupe": {"enabled": True, "update_canonical_metadata": False}})

def test_dedupe_config_rejects_skip_duplicate_indexing_false():
    with pytest.raises(ValueError):
        load_app_config({"dedupe": {"enabled": True, "skip_duplicate_indexing": False}})

def test_dedupe_config_allows_archive_duplicates_false():
    cfg = load_app_config({"dedupe": {"enabled": True, "archive_duplicates": False}})
    assert cfg["dedupe"]["archive_duplicates"] is False
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_dedupe_hashing.py tests/test_dedupe_config.py tests/test_dedupe_archival.py -v`
Expected: FAIL because helpers do not exist.

- [ ] **Step 3: Implement `core/dedupe.py`**

Implement:
- dataclass for exact identity
- `compute_file_identity(path)`
- `compute_text_identity(text)`
- `is_zero_payload(...)`
- archive path builders
- `archive_duplicate_file(...)`
- `archive_duplicate_record(...)`

Add explicit runtime dependency for `blake3` in `requirements.txt` as part of this task before code relies on it.

Archive requirements from spec:
- filesystem path shape: `.../filesystem/<canonical_doc_id>/<timestamp>__<encoded_original_rel_path>`
- PostgreSQL path shape: `.../postgres/<source_name>/<canonical_doc_id>/<natural_key>.json`
- return archive path for registry metadata
- preserve original rel path or natural key in archived payload metadata
- use a reversible path-safe encoding for filesystem `original_rel_path` such as URL-quoting with `safe=""`
- filesystem archival must be non-destructive: copy/snapshot duplicate into archive, keep original source file in place

- [ ] **Step 4: Load dedupe config**

Add config defaults/validation in `core/config.py`:

```python
dedupe = cfg.setdefault("dedupe", {})
dedupe.setdefault("enabled", False)
dedupe.setdefault("mode", "exact")
dedupe.setdefault("hash_algo", "blake3")
dedupe.setdefault("archive_root", str(Path(cfg["index_root"]) / "duplicates"))
dedupe.setdefault("archive_duplicates", True)
dedupe.setdefault("update_canonical_metadata", True)
dedupe.setdefault("skip_duplicate_indexing", True)
```

Behavior requirements:
- `dedupe.enabled = false` bypasses exact dedupe path and preserves normal indexing behavior
- reject unsupported `dedupe.mode != "exact"` during config validation
- reject unsupported `dedupe.hash_algo != "blake3"` during config validation
- reject unsupported `dedupe.update_canonical_metadata = false` during config validation
- `dedupe.skip_duplicate_indexing = false` is not supported in this plan; reject it during config validation instead of silently ignoring it

- [ ] **Step 5: Re-run hashing and archival tests**

Run: `pytest tests/test_dedupe_hashing.py tests/test_dedupe_config.py tests/test_dedupe_archival.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add core/dedupe.py core/config.py requirements.txt tests/test_dedupe_hashing.py tests/test_dedupe_config.py tests/test_dedupe_archival.py
git commit -m "feat: add exact duplicate hashing and archive helpers"
```

## Task 3: Attach Duplicate Provenance to Canonical LanceDB Metadata

**Files:**
- Modify: `lancedb_store.py`
- Test: `tests/test_duplicate_indexing.int.test.py`

- [ ] **Step 1: Write failing provenance tests**

Cover:
- canonical metadata includes `dup_count`
- canonical metadata includes `dup_sources`
- canonical metadata includes `dup_locations`
- canonical metadata includes `dup_archive_paths`
- canonical metadata includes `dup_natural_keys`
- stale registry canonical with missing LanceDB doc raises integrity failure

Example:

```python
def test_canonical_metadata_tracks_duplicate_locations(indexed_store):
    hit = indexed_store["store"].list_recent_docs(limit=10)[0]
    assert hit["dup_count"] == 1
    assert hit["dup_sources"] == ["comm_messages"]
    assert hit["dup_locations"] == ["1-Projects/a.pdf"]
    assert hit["dup_archive_paths"]
    assert hit["dup_natural_keys"] == ["comm_messages:123"]
```

- [ ] **Step 2: Run targeted provenance tests to verify failure**

Run: `pytest tests/test_duplicate_indexing.int.test.py::test_canonical_metadata_tracks_duplicate_locations -v`
Expected: FAIL.

- [ ] **Step 3: Add canonical metadata update helper**

Implement helper in `lancedb_store.py` to:
- fetch all chunks for canonical doc
- merge duplicate provenance metadata
- re-upsert canonical chunks with updated metadata
- make `dup_*` fields visible through current recent/document-listing read paths used by clients

Use compact metadata fields from spec:
- `dup_count`
- `dup_sources`
- `dup_locations`
- `dup_archive_paths`
- `dup_natural_keys`

Integrity guard requirement:
- before attaching duplicate provenance, verify canonical doc actually exists in LanceDB
- if SQLite canonical row exists but LanceDB doc is missing, surface integrity failure instead of silently attaching duplicate to a broken canonical

- [ ] **Step 4: Re-run provenance tests**

Run: `pytest tests/test_duplicate_indexing.int.test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lancedb_store.py tests/test_duplicate_indexing.int.test.py
git commit -m "feat: attach duplicate provenance to canonical docs"
```

## Task 4: Short-Circuit Empty Payloads and Detect Duplicates Before Indexing

**Files:**
- Modify: `flow_index_vault.py`
- Test: `tests/test_duplicate_indexing.int.test.py`

- [ ] **Step 1: Write failing integration tests**

Cover:
- zero-byte filesystem file gets skipped before parser
- empty PostgreSQL/raw-text record gets skipped before enrichment/indexing
- non-duplicate filesystem file still indexes normally
- non-duplicate PostgreSQL/raw-text record still indexes normally
- non-duplicate mixed-source records still both index normally
- identical second filesystem file gets archived and skipped
- identical second PostgreSQL text record gets archived and skipped
- identical filesystem vs PostgreSQL payload dedupes across sources
- first-seen doc remains canonical
- duplicate branch updates canonical `dup_*` metadata
- `dedupe.enabled = false` bypasses duplicate suppression
- archive-disabled duplicate still skipped and logged
- archive failure logged without destructive source action

Example:

```python
def test_exact_duplicate_file_is_not_indexed_twice(vault_and_index):
    first = make_pdf_bytes(b"same payload")
    second = make_pdf_bytes(b"same payload")
    # create two source files
    # run flow
    # assert one canonical doc_id in LanceDB
    # assert duplicate row marked in registry
    # assert duplicate archive path stored
    # assert canonical dup_* metadata updated
```

- [ ] **Step 2: Run failing integration tests**

Run: `pytest tests/test_duplicate_indexing.int.test.py::test_exact_duplicate_file_is_not_indexed_twice -v`
Expected: FAIL.

- [ ] **Step 3: Add pre-index dedupe decision in flow**

In `flow_index_vault.py`, before extraction:
- materialize candidate payload for hashing
- reject zero-byte filesystem payloads early
- reject empty PostgreSQL/raw-text payloads early
- compute exact identity for current source record
- use registry atomic claim/lookup path for exact match
- if duplicate:
  - archive payload/record using `core/dedupe.py` helpers
  - mark duplicate in registry
  - append audit event
  - update canonical metadata via `lancedb_store.py` helper
  - return without normal indexing

Archive policy requirements in flow:
- if `dedupe.enabled` is false, bypass exact dedupe and continue normal indexing path
- if `dedupe.archive_duplicates` is false, still mark duplicate, log it, update canonical metadata, and skip indexing
- if archive write fails, log explicit archive-failure event and skip duplicate without destructive source action

- [ ] **Step 4: Preserve canonical-first semantics**

Make duplicate check deterministic:
- first seen canonical
- later exact match duplicate
- no source-type override

- [ ] **Step 5: Re-run targeted integration tests**

Run: `pytest tests/test_duplicate_indexing.int.test.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add flow_index_vault.py tests/test_duplicate_indexing.int.test.py
git commit -m "feat: skip exact duplicates during indexing"
```

## Task 5: Handle Legacy Registry Rows Without Hashes

**Files:**
- Modify: `flow_index_vault.py`
- Modify: `doc_id_store.py`
- Test: `tests/test_duplicate_indexing.int.test.py`

- [ ] **Step 1: Write failing lazy-fill test**

```python
def test_existing_registry_row_without_hash_is_backfilled_when_reencountered(...):
    # seed doc_registry row with null hash fields
    # re-scan same record
    # assert size/hash fields get populated
```

- [ ] **Step 2: Run failing lazy-fill test**

Run: `pytest tests/test_duplicate_indexing.int.test.py::test_existing_registry_row_without_hash_is_backfilled_when_reencountered -v`
Expected: FAIL.

- [ ] **Step 3: Implement chosen migration strategy**

Chosen strategy for this plan:
- lazy fill on first re-encounter of existing source record
- do not block new ingest on full backfill

Implementation requirements:
- if registry row exists but hash fields are null, compute and persist identity during normal ingest
- make this path work for both filesystem and PostgreSQL records

- [ ] **Step 4: Re-run targeted lazy-fill test**

Run: `pytest tests/test_duplicate_indexing.int.test.py::test_existing_registry_row_without_hash_is_backfilled_when_reencountered -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add flow_index_vault.py doc_id_store.py tests/test_duplicate_indexing.int.test.py
git commit -m "feat: lazily backfill duplicate hashes"
```

## Task 6: Expose MCP Duplicate Check Tool

**Files:**
- Modify: `mcp_server.py`
- Test: `tests/test_duplicate_check_mcp.py`

- [ ] **Step 1: Write failing MCP contract tests**

```python
def test_duplicate_check_returns_canonical_match(monkeypatch):
    result = mcp_server._file_duplicate_check_impl(...)
    assert result["duplicate"] is True
    assert result["canonical_doc_id"] == "documents::00001"
    assert result["canonical_rel_path"] == "a.pdf"
    assert result["canonical_source_name"] == "documents"
    assert "archive_path" in result
    assert "first_seen_at" in result
```

Also cover:
- filesystem path input
- raw text/content input
- PostgreSQL-backed source metadata input
- optional precomputed hash input
- non-duplicate response payload

- [ ] **Step 2: Run MCP test to verify failure**

Run: `pytest tests/test_duplicate_check_mcp.py -v`
Expected: FAIL because tool does not exist.

- [ ] **Step 3: Implement internal handler**

Add `_file_duplicate_check_impl(...)` in `mcp_server.py` that:
- accepts file path or raw text/content
- optionally accepts precomputed hash
- supports PostgreSQL/source metadata inputs for non-file-backed callers
- queries `doc_registry`
- returns:
  - `duplicate`
  - `canonical_doc_id`
  - `canonical_rel_path`
  - `canonical_source_name`
  - `archive_path`
  - `first_seen_at`
  - duplicate provenance summary

- [ ] **Step 4: Expose MCP tool wrapper**

Add public tool docstring and tool registration for:
- `file_duplicate_check`

- [ ] **Step 5: Re-run MCP tests**

Run: `pytest tests/test_duplicate_check_mcp.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mcp_server.py tests/test_duplicate_check_mcp.py
git commit -m "feat: expose duplicate check MCP tool"
```

## Task 7: Document Config and Operator Workflow

**Files:**
- Modify: `config.yaml.example`
- Modify: `config.local.yaml.example`
- Modify: `config.vps.yaml.example`
- Modify: `config_test.yaml.example`

- [ ] **Step 1: Add dedupe config examples**

```yaml
dedupe:
  enabled: true
  mode: "exact"
  hash_algo: "blake3"
  archive_root: "/data/index/duplicates"
  archive_duplicates: true
  update_canonical_metadata: true
  skip_duplicate_indexing: true
```

- [ ] **Step 2: Add concise operator notes**

Document:
- first-seen wins
- duplicates archived, not deleted
- duplicate-check MCP tool
- zero-byte payloads skipped early

- [ ] **Step 3: Run config/example sanity checks**

Run:
- `rg -n "^dedupe:|archive_root|update_canonical_metadata" config*.example`

Expected: new config/tool references present.

- [ ] **Step 4: Commit**

```bash
git add config.yaml.example config.local.yaml.example config.vps.yaml.example config_test.yaml.example
git commit -m "docs: add duplicate dedupe configuration"
```

## Task 8: Final Verification

**Files:**
- Modify as needed based on failed verification only

- [ ] **Step 1: Run focused unit tests**

Run:
- `pytest tests/test_dedupe_registry.py tests/test_dedupe_hashing.py tests/test_dedupe_archival.py tests/test_duplicate_check_mcp.py -v`
- `pytest tests/test_dedupe_config.py -v`

Expected: all PASS.

- [ ] **Step 2: Run integration tests**

Run:
- `pytest tests/test_duplicate_indexing.int.test.py -v`

Expected: PASS.

- [ ] **Step 3: Run adjacent regression tests**

Run:
- `pytest tests/test_doc_id_store.py tests/test_doc_id_integration.int.test.py tests/test_mcp_contract.py tests/test_indexer_concurrency.int.test.py -v`

Expected: PASS.

- [ ] **Step 4: Run change-scope check**

Run:
- `git status -sb`
- `python3 -m compileall core doc_id_store.py flow_index_vault.py lancedb_store.py mcp_server.py`

Expected:
- no unexpected files
- compile step exit 0

- [ ] **Step 5: Detect changed-scope before finishing**

Run:
- `gitnexus_detect_changes(scope="all")`

Expected:
- changed files only in planned dedupe surface

- [ ] **Step 6: Final commit**

```bash
git add doc_id_store.py flow_index_vault.py lancedb_store.py mcp_server.py core/dedupe.py core/config.py config*.example tests/
git commit -m "feat: add exact duplicate dedupe workflow"
```

## Notes for Implementation

- Keep duplicate detection exact-only. Do not add fuzzy matching in this plan.
- Keep source systems non-destructive. Filesystem duplicates may be archived; PostgreSQL duplicates must only be snapshotted.
- Do not introduce a new database. Reuse `doc_registry.db`.
- Keep lookup disk-backed through SQLite indexes. No global in-memory hash map.
- Fix zero-byte file handling as part of duplicate pipeline because current failures show real empty-file input.
- Existing `doc_registry` rows will not all have hashes initially. Implementation must choose and test one path:
  - lazy hash fill on first re-encounter, or
  - explicit backfill helper/maintenance command
  This must not block new ingest.
