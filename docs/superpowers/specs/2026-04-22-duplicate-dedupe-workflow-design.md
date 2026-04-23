# Exact Duplicate Dedupe Workflow Design

Date: 2026-04-22
Status: Draft for review

## Summary

Add exact-duplicate detection to indexing across all sources.

Core behavior:
- First-seen record wins and becomes canonical.
- Exact duplicates are not indexed as separate LanceDB documents.
- Duplicate payloads/records are archived non-destructively.
- Canonical document metadata records where duplicate copies were found.
- Duplicate checks must stay low-compute, low-latency, and disk-backed.

This design reuses the existing SQLite `doc_registry` as the duplicate ledger.
No new database is introduced.

## Goals

- Prevent exact duplicate content from creating duplicate indexed documents.
- Keep duplicate detection cheap enough to run during normal ingest.
- Preserve traceability for where duplicate copies came from.
- Support all sources:
  - filesystem
  - PostgreSQL-backed sources (`comm_messages`, `sor`, future sources)
- Expose duplicate checking via MCP for external software.

## Non-Goals

- Near-duplicate or fuzzy duplicate detection.
- Content similarity across OCR noise, formatting changes, or metadata-only changes.
- Destructive cleanup of source systems.
- Automatic deletion of duplicate PostgreSQL rows.

## Design Principles

- Exact-only: dedupe based on exact content hash equality.
- First-seen wins: canonical record never changes unless manually repaired later.
- Non-destructive: duplicates are skipped from index, not deleted from source.
- Existing storage first: reuse current SQLite registry and LanceDB metadata.
- Disk-backed, not RAM-backed: no in-memory global hash table required.

## Why Exact Hashing

Exact duplicate detection is cheap and reliable.

Recommended fingerprint:
- `BLAKE3`

Reasoning:
- Fast enough for inline ingest checks.
- Streaming-friendly for large files.
- Stable fixed-size output.
- Much cheaper than OCR, embedding, or enrichment.

Comparison approach:
1. Check `size_bytes == 0` early and reject empty payloads immediately.
2. Compute `BLAKE3` for new or changed record payload.
3. Query indexed SQLite fields for matching `(size_bytes, content_hash)`.
4. If match exists, treat new record as duplicate.

## Storage Model

Use existing `doc_registry` table.

Add columns:
- `size_bytes INTEGER`
- `content_hash BLOB`
- `hash_algo TEXT`
- `dedupe_status TEXT NOT NULL DEFAULT 'canonical'`
- `canonical_doc_id TEXT`
- `archive_path TEXT`
- `duplicate_reason TEXT`
- `first_seen_at REAL`
- `last_seen_at REAL`

Recommended values:
- `hash_algo = 'blake3'`
- `dedupe_status IN ('canonical', 'duplicate')`

Add index:
- `(size_bytes, content_hash)`

Optional future index:
- `(canonical_doc_id)`

Rationale:
- One row per observed document identity still works.
- Duplicate state stays next to existing doc registration state.
- Indexed equality lookup avoids full scans.

## Canonical vs Duplicate Semantics

Canonical record:
- first-seen record for a given exact payload
- keeps existing LanceDB document/chunks
- receives duplicate-source metadata updates

Duplicate record:
- exact payload match to existing canonical record
- not embedded
- not enriched
- not inserted as a new LanceDB document
- archived and linked to canonical record in registry/audit trail

Tie-breaking:
- first-seen wins
- no source-type override

## Ingest Workflow

### Shared flow

For each source record:
1. Materialize candidate payload for hashing.
2. Reject empty payload immediately if zero bytes or empty text.
3. Compute exact `BLAKE3`.
4. Look up matching `(size_bytes, content_hash)` in `doc_registry`.
5. If no match:
   - register/update as canonical
   - continue normal extraction/enrichment/embedding/upsert flow
6. If match:
   - keep original canonical
   - archive duplicate payload/record
   - mark duplicate in `doc_registry`
   - write audit log event
   - update canonical metadata with duplicate source/location reference
   - skip normal indexing for duplicate

### Filesystem source behavior

If duplicate found:
- move or copy duplicate file into configured archive directory
- preserve source-relative path in metadata
- store archive path in registry
- skip indexing duplicate file

Archive layout should be deterministic and traceable.

Recommended archive shape:
- `duplicate_archive_root/filesystem/<canonical_doc_id>/<timestamp>__<original_rel_path>`

### PostgreSQL source behavior

If duplicate found:
- serialize source record to JSON
- write JSON snapshot into configured archive directory
- store source natural key and archive path in registry
- skip indexing duplicate row

Recommended archive shape:
- `duplicate_archive_root/postgres/<source_name>/<canonical_doc_id>/<natural_key>.json`

No delete or mutation of source PostgreSQL rows.

## Canonical Metadata in LanceDB

Canonical LanceDB record must expose duplicate provenance to search clients.

Add metadata fields on canonical chunks/doc:
- `dup_count`
- `dup_sources`
- `dup_locations`
- `dup_archive_paths`
- `dup_natural_keys`

Representation can be JSON strings in metadata fields if needed by current schema constraints.

Example outcomes:
- one canonical indexed document
- many alternate locations/sources attached to that canonical document

This lets external software ask:
- where else was this same file found?
- was this duplicate archived?
- which source originally produced it?

## MCP Tooling

Add duplicate-check MCP tool.

Suggested shape:
- `file_duplicate_check`

Inputs:
- filesystem path
- raw text/content
- source metadata for PostgreSQL-backed record
- optional precomputed hash

Outputs:
- `duplicate: true|false`
- `canonical_doc_id`
- `canonical_rel_path`
- `canonical_source_name`
- `archive_path`
- `first_seen_at`
- duplicate provenance summary

This tool is intended for quick “does this already exist?” checks before other software creates redundant content.

## Config

Add dedupe config block.

Suggested config:

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

Behavior requirements:
- if archive disabled, duplicate still skipped and logged
- if archive write fails, duplicate handling should fail closed or be policy-controlled

Recommended default:
- fail duplicate archival step loudly, but do not destroy source

## Performance Notes

This design is intended to stay cheap.

Expected characteristics:
- no full in-memory hash table
- SQLite database remains on disk
- only index/data pages needed for lookup are pulled into memory
- BLAKE3 computed in streaming mode for large files
- duplicate lookup cost should be tiny compared to OCR/LLM/embed stages

For filesystem:
- full-file hash adds one sequential read
- still small relative to downstream processing cost

For PostgreSQL text rows:
- hash cost is negligible

## Failure Handling

### Empty files / empty payloads

Reject early:
- mark as invalid input
- log audit event
- do not send into PDF parser / OCR / enrichment path

### Archive failure

If duplicate archival fails:
- do not delete source
- log structured error
- mark duplicate handling failure in registry/audit log
- policy decision during implementation:
  - either skip indexing and mark archive failure
  - or fall back to non-archived duplicate skip

Recommended default:
- skip indexing duplicate anyway, but record archive failure explicitly

### Registry mismatch / stale records

If registry says canonical exists but LanceDB no longer has canonical document:
- treat as integrity issue
- log and surface
- do not silently attach duplicate to broken canonical without repair

## Migration

Migration steps:
1. Add new `doc_registry` columns.
2. Backfill `first_seen_at` from `created` where possible.
3. Leave existing rows with null `content_hash` until next touch or explicit backfill.
4. Add exact-hash index.
5. Start dedupe only for rows with new hashing path.

Optional later maintenance:
- backfill hashes for already indexed canonical records
- repair old bare-ID / namespaced-ID duplicate registry rows separately

## Testing

Must cover:
- identical filesystem file skipped as duplicate
- identical PostgreSQL row skipped as duplicate
- duplicate archived and canonical linked
- first-seen wins deterministically
- empty file rejected before parser
- non-duplicate file still indexed normally
- duplicate-check MCP tool returns expected canonical
- archive failure logged without destructive source action
- canonical metadata updated with duplicate provenance

## Open Questions

Open implementation choice:
- exact archive-failure behavior
- exact metadata encoding format for duplicate provenance in LanceDB
- whether duplicate provenance should be stored only in registry or mirrored into every canonical chunk metadata row

Current decisions already locked:
- exact duplicates only
- first-seen wins
- all sources participate
- no destructive delete
- skip indexing duplicates
- reuse existing SQLite registry
