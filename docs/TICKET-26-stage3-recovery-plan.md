# TICKET-26 — Stage 3 recovery + encoding fix plan

Status: **proposed, not executed.** Stages 0–1 are done (see bottom). Stage 2
(compaction) is **blocked** by the same overflow Stage 3 must fix.

## Root cause (confirmed)

`chunks.lance` stores a very wide `metadata` **struct** (~100 string fields:
`enr_*`, `_node_content`, `message_body`, …). A few documents carry **huge
blobs** in that struct (almost certainly `_node_content` holding full document/
transcript text). Under **LanceDB storage version 2.0**, decoding such a
fragment overflows Arrow's **int32 offset buffer** (`Offset overflow error:
3190264719`, `lance-encoding-4.0.0/.../previous/encodings/logical/struct.rs`).

Consequences:
- Full-table scans (FTS index build, freshness SQL) fail → `index_freshness_available: false`.
- `compact_files` fails (it must decode `metadata`).
- A normal read-and-migrate also fails — even `batch_size=256` scans error,
  because the bad **page** decodes before batching. Only *some* fragments are bad.

The ticket's "patch int32→int64 in our code" does **not** apply: the int32 is in
the Lance Rust library, not this repo. The real fix is (a) stop storing oversized
blobs in `metadata`, and (b) write with storage version **2.1** (64-bit offsets).

## Plan

**S3.1 — Identify bad fragments (read-only).** Loop all fragments; for each, scan
only its `metadata` for a few rows; record fragment ids + the `doc_id`s that
overflow. `id/doc_id/vector/text` are independent columns and decode fine even in
a bad fragment.

**S3.2 — Recover.** Good fragments: keep as-is. Bad fragments: read
`id,doc_id,vector,text` (skip `metadata`); collect their `doc_id`s for clean
re-ingestion (so enrichment metadata is regenerated, not lost silently).

**S3.3 — Code fix (`flow_index_vault.py` write path).**
- Cap/strip the oversized metadata field(s) before indexing — do **not** store
  full `_node_content` (it duplicates `text`); cap any string field to ~64 KB.
  Keep the `enr_*` enrichment fields.
- Set `data_storage_version="2.1"` on dataset create/write (defense in depth).

**S3.4 — Migrate.** Pause indexer (clean `docker stop`). Write a NEW
`chunks.lance` (v2.1, trimmed metadata) from recovered good rows in small
batches; insert bad-fragment rows with trimmed/empty metadata (or re-ingest).
Atomic-swap; keep old dir as backup until verified. Disk is fine (307 G free;
trimmed live data is a few GB).

**S3.5 — Rebuild FTS + verify.** Rebuild FTS on `text`; confirm the freshness SQL
scan succeeds (no overflow) and `index_health.json` → `index_freshness_available: true`.

**S3.6 — Prevent recurrence.** The 17 k-version bloat came from frequent writes
with no maintenance. Add periodic `cleanup_old_versions` + `compact_files` to the
indexer (scheduled or threshold-triggered).

## Safeguards
- Keep old `chunks.lance` as backup until the new one is verified.
- Pause the indexer during migration (clean stop; force-kill risks corruption —
  we saw `exit 137` earlier, though Lance's atomic commits held).
- Verify row count and re-ingest list before deleting the backup.

## Done so far (Stages 0–1, 2026-06-18)
- Cleared 2.8 G cruft (`*.corrupt`, taxonomy `*.bak-*`).
- `cleanup_old_versions`: removed 17,817 dead versions / **188 GB**
  (`chunks.lance` 229 G → 54 G; disk 86 % → 65 %). Live rows 32,815 intact.
- Stage 2 compaction attempted → blocked by the overflow above. Service restarted.
