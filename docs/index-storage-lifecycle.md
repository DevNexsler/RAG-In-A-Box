# Index Storage Lifecycle & Recovery

How the LanceDB index (`/data/index/chunks.lance`) stores data, keeps itself
from bloating, retains restore points, and how to recover from a bad state.
This is the operational companion to the per-chunk schema in
[`architecture.md`](architecture.md#lancedb-schema-per-chunk).

Everything here lives in `lancedb_store.py` (`LanceDBStore.ensure_fts_index` →
`_optimize_and_prune` → restore-point expiry/prune/tag) plus
`scripts/backup_index.sh`.

## Mental model: LanceDB is copy-on-write

LanceDB never edits data files in place. Every write — an upsert, and crucially
every `optimize()` (which **compacts** small fragments into larger ones) —
writes *new* immutable data files and records a new **version** pointing at
them. Old versions stay on disk so in-flight readers keep a consistent
snapshot. It is closer to git than to a mutable database.

Consequence: **something must delete old versions**, or they accumulate
forever. Three mechanisms below manage that, plus a fourth (an independent
copy) for the "the directory itself is gone" case.

## 1. Index merge (every run) + data compaction (once per day)

`ensure_fts_index()` runs at the end of each indexing flow. Every run merges
freshly-written rows into the native BM25 inverted index
(`optimize.optimize_indices()` — no data rewrite). Keyword search is correct
even before this runs; the merge is a performance step.

Full data compaction (`compact_files(compaction_mode="try_binary_copy")`, which
binary-copies compatible fragments and safely re-encodes incompatible ones)
runs at most **once per calendar day**. Compacting every ~15-min
run was the #0232 outage mechanism: each rewrite supersedes every data file,
and any retained restore-point tag pins the superseded set — per-run
compaction × daily tags held hundreds of full-table rewrites on disk (350+ GB
for a ~5 GB table). The cadence is tracked by a plain-text marker next to the
dataset (`chunks.lance.last-compaction`, atomic replace): it survives restarts
without pinning any Lance version, and a failed compaction leaves it unwritten
so the next run retries.

If the incremental merge hits a transient Lance commit conflict (another writer
raced it) it raises, and the flow falls back to a **full FTS rebuild**
(`create_fts_index`, which only reads the small `text` column). That fallback
is why keyword search never silently goes stale. Compaction failures are
non-fatal (logged, retried next run).

The `doc_id` BTree is created only when missing. Opening a store for status or
search must never replace a valid existing scalar index: replacement is a
write, creates a new Lance version, and would turn a read path into version and
disk churn while moving latest past the finalized restore tag.

## 2. Version pruning — keeps disk bounded (#0112)

`optimize()` compacts but does **not** delete the versions it supersedes. Left
alone, the index regrew **5.0 GB → 8.6 GB in ~11 h** of normal traffic (only
~68 new docs — the rest was dead version churn).

`_optimize_and_prune` prunes twice per run via the dataset-level
`cleanup_old_versions(older_than=<retention>, error_if_tagged_old_versions=False)` —
once *before* maintenance (frees headroom before any rewrite) and once *after*
restore-point expiry (so versions unpinned by an expiring tag are reclaimed in
the same run, not the next one). Each pass logs reclaimed bytes/files:

- **Retention window** `LANCE_VERSION_RETENTION_MINUTES` (default **30**):
  versions younger than this are kept so a slow reader / long indexing scan is
  never cut off mid-read. Disk settles at a *bounded plateau* of a few copies'
  worth, not the old unbounded creep. Lower it (e.g. 10) to hold fewer copies;
  it must exceed the longest concurrent read (~indexing-flow duration).
- Runs via the **dataset** API, not `table.optimize(cleanup_older_than=…)`,
  because the latter cannot skip tagged versions and would error every run once
  a daily restore-point tag exists (see #3).
- Best-effort: a prune failure is logged, never fatal — stale versions reclaim
  on the next run. Transient commit conflicts are retried
  (`core.resilience.call_with_retry`).

Manual one-off reclaim (e.g. after a big repair), takes it to the true floor:

```python
from datetime import timedelta
store._vs.table.optimize(cleanup_older_than=timedelta(0))   # delete ALL superseded
```

## 3. Daily restore points — logical rollback (#0113)

Pruning would also delete anything you might want to roll back to. Each run
first stages today's `daily-<YYYY-MM-DD>` Lance tag, then expires old managed
tags, prunes newly unpinned versions, and retags today at the exact latest
manifest. The safety tag blocks destructive expiry/pruning if its creation
fails and remains readable if the final retag fails. Retagging follows pruning
because cleanup can commit a metadata-only manifest version. Tagged versions
are immune to later prunes, so each daily tag is an in-place snapshot.

- Retention `LANCE_DAILY_RESTORE_POINTS` (default **7**) counts retained
  points *exactly*: today plus N-1 prior days. Older `daily-*` tags are deleted
  so their versions become reclaimable; `0` disables the feature **and removes
  previously created daily tags**. Tags that aren't `daily-*` are never touched.
- **Not free**: tags share immutable data files, but with once-daily compaction
  each retained tag can pin up to one full rewrite of the table for its day —
  which is why the default window is 7, not 30 (#0232).
- **Scope**: protects against *logical* disaster — a bad indexing run, a botched
  migration, a corruption event like the #0108 `_node_content` bloat. It does
  **not** survive loss/corruption of the `chunks.lance` directory itself; that's
  what the tarball backup (#4) is for.

**Roll back** to a day (inspect first, then promote):

```python
import lance
path = "/data/index/chunks.lance"
lance.dataset(path, version="daily-2026-07-01").count_rows()   # inspect the snapshot
lance.dataset(path, version="daily-2026-07-01").restore()      # make it the new current version
```

List available restore points: `store._vs.table.tags.list()`.

## 4. Nightly tarball backup — physical DR (#0113)

`scripts/backup_index.sh` tars the whole index volume to
`/home/danpark/backups/doc-organizer/` nightly (~04:30). This is an
**independent** copy (a tarball shares no inodes with the live dataset), so it
survives the live directory being deleted or corrupted — the case the
in-dataset tags can't cover.

- Retention: **14 daily + 8 weekly** (Sunday copies promoted to `weekly/`).
  30 *full* daily copies would be ~150 GB and won't fit the disk, so the long
  granular tail lives in the cheap in-dataset tags (#3) instead; the tarballs
  give ~2 months of coarse physical DR.
- Excludes the transient shadow table, `*.corrupt`, and logs.
- Single-disk host: `backups/` is on the same volume as the index, so this
  protects against logical loss of the directory, **not** a disk failure. True
  off-disk DR would need an offsite/second-volume target (not present today).

Restore from a tarball (full recipe in the script header):

```bash
docker compose stop doc-organizer
docker run --rm -v rag-in-a-box_doc-organizer-data:/vol \
  -v /home/danpark/backups/doc-organizer:/backup alpine \
  sh -c "rm -rf /vol/chunks.lance && tar xzf /backup/<file>.tar.gz -C /vol"
docker compose start doc-organizer
```

## Recovery decision guide

| Symptom | First move |
|---|---|
| Bad data written today / botched run, index otherwise fine | Roll back to yesterday's `daily-*` tag (#3) — instant, no downtime |
| Keyword search stale / `Incremental FTS update failed` in logs | Self-heals via full rebuild; check `/health` `fts_rebuild_failed`. See [[node-content-nesting-fts-overflow]] history |
| Index disk ballooning | Confirm prune is running (`FTS index optimized` in `indexer.log`); manual reclaim (#2). Check for row bloat (§ #0108 playbook) |
| `chunks.lance` directory corrupt / missing | Restore from nightly tarball (#4) |
| A single row is enormous (page decode "offset overflow") | The #0108 repair playbook: identify bloated docs, rewrite in place or delete+reindex, then prune |

## Health signals

- `GET /health` — `fts_rebuild_failed` non-zero ⇒ keyword search degrading.
  Every payload also carries `disk_used_percent` / `disk_free_bytes` /
  `disk_max_percent` for the index filesystem; at/above
  `DISK_USAGE_MAX_PERCENT` (default 90) the probe returns **503** with
  `status: disk_full` (#0232).
- `indexer.log` — `FTS index optimized (incremental merge)` = healthy;
  `Lance version prune (…): reclaimed N bytes` = the prune is doing real work;
  `Incremental FTS update failed … falling back to full rebuild` = the merge
  conflicted/failed that run (self-healed, but investigate if persistent);
  `Daily Lance compaction failed` = non-fatal, retried next run (investigate
  if persistent).
- Disk: `du -sh /data/index/chunks.lance`. Live footprint is ~1 GB per ~40 K
  chunks; a bounded plateau above that is retained versions + daily tags.

## Env var reference

| Var | Default | Effect |
|---|---|---|
| `LANCE_VERSION_RETENTION_MINUTES` | 30 | Prune versions older than this each run (#2). 0 = prune all superseded. |
| `LANCE_DAILY_RESTORE_POINTS` | 7 | Exactly this many `daily-*` restore-point tags kept (#3). 0 = disable tagging and drop existing daily tags. |
| `DISK_USAGE_MAX_PERCENT` | 90 | `/health` 503s (`disk_full`) when the index filesystem is at/above this used-percent. |

All three are wired in `docker-compose.yml`.
