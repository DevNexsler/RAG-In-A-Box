# Index Storage Lifecycle & Recovery

How the LanceDB index (`/data/index/chunks.lance`) stores data, keeps itself
from bloating, retains restore points, and how to recover from a bad state.
This is the operational companion to the per-chunk schema in
[`architecture.md`](architecture.md#lancedb-schema-per-chunk).

Everything here lives in `lancedb_store.py` (`LanceDBStore.ensure_fts_index` →
`_optimize_and_prune` → `_manage_restore_points`) plus `scripts/backup_index.sh`.

## Mental model: LanceDB is copy-on-write

LanceDB never edits data files in place. Every write — an upsert, and crucially
every `optimize()` (which **compacts** small fragments into larger ones) —
writes *new* immutable data files and records a new **version** pointing at
them. Old versions stay on disk so in-flight readers keep a consistent
snapshot. It is closer to git than to a mutable database.

Consequence: **something must delete old versions**, or they accumulate
forever. Three mechanisms below manage that, plus a fourth (an independent
copy) for the "the directory itself is gone" case.

## 1. Compaction + FTS merge (every indexing run)

`ensure_fts_index()` runs at the end of each indexing flow. If the FTS index
exists it calls `table.optimize()`, which compacts new/​small fragments and
merges freshly-written rows into the native BM25 inverted index. Keyword search
is correct even before this runs; the merge is a performance step.

If the incremental merge hits a transient Lance commit conflict (another writer
raced it) it raises, and the flow falls back to a **full FTS rebuild**
(`create_fts_index`, which only reads the small `text` column). That fallback
is why keyword search never silently goes stale.

## 2. Version pruning — keeps disk bounded (#0112)

`optimize()` compacts but does **not** delete the versions it supersedes. Left
alone, the index regrew **5.0 GB → 8.6 GB in ~11 h** of normal traffic (only
~68 new docs — the rest was dead version churn).

`_optimize_and_prune` now prunes every run via the dataset-level
`cleanup_old_versions(older_than=<retention>, error_if_tagged_old_versions=False)`:

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

Pruning would also delete anything you might want to roll back to. So each run,
after compaction, `_manage_restore_points` **tags** the current version
`daily-<YYYY-MM-DD>` (Lance tags). Tagged versions are immune to the prune, so
each daily tag is an in-place, point-in-time snapshot.

- Retention `LANCE_DAILY_RESTORE_POINTS` (default **30** days). Daily tags older
  than the window are deleted so their versions become reclaimable. Tags that
  aren't `daily-*` are never touched.
- **Cheap**: tags share immutable data files, so 30 days costs only each day's
  delta (append-heavy message corpus = small), not 30 full copies.
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
- `indexer.log` — `FTS index optimized (incremental merge)` = healthy;
  `Incremental FTS update failed … falling back to full rebuild` = the merge
  conflicted/failed that run (self-healed, but investigate if persistent).
- Disk: `du -sh /data/index/chunks.lance`. Live footprint is ~1 GB per ~40 K
  chunks; a bounded plateau above that is retained versions + daily tags.

## Env var reference

| Var | Default | Effect |
|---|---|---|
| `LANCE_VERSION_RETENTION_MINUTES` | 30 | Prune versions older than this each run (#2). 0 = prune all superseded. |
| `LANCE_DAILY_RESTORE_POINTS` | 30 | Days of `daily-*` restore-point tags to keep (#3). 0 = disable tagging. |

Both are wired in `docker-compose.yml`.
