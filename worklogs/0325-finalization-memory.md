# #0325 finalization memory work log

## Scope and baseline

- Worktree: `/home/danpark/projects/RAG-in-a-Box/.worktrees/maint-0325-finalization-memory-20260718`
- Base: `b1c668fa9552f130346cc20229eb6e8b15dca6cd` (merged PR #69)
- Production state, cron, containers, and configuration remained read-only.
- Production image observed: digest `sha256:58bdbc5558854eb201201d7af2d0ea563c2459538c84fa4f41fbb9971f7b0311`, OCI revision `4d81116a765b51f8574a37581369487d4e900e9e`.
- Production timeline correlated the failure with daily Lance maintenance: 1,624 document tasks finished by 14:22:15; taxonomy usage finished 14:23:06; pre-prune began 14:23:13; daily compaction marker changed 14:23:35; FTS finished 14:23:39. Cgroup memory rose from 6.47 GiB at 14:22:29 to the 8 GiB limit around 14:23:29.
- GitNexus was indexed before PR #69 and its final `detect_changes` result described unrelated stale files. Exact local source and git diff were used as authority. Pre-edit impact classified `_merge_index_deltas`, `_compaction_due`, and `_record_compaction` HIGH transitively but with one direct caller; `_optimize_and_prune` and `increment_usage_many` were LOW. All direct callers were inspected and covered.

Focused baseline:

```text
pytest -q tests/test_taxonomy_store.py tests/test_store.py
134 passed
```

## Isolated reproduction

Source was the pre-existing, non-production Docker volume `rag-pr66-profile-86d5a0c-index`, mounted read-only. Identity:

```text
chunks rows=45083 docs=32978 fragments=1272 lance_version=50462 versions=1069
serialized manifest sha256=21aad1f28aff0b76327435efc0c9962aa7aa8ce5dd9ad194a61d63e072f81dc0
taxonomy rows=386
```

Every write test used a new disposable clone, exact PR #69 image, no network, `--memory=8g --memory-swap=8g --pids-limit=512`, and 100 ms cgroup/RSS/PID sampling. Harness: `worklogs/0325-finalization-memory-profile.py`.

Due daily compaction, production order:

```text
taxonomy_embed_calls=386
taxonomy peak: memory.current=486649856 RSS=563544064
pre-prune peak: memory.current=777265152 RSS=759316480
daily compaction peak: memory.current=4630274048 RSS=2962001920 pids=121
final memory.current=4588953600 RSS≈2.61GB
memory.events delta: max=0 oom=0 oom_kill=0
exit=0 elapsed=112.68s
```

Identical current-marker control:

```text
taxonomy_embed_calls=386
peak pre-prune=903331840; index merge=705384448; final=858685440
RSS peak=915869696; pids peak=113
memory.events delta: max=0 oom=0 oom_kill=0
exit=0 elapsed=89.971s
```

Identical explicit `compact_files(num_threads=1, batch_size=512)` trial:

```text
daily compaction peak: memory.current=4764282880 RSS=2750148608 pids=155
final memory.current=4702126080
memory.events delta: max=0 oom=0 oom_kill=0
exit=0 elapsed=99.758s
```

A first production-shaped candidate exposed a second part of the same Lance
rewrite pressure that the isolated process profile did not include: after
successful default compaction, a live MCP status/search sequence raised the
whole 8 GiB service cgroup to 8,240,070,656 bytes. The 7.5 GiB observation
guard stopped the candidate before kernel OOM; `memory.events` stayed zero.
The child observer showed only 2,097,807,360 bytes RSS after compaction, so the
remaining pressure was cgroup-charged streaming I/O/cache plus the persistent
server and live query. This failed candidate was discarded.

Lance's documented `try_binary_copy` compaction mode preserves the general
compaction contract, binary-copies compatible fragments, and falls back to
re-encoding incompatible fragments. On a fresh identical clone, default
binary-copy settings produced:

```text
rows=45083 -> 45083; fragments=1272 -> 1; version=50462 -> 50464
peak memory.current=2997174272; pids=80
memory.events delta: max=0 oom=0 oom_kill=0
exit=0 elapsed=11.348s
```

Explicitly restricting binary-copy threads/read batches was worse
(`memory.current=3,902,521,344`) and was not shipped. The implementation uses
only `compaction_mode="try_binary_copy"`; no corpus thresholds or cache
dropping are involved.

Root cause: full-table data compaction itself adds about 3.7 GiB over the current-marker control. Lance thread/batch limits did not lower its cgroup footprint. Production ran that valid rewrite after document processing/taxonomy state had already raised the baseline, causing the 8 GiB cgroup limit event. Taxonomy finalization separately re-embedded all 386 entries and made one delete/add transaction per entry because public reads omit vectors.

## RED/GREEN implementation

Taxonomy RED:

```text
pytest -q tests/test_taxonomy_store.py::TestIncrementUsage::test_increment_many_preserves_vectors_without_reembedding_or_per_row_commits
FAILED: embed_calls contained Alpha description and Beta description
```

Taxonomy fix reads current Arrow rows with vectors intact, computes positive known-ID increments, and performs one `merge_insert` transaction. No embedding regeneration or unknown-ID insertion.

Finalization RED:

```text
8 focused tests failed
- missing prepare_indexing_maintenance/_compact_data_files APIs
- missing pre_index_maintenance observer phase
- final ensure_fts_index had no compact_data phase control
```

Binary-copy RED:

```text
pytest -q tests/test_store.py::test_data_compaction_prefers_streaming_binary_copy
FAILED: expected compact_files(compaction_mode='try_binary_copy'); got compact_files()
```

Live restore-point RED (from the first successful resource-gated candidate):

```text
post-merge daily tag version=50470; post-prune latest version=50471
pytest -q tests/test_store.py::test_post_prune_restore_point_tracks_exact_latest_version
FAILED: tagged version 2 != post-prune latest version 4

pytest -q tests/test_store.py::test_final_tag_failure_preserves_staged_readable_restore_point
FAILED: expected staged + final tag calls; got one post-prune call
```

Finalization fix:

- Existing non-shadow add/update runs attempt due compaction before `_process_docs`.
- Explicit `pre_index_maintenance` start/finish samples prove phase order.
- Fresh and shadow stores skip pre-compaction. Source-scoped runs use the
  shared table's pre-filter state, so the first document for a new source still
  pre-compacts an already-populated shared table.
- Deletion-only runs retain delete then final all-in-one compaction.
- Mixed add/delete runs compact existing fragments first; later writes/tombstones materialize at the next due daily cycle, like every post-compaction write.
- Finalization after a pre-attempt always skips data compaction, but still merges index deltas, stages today's safety tag, expires old restore tags, post-prunes, then retags the exact latest manifest. The missing-FTS creation path follows the same sequence without adding a data rewrite.
- Data compaction uses a fresh Lance dataset handle. Marker writes only follow success. Failure is non-fatal, leaves marker absent, and is not retried over resident processing state in the same run.
- Compatible fragments use binary-copy compaction; incompatible fragments
  automatically fall back to re-encoding.
- Restore expiry and latest tagging are separate helpers because Lance cleanup
  can commit a metadata-only manifest. A staged tag precedes destructive work;
  expiry then cleanup reclaim old versions; final retag targets exact latest.
  If staging fails cleanup aborts, and if retagging fails the staged restore
  remains readable.
- Index merge failure still propagates to the existing full-FTS rebuild fallback.

Focused verification after implementation:

```text
ruff check flow_index_vault.py lancedb_store.py taxonomy_store.py tests/test_scan.py tests/test_store.py tests/test_taxonomy_store.py worklogs/0325-finalization-memory-profile.py
All checks passed!

pytest -q tests/test_taxonomy_store.py tests/test_store.py tests/test_scan.py
202 passed, 232 warnings in 10.20s
```

The warnings were existing LanceDB deprecations and Prefect missing-flow-context logging warnings. A Prefect temporary-server logger emitted a post-pytest closed-stream logging error after the successful result; pytest exit remained 0.

## Qualification

Pending frozen-SHA full gate and isolated production-shaped live candidate evidence.
