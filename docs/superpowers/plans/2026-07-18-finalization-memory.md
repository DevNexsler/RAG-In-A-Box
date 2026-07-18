# Finalization Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep a production-sized daily index finalization below the 7.5 GiB safety gate without weakening Lance compaction, restore-point, or indexing semantics.

**Architecture:** First separate taxonomy-flush allocation from Lance prune/compaction/index-merge allocation in a fresh 8 GiB candidate using an immutable, checksummed 45,083-row source snapshot. Replace per-entry taxonomy read/embed/delete/add work with one vector-preserving batch mutation. Production-sized comparison showed thread/batch tuning does not reduce daily compaction's ~4 GiB cgroup footprint, so move due data compaction before document processing; finalization then merges newly written index deltas, tags the post-run latest version, and prunes. This separates two valid high-memory phases without skipping maintenance, cache hacks, or corpus thresholds.

**Tech Stack:** Python 3.11, Lance 4.0.0, LanceDB 0.30.2, PyArrow, pytest, Docker/Compose.

---

### Task 0: Load contract and impact context

**Files:**
- Read: `/home/danpark/projects/Maint-Manager/.briefs/0325-finalization-memory-20260718.md`
- Read: `/home/danpark/projects/Maint-Manager/tickets/in-progress/0325-doc-organizer-indexer-memory-runaway-earlyoom.md`
- Read: `/home/danpark/projects/Maint-Manager/knowledge/infra/fleet.md`
- Read: `/home/danpark/projects/Maint-Manager/knowledge/runbooks/doc-organizer-full-test-suite.md`
- Read: `/home/danpark/projects/Maint-Manager/knowledge/runbooks/prefect-ephemeral-orphan-servers.md`
- Read: `AGENTS.md`

- [x] **Step 1: Read full operational context**

Read brief, ticket, applicable infra/runbooks, and repository AGENTS before diagnostics. Confirm supplied worktree/branch and production read-only boundary.

- [x] **Step 2: Load GitNexus overview and symbol context**

Read `gitnexus://repo/RAG-in-a-Box/context`, query finalization flows, and load context for suspected symbols. Note stale graph state and treat exact local source as authority.

- [x] **Step 3: Check impact before every source edit**

Run upstream GitNexus impact for `increment_usage_many` before Task 2 and every Task 3 symbol before edit. Stop and report HIGH/CRITICAL risk.

### Task 1: Isolate allocation source

**Files:**
- Create: `worklogs/0325-finalization-memory-profile.py`
- Create: `worklogs/0325-finalization-memory.md`

- [x] **Step 1: Validate copied corpus identity**

Run an exact-PR69 image against source volume `rag-pr66-profile-86d5a0c-index` read-only. Record Lance version, row/fragment counts, compaction marker, serialized-manifest SHA-256, and source image revision.

- [x] **Step 2: Clone source into disposable writable volume**

Copy source snapshot into `rag-0325-finalize-red`. Re-read all identity fields and require exact match before running writes.

- [x] **Step 3: Profile production order from fresh process**

Run one fixed, non-empty taxonomy-count replay, pre-prune, due compaction, index merge/restore-point management, post-prune, and final count from a network-disabled 8 GiB/512 PID container. Sample process RSS, cgroup `memory.current`, `memory.events`, and `pids.current` every 100 ms. Abort on memory >= 7.5 GiB, PIDs >= 480, or any `max/oom/oom_kill` increment outside the intentional RED reproduction; RED remains capped at 8 GiB/512 PIDs and stops on OOM.

- [x] **Step 4: Run current-marker control**

Start a second fresh process on an identical clone whose daily marker is current and replay the exact same non-empty taxonomy counts. Require control to skip data compaction and record each phase peak. Compare taxonomy-only and due-compaction increments. If compaction remains causal, compare default and bounded `compact_files` settings on identical fresh clones and derive defaults from measured peaks. Document one root-cause statement supported by measurements.

### Task 2: Batch taxonomy usage mutations

**Files:**
- Modify: `taxonomy_store.py:270-284`
- Test: `tests/test_taxonomy_store.py:109-120`

- [x] **Step 1: Write failing regression**

Add a real-store test that creates several taxonomy rows, clears embed-call history, calls `increment_usage_many` with different deltas plus a missing ID, and asserts: exact persisted counters, zero new embeddings, missing ID ignored, unchanged vectors, and one Lance transaction for all existing rows.

Proceed only if Task 1 measures taxonomy flush as a causal contributor (external embeddings and/or per-row mutation memory/version growth). Otherwise omit Task 2.

- [x] **Step 2: Verify RED**

Run `pytest tests/test_taxonomy_store.py::<exact-test> -q`. Expected failure: one embedding plus delete/add transactions per existing entry.

- [x] **Step 3: Implement minimal batch mutation**

Read matching rows with vectors preserved, apply deltas in memory, and commit changed rows through one `merge_insert("id").when_matched_update_all()` transaction. Do not insert unknown IDs or regenerate embeddings.

- [x] **Step 4: Verify GREEN and adjacent behavior**

Run full `tests/test_taxonomy_store.py` and taxonomy-flow tests in `tests/test_scan.py`.

### Task 3: Separate daily compaction from post-processing finalization

**Files:**
- Modify: `lancedb_store.py:1064-1187`
- Modify: `flow_index_vault.py:2288-2344`
- Test: `tests/test_store.py:1263-1345`
- Test: `tests/test_scan.py`

- [x] **Step 1: Record failed bounded-compaction hypothesis**

On identical clones, compare installed Lance default optimization with `compact_files(num_threads=1,batch_size=512)`. Record that both consume roughly 4 GiB cgroup memory if measurements confirm; do not ship ineffective resource knobs.

- [x] **Step 2: Write failing phase-sequencing regressions**

Add store/flow tests proving: due data compaction runs before `_process_docs`; current marker skips data compaction; finalization merges index deltas after processing; restore tagging follows the final merge and targets latest version; compact failure leaves marker absent, does not retry after processing in the same run, and final index merge still runs; final merge failure propagates to existing FTS rebuild handling. Also prove a fresh store skips pre-compaction and creates its table/FTS successfully, while a deletion-only run preserves the existing delete-then-compact order. Mixed add/delete runs deliberately materialize their post-compaction writes and tombstones at the next due daily compaction, matching all writes performed after a pre-processing compaction. Existing code must fail because data compaction occurs inside finalization.

- [x] **Step 3: Verify GitNexus impact before symbol edits**

Run upstream impact for `_optimize_and_prune`, `ensure_fts_index`, `index_vault_flow`, `_process_docs`, and any extracted maintenance helper. Warn root before edits on HIGH/CRITICAL risk and cover every d=1 dependent.

- [x] **Step 4: Implement non-overlapping maintenance phases**

Extract daily data compaction from `_optimize_and_prune` into an idempotent pre-processing method. Call it before `_process_docs` only when an existing non-shadow table has documents to process; fresh stores and shadow rebuilds skip it. Sample explicit `pre_index_maintenance` start/finish observer boundaries. Prune for headroom, compact only when marker is stale, and write marker only after successful compaction. Final `ensure_fts_index` runs with data compaction disabled after any pre-compaction attempt, always merges new index deltas, refreshes restore points against latest post-run version, and post-prunes. Deletion-only runs retain current ordering: delete first, then all-in-one maintenance/compaction, because no processing allocation needs separation. Default callers retain one-call maintenance behavior.

- [x] **Step 5: Verify focused GREEN**

Run affected store/flow tests plus same-day control, due-day retry, restore-point, prune-failure, incremental FTS, and shadow-rebuild tests.

### Task 4: Freeze tree, full verification, and release candidate

**Files:**
- Modify: `worklogs/0325-finalization-memory.md`

- [ ] **Step 1: Verify scope and freeze tracked tree**

Run GitNexus detect-changes and inspect `git diff --check`, `git status`, and all d=1 dependents. Finalize tracked worklog evidence available so far, commit source/tests/docs/worklog, and record exact commit tree. Do not change tracked files after qualification starts; append later candidate evidence to PR/ticket, or commit and repeat all qualification against the new SHA.

- [ ] **Step 2: Run full gate from exact commit**

Run exact repository gate (`make gate`) with repository venv. Require exit 0 for every tier, including live; record static collection, unit, integration, staging E2E, live totals, and 21/21 tool coverage. Stop and escalate instead of opening a PR if any tier cannot run or pass.

- [ ] **Step 3: Pin candidate SHA and image digest**

Build production-shaped image from exact commit, record OCI revision and digest, and refuse to qualify a different tree.

- [ ] **Step 4: Run exact-SHA live candidate**

Use fresh snapshot clones, isolated network/state/ports/credentials, 8 GiB/512 PID limits, and production entrypoint shape. Seed one isolated changed input before each run so `should_update_fts` is true. Exercise due compaction and current-marker neighbor paths through actual supervised MCP `file_index_update`; wait for durable terminal success, then call MCP search and facets. Continuously sample and abort on memory >= 7.5 GiB, PIDs >= 480, or any `memory.events max/oom/oom_kill` increment. Require recorded observer order `pre_index_maintenance` finish before `process` start and `finalize` after `process` finish, one daily-marker write, and today's restore tag pointing at the exact latest post-run version. Also require peak memory `< 7.5 GiB`, peak PIDs `< 480`, events unchanged, health green, restart 0, OOM false, one Prefect server maximum, physical rows equal unique chunk IDs, seeded content searchable, facets healthy, and cleanup proof.

- [ ] **Step 5: Verify qualified identity and external evidence**

Verify running candidate OCI revision equals committed SHA, image digest is unchanged, worktree tracked state remains unchanged, candidate state is isolated, and all containers/networks/volumes/secrets are removed. Record post-commit evidence in PR/ticket rather than altering qualified tree.

- [ ] **Step 6: Push and open PR**

Push `maint/0325-finalization-memory-20260718`, create PR referencing #0325 with exact test/candidate evidence, then coordinate Maint-Manager ticket update with root. Do not merge, deploy, enable cron, or mutate production.
