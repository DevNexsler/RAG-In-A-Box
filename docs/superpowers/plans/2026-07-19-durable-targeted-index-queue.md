# Durable Targeted-Index Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep one stable Lance writer per table session while durably accepting and eventually processing targeted attachment-index requests.

**Architecture:** A focused SQLite queue coalesces fire-and-forget requests before lock acquisition. Full and targeted entry points share one reentrant process/cross-process table lock and one unlocked targeted worker; drain batches reuse one store. Daily Lance compaction runs in a short-lived child process so native state exits before document processing.

**Tech Stack:** Python 3.12/3.13, SQLite, `fcntl.flock`, Starlette, Prefect, Lance/LanceDB, pytest, Docker Compose.

---

### Task 1: Durable queue core

**Files:**
- Create: `core/index_request_queue.py`
- Create: `tests/test_index_request_queue.py`

- [ ] Write failing tests for normalized-key dedupe, atomic force escalation, monotonic revision, revision-guarded success/failure, ordering/limits, failure metadata, reopen-after-process-exit durability, crash-mid-drain durability (child exits after work starts; unchanged row later processes), and concurrent SQLite writers.
- [ ] Run `python3 -m pytest tests/test_index_request_queue.py -q`; expect import/test failures.
- [ ] Implement `IndexRequest`, `IndexRequestQueue.enqueue()`, `pending()`, `complete()`, and `fail()` using composite uniqueness, WAL, `synchronous=FULL`, `busy_timeout`, explicit transactions, and UTC timestamps.
- [ ] Re-run queue tests; expect all pass.
- [ ] Run Ruff on module/test and commit `feat: add durable targeted-index queue`.

### Task 2: Correct table-session lock

**Files:**
- Modify: `core/index_write_lock.py`
- Modify: `tests/test_index_write_lock.py`

- [ ] Add failing tests for nested same-thread acquisition, nonblocking thread/process exclusion, distinct-table independence, and release after unlock/close failure.
- [ ] Run `python3 -m pytest tests/test_index_write_lock.py -q`; verify the nested test times out/fails for the existing implementation.
- [ ] Implement per-key process `RLock` depth/file-handle state so nested acquisition shares the outer `flock`; guarantee process-lock release in every error path.
- [ ] Re-run lock tests; expect all pass without hung children.
- [ ] Commit `fix: make index writer session safely reentrant`.

### Task 3: Queue-aware targeted indexing

**Files:**
- Modify: `flow_index_vault.py`
- Modify: `api_server.py`
- Modify: `tests/test_targeted_index.py`
- Modify: `tests/test_targeted_index_api.py`

- [ ] Add failing tests proving enqueue commits before lock attempt, busy returns queued, API maps queued to 202, current revision runs first, batches are bounded, one store/registry serves the batch, failure retains work while later rows progress, terminal not-found completes, and a late revision survives old-revision completion.
- [ ] Add API regression proving SQLite enqueue failure returns non-202 and never attempts the table lock.
- [ ] Run targeted tests; verify expected failures.
- [ ] Extract `_index_document_unlocked(config, request, store, registry)` from the public flow. It must never acquire the session lock or reopen/refresh Lance.
- [ ] Implement `_drain_index_requests(...)` with a bounded snapshot, current-key priority, once-per-invocation processing, revision-safe completion/failure, and per-item result capture.
- [ ] Change `index_document_flow` to load config once, durably enqueue, attempt the nonblocking table lock, return queued on contention, otherwise open one store/registry and drain.
- [ ] Map queued REST results to HTTP 202. Preserve MCP JSON contract.
- [ ] Re-run targeted/API tests; expect all pass.
- [ ] Commit `feat: durably queue targeted index requests`.

### Task 4: Full-sweep integration and stable-handle invariants

**Files:**
- Modify: `flow_index_vault.py`
- Modify: `lancedb_store.py`
- Modify: `core/storage.py`
- Modify: `tests/test_scan.py`
- Modify: `tests/test_store.py`

- [ ] Add failing tests for config loaded once, `_RUNTIME` untouched before lock acquisition, no-change sweep draining, queued arrival draining before release, and no per-document checkout/dataset/store construction for insert, upsert, or delete under the session.
- [ ] Add source-scoped sweep regression where a queued request from another configured source drains through shared store/registry without inheriting the sweep's filtered source runtime.
- [ ] Add stale-completed-cache regression: insert → delete → standalone insert must restore rows.
- [ ] Run focused scan/store tests; verify failures.
- [ ] Pass the lock-derived config object into the full flow without reload; clear runtime only inside the acquired session.
- [ ] Drain the queue after the core sweep and before lock release, reusing `_RUNTIME`'s active store and registry while rebuilding per-request source/provider runtime from the unfiltered loaded config.
- [ ] Add explicit externally-guarded writer-session state to `LanceDBStore`; session-locked insert/upsert/delete skip per-document checkout/probe/reopen, while standalone defaults retain them. Ensure shadow/recovery stores inherit session state and clear it before lock release.
- [ ] Invalidate completed-insert IDs on delete. Retain conditional latest probe for standalone inserts and exception-only ambiguous-commit recovery.
- [ ] Re-run focused tests; expect all pass.
- [ ] Commit `fix: serialize full and targeted Lance writers`.

### Task 5: Short-lived compaction worker

**Files:**
- Create: `core/lance_maintenance.py`
- Modify: `lancedb_store.py`
- Create: `tests/test_lance_maintenance.py`
- Modify: `tests/test_store.py`

- [x] Add and observe failing tests for subprocess dispatch, binary-copy mode, and stderr preservation for commit-conflict retry classification.
- [x] Implement `python -m core.lance_maintenance compact <dataset>` with `close_fds=True`, captured diagnostics, and no child lock acquisition.
- [ ] Verify parent refreshes once after successful child exit and marker remains absent on child failure.
- [ ] Run maintenance/store tests; expect all pass.
- [ ] Commit `fix: release compaction native state before indexing`.

### Task 6: Review and deterministic verification

**Files:**
- Modify: `worklogs/0325-finalization-memory.md`

- [ ] Run `ruff check` and `git diff --check` over changed files.
- [ ] Run complete unit suite (`python3 -m pytest -m unit -q`) and record totals.
- [ ] Run every `scripts/run_full_gate.sh` phase against the same reviewed source: static, complete unit, complete integration, staging E2E, live E2E, and coverage.
- [ ] Dispatch independent code reviewer; address every correctness/cleanliness issue and re-run affected tests.
- [ ] Commit worklog evidence and push PR branch.

### Task 7: Exact candidate live/resource qualification

**Files:**
- Modify: `worklogs/0325-finalization-memory.md`

- [ ] Commit/push tracked worklog evidence before the final build. Build production-shaped image from that exact reviewed SHA. Record source SHA, image ID/digest, label, and forbidden-file inspection in untracked gate artifacts and the Maint-Manager ticket/PR evidence.
- [ ] Clone the representative 1,624-document snapshot to isolated volume/network/Postgres/provider state. Use 8 GiB memory/swap, 512 PIDs, and production `nofile=65535`.
- [ ] Force due compaction; run full sweep plus live E2E. Record 1,624/1,624 outcome, compaction-child exit, post-exit PID/RSS/FD state, cgroup peak, OOM/PID events, health/restarts, DB/index counts, and cleanup.
- [ ] Repeat neighboring current-marker path without compaction; run the complete public-boundary live E2E suite.
- [ ] If any tracked file changes after image build, commit/push, rebuild, and repeat every deterministic and candidate-container gate. Do not create a circular final worklog edit; final exact-SHA evidence lives in immutable untracked artifacts plus Maint-Manager ticket/PR comment.
- [ ] Obtain final independent review, verify GitHub checks/mergeability, then hand PR to manager for merge/deploy decision.
