# Index Run Supervision and Memory Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make background indexing single-flight, durably observable, kill-clean, health-visible after crashes, bounded in queued work, and profile-ready without changing Lance schema behavior.

**Architecture:** Add a disk-backed `IndexRunSupervisor` that atomically owns launch state, monitors the process, records terminal exit/signal and peak RSS, reconciles interrupted state on startup, and terminates the process group during server shutdown. Keep MCP launch non-blocking, expose additive run-attempt/success freshness in `file_status` and `/health`, bound executor futures to worker count, and add disabled-by-default RSS/Arrow samples at pipeline and document boundaries.

**Tech Stack:** Python 3.13, `fcntl` file locking, atomic JSON replacement, subprocess process groups, `/proc` RSS sampling, PyArrow allocator counters, pytest, Docker Compose, GitNexus.

---

### Task 1: Deployment process ownership

**Files:**
- Modify: `docker-compose.yml`
- Modify: `server.py`
- Test: `tests/test_deployment_config.py`
- Test: `tests/test_server_entrypoint.py`

- [x] Add failing tests requiring `init: true`, direct exec-form `python server.py`, and server-finally supervisor shutdown.
- [x] Run focused tests and verify failures describe missing lifecycle contracts.
- [x] Replace shell PID 1 command with direct exec form; add init/reaper; initialize/reconcile supervisor before serving and terminate supervised process groups after serving stops.
- [x] Run focused tests green.

### Task 2: Durable single-flight run supervisor

**Files:**
- Create: `index_run_supervisor.py`
- Modify: `mcp_server.py`
- Test: `tests/test_index_run_supervisor.py`
- Test: `tests/test_index_nonblocking.py`

- [x] Add failing tests for atomic concurrent launch, durable active/terminal state, success and signal exit, peak RSS, dead-run startup reconciliation, legacy PID adoption, and TERM→KILL process-group shutdown.
- [x] Verify tests fail because supervisor API/state does not exist.
- [x] Implement versioned atomic state file plus `flock` launch lock, parent-owned PID file, monitor threads, startup reconciliation, injected process/RSS/signal seams, and registry shutdown.
- [x] Route `_file_index_update_impl` through supervisor while preserving immediate `started`/`already_running` responses and source validation.
- [x] Run supervisor, real OS process-group, and nonblocking tests green.

### Task 3: Honest status and health freshness

**Files:**
- Modify: `mcp_server.py`
- Modify: `docker-compose.yml`
- Test: `tests/test_mcp_contract.py`
- Test: `tests/test_deployment_config.py`

- [x] Add failing tests for additive `index_run` status fields, idle-after-signal failure, startup-lost failure, last-success comparison, and active-run compatibility.
- [x] Add failing `/health` tests: unresolved terminal failure returns 503; active fresh heartbeat remains healthy; disk pressure keeps priority.
- [x] Implement one shared run summary carrying current/last-attempt/last-success/latest-terminal state, peak RSS, and unresolved-failure freshness.
- [x] Preserve existing result keys and deep-health behavior; add run summary rather than changing schema consumers.
- [x] Run status/health tests green.

### Task 4: Bounded submissions and opt-in memory profile samples

**Files:**
- Create: `memory_observer.py`
- Modify: `flow_index_vault.py`
- Modify: `config.yaml.example`
- Test: `tests/test_indexer_concurrency.int.test.py`
- Test: `tests/test_index_memory_observability.py`

- [x] Add a failing executor test proving outstanding futures never exceed configured workers while all docs still complete and failure ordering remains stable.
- [x] Replace eager `Executor.map` with a completion-driven bounded window of at most `concurrency` futures.
- [x] Add failing tests for disabled zero-cost behavior and enabled structured RSS/current peak/Arrow allocation samples at phase and document boundaries.
- [x] Implement a small observer reading `/proc/self/status` and `pyarrow.total_allocated_bytes()` only when config enables it; emit machine-parseable structured log records.
- [x] Add initialize/scan-diff/process/finalize and per-document samples. Do not alter Lance schema or `_evolve_metadata_schema`.
- [x] Run concurrency, observer, scan, and tracing tests green.

### Task 5: Release verification and handoff

**Files:**
- Modify: `docs/superpowers/plans/2026-07-15-index-run-supervision-memory-observability.md`
- Modify externally: Maint-Manager ticket/worklog and PR #66 body

- [x] Run `make gate-fast`: static, 1,175 unit, 70 integration tests passed.
- [x] Run GitNexus `detect_changes`; indexed main-checkout mismatch documented in ticket/worklog. Review HIGH-risk status surface and direct dependents via targeted/full tests.
- [x] Run isolated staging E2E: 26 tests and 21/21 tool coverage/tracing passed. Production live tier deliberately not run under no-live-service assignment.
- [x] Commit branch and prepare same PR head/ticket/worklog handoff. Push/update follow immediately; do not merge or deploy.
