# Durable Targeted-Index Queue Design

## Problem

Full-corpus indexing must keep one stable Lance writer session. Refreshing or
reopening Lance for every new document leaks native manifest/cache state on the
production-sized corpus; allowing a targeted attachment writer to interleave
instead creates stale-snapshot duplicate/lost-update races. The existing CDS
attachment hook is fire-and-forget, so returning a `busy` body with HTTP 200
drops work rather than retrying it.

Daily compaction also leaves several GiB of native state resident in the
long-lived index process. The measured representative run completed all 1,624
documents and finalization, then crossed the 7.5 GiB cgroup guard during process
teardown.

## Considered Approaches

1. **Compaction subprocess only.** Smallest patch, but leaves targeted/full
   stale-snapshot races and the long-lived manifest refresh leak. Rejected.
2. **Block targeted HTTP calls behind the full sweep.** Preserves consistency,
   but a production sweep takes about 15 minutes while callers time out in
   seconds. Rejected.
3. **Table session lock plus durable SQLite queue.** RAG accepts ownership of
   fire-and-forget requests, processes them under the same writer session, and
   survives process restarts. Chosen.

## Components

### `core/index_request_queue.py`

Owns a SQLite database beside the Lance table. One row represents one normalized
`(table_name, source_name, target)` request. A uniqueness constraint coalesces
duplicates; any repeated `force=true` request escalates the stored request and
can never be downgraded. Rows contain `status`, `attempts`, created/updated UTC
timestamps, and `last_error`.

Queue transactions are short and independent of Lance. Connections use WAL,
`synchronous=FULL`, and a bounded busy timeout; enqueue failure propagates and
must never produce HTTP 202. Enqueue is one atomic UPSERT and always commits
before attempting the table session lock, preventing SQLite-to-Lance lock
inversion. A monotonically increasing row revision closes the coalescing race:
the drainer deletes or marks failure only when the processed revision still
matches. A concurrent force escalation therefore remains pending rather than
being erased by completion of an older non-force revision.

A drainer snapshots a bounded ordered batch and processes every snapshot row at
most once per invocation. Rows remain `pending` during work, so a process crash
needs no lease reclamation. Successful or terminal-not-found work is deleted
only after the unlocked targeted worker returns. Exceptions retain the row,
increment attempts, record the error, and do not stop later snapshot rows; one
poison request cannot spin or starve its neighbors.

### `core/index_write_lock.py`

Provides process-thread and cross-process exclusion per index root/table. Nested
same-thread acquisition shares the outer `flock`; other threads/processes remain
excluded. Cleanup releases the process lock even if unlock or close raises.

### Flow integration

`index_document_flow` first enqueues, then attempts the table lock without
blocking. If a full sweep owns it, the function returns `status=queued`. If it
acquires the lock, it drains pending requests through a private unlocked targeted
worker. Its own revision is processed first so the synchronous caller receives
that result; only a configured bounded number of neighboring requests follow,
preventing one HTTP request from draining an unlimited backlog.

`index_vault_flow` loads configuration once, acquires the matching table lock,
runs the existing sweep, then drains pending requests before releasing the lock.
Every scheduled sweep drains even when its main scan has no changes. A request
that lands after the final empty queue read but before unlock remains durable for
the next targeted request or scheduled sweep; guarantee is eventual delivery,
not immediate delivery during that narrow edge.

The unlocked worker is the only implementation of targeted indexing. Queue
draining never calls the decorated public flow, preventing nested session-lock
acquisition and runtime clearing. One Lance store handle and one document
registry handle are reused for the whole session/drain batch. Under a session
lock, insert/upsert/delete performs zero per-document `checkout_latest`, Lance
dataset construction, or store reopening; standalone storage calls retain their
conditional latest-manifest probe. Full-flow runtime is cleared only after lock
acquisition and is not disturbed while another writer owns the session.

### HTTP/MCP contract

The REST endpoint maps `status=queued` to HTTP 202. CDS can continue its current
fire-and-forget behavior because RAG has durably accepted delivery. MCP returns
the same queued payload. Indexed, skipped, and terminal-not-found behavior stays
compatible.

### Compaction isolation

Daily binary-copy compaction runs as `python -m core.lance_maintenance compact`
in a short-lived subprocess while the parent owns the table session. The child
is authorized by that parent session, does not reacquire the lock, and inherits
no lock file descriptors. It must exit successfully before the first document;
its native Lance/Arrow PID, RSS, and descriptors therefore disappear before
processing. Worker stderr is preserved in the parent exception so existing
retryable-commit-conflict classification still works. The parent refreshes its
single store handle exactly once through the existing post-compaction path.

## Failure Handling and Invariants

- Queue row committed before any nonblocking session-lock attempt.
- Exactly one Lance writer session per table; no queue drainer bypasses it.
- Queue request deleted only after success, unchanged skip, or terminal
  not-found result, and only if its revision is unchanged.
- Processing exceptions retain request, increment attempts, and record error.
- Duplicate requests coalesce; force only escalates.
- Process crash after enqueue or mid-drain preserves work.
- Configuration used to derive lock identity is the same loaded object used by
  the flow; no reload can switch roots/tables while blocked.
- No production state participates in candidate tests.

## Verification

Automated regressions cover cross-process and nested lock behavior, queue crash
durability, mid-drain crash semantics, revision-safe dedupe/force escalation,
bounded ordering/current-result behavior, failure retention without starvation,
SQLite contention, REST 202, drain success, late-arrival persistence, runtime
isolation, one-store/no-refresh behavior, and no-change sweep draining. Then
run the complete static/unit/integration/staging/live suites. Build an exact-SHA
candidate container and repeat the 1,624-document due-compaction resource gate
under 8 GiB/512 PID limits plus the current-marker neighboring path. Record image
digest, cgroup peak, OOM/PID events, compaction-child exit and post-exit
RSS/PID/FD state, health, restart state, and cleanup.
