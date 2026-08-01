# Test Environment Isolation Design

## Problem

Local `make gate-fast` is unreliable for two environment reasons:

1. `Makefile` invokes `python`, but host exposes only `python3` outside project virtual environment.
2. Prefect-backed unit tests inherit default `PREFECT_HOME=~/.prefect`. A stale user database containing obsolete Alembic revision `14806cb26270` prevents Prefect's ephemeral API server from starting. Nineteen tests then fail through one shared startup error.

Repository code must not delete or migrate developer-owned Prefect state merely to run tests.

## Design

### Portable Python selection

`Makefile` will define overridable `PYTHON`. Default selection will prefer project `.venv/bin/python` when present and otherwise use `python3`. Every Python-driven target will use this variable. Callers may still override it, for example `make PYTHON=/opt/python gate-fast`.

### Per-gate Prefect isolation

`scripts/gate.py` will create a fresh, uniquely named Prefect home directory inside the gate run directory for every invocation. It will place that path in the gate process environment before dispatching child commands. Existing explicit `PREFECT_HOME` values will not be reused by gate children.

The unique child directory prevents collisions when two default run names share a timestamp and prevents stale-state reuse when a caller repeats `--run-dir`. Keeping Prefect state below the artifact directory preserves failure evidence and avoids global state.

Direct `pytest` behavior remains unchanged. Scope is gate reliability, matching documented `make gate-fast` workflow.

## Data Flow

1. Make resolves Python interpreter.
2. Make launches `scripts/gate.py` with resolved interpreter.
3. Gate resolves run directory and creates `<run-dir>/prefect-home-<unique-suffix>`.
4. Gate sets `PREFECT_HOME` to that path in its child-process environment.
5. Every child command inherits isolated Prefect state: static checks and collection, unit, integration, staging e2e, live, real-provider e2e, audits, and reports. Commands that do not use Prefect ignore it.
6. Gate writes existing JUnit and result artifacts as before.

## Error Handling

- Failure to create run or Prefect directory remains a hard gate error.
- pytest failures retain existing fail-fast tier semantics.
- Gate never removes or modifies `~/.prefect`.
- Caller-provided `--run-dir` remains supported; every invocation receives a new child Prefect home.

## Testing

Tests will first demonstrate failures, then verify:

- Makefile no longer requires a `python` executable and consistently uses `$(PYTHON)`.
- Gate supplies run-local `PREFECT_HOME` to test-tier subprocesses.
- Explicit ambient `PREFECT_HOME` cannot leak into gate tests.
- Concurrent invocations and repeated use of one `--run-dir` receive different Prefect homes.
- Existing tier ordering, skip behavior, and result reporting remain intact.
- Formerly failing Prefect-backed tests pass under isolated state.
- Full `make gate-fast` passes: static, unit, integration.

## Non-goals

- Deleting or repairing developer Prefect databases.
- Changing production Prefect server lifecycle.
- Disabling Prefect orchestration in tests.
- Running paid real-provider tests.
