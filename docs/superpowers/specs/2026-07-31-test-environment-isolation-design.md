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

`scripts/gate.py` will create a Prefect home directory inside each unique gate run directory and pass it only to pytest subprocesses through their environment. Existing explicit `PREFECT_HOME` values will not be reused for gate tests. Static tools not needing Prefect remain unchanged.

Each gate invocation already owns a timestamped or caller-provided run directory. Keeping Prefect state below that directory gives deterministic isolation, preserves failure artifacts, avoids global state, and naturally separates concurrent gate runs.

Direct `pytest` behavior remains unchanged. Scope is gate reliability, matching documented `make gate-fast` workflow.

## Data Flow

1. Make resolves Python interpreter.
2. Make launches `scripts/gate.py` with resolved interpreter.
3. Gate resolves run directory and creates `<run-dir>/prefect-home`.
4. Gate builds child environment with `PREFECT_HOME` set to that path.
5. Unit and integration pytest subprocesses inherit isolated Prefect state.
6. Gate writes existing JUnit and result artifacts as before.

## Error Handling

- Failure to create run or Prefect directory remains a hard gate error.
- pytest failures retain existing fail-fast tier semantics.
- Gate never removes or modifies `~/.prefect`.
- Caller-provided `--run-dir` remains supported; its Prefect state belongs to that run directory.

## Testing

Tests will first demonstrate failures, then verify:

- Makefile no longer requires a `python` executable and consistently uses `$(PYTHON)`.
- Gate supplies run-local `PREFECT_HOME` to test-tier subprocesses.
- Explicit ambient `PREFECT_HOME` cannot leak into gate tests.
- Existing tier ordering, skip behavior, and result reporting remain intact.
- Formerly failing Prefect-backed tests pass under isolated state.
- Full `make gate-fast` passes: static, unit, integration.

## Non-goals

- Deleting or repairing developer Prefect databases.
- Changing production Prefect server lifecycle.
- Disabling Prefect orchestration in tests.
- Running paid real-provider tests.
