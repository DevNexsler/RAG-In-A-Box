# Final-fix report — #1688 index maintenance

Commit: final-fix commit below.

## Findings addressed

1. Interrupted boots left the resumed sweep due. `tick()` ran that sweep before
   maintenance; the sweep could claim the writer lock, maintenance returned
   `writer_busy`, and its interval clock advanced. Maintenance now runs after
   queue drain but before sweep and compaction, so it owns boot's idle window
   while preserving queue-first behavior.
2. Added interrupted-boot regression. It simulates the resumed sweep taking
   the writer lock and requires maintenance to run first with `attempted`.
3. `maintain_index_if_idle()` now returns `attempted`, not `maintained`.
   `_finish_index_maintenance()` may skip best-effort later work after failure,
   so completion was not guaranteed.

## TDD evidence

Red before production edits:

```text
../../.venv/bin/pytest tests/test_index_scheduler.py -q -k interrupted_boot_runs_maintenance_before_sweep_takes_writer_lock
1 failed, 24 deselected
events: ['sweep']; expected ['maintenance', 'sweep']

../../.venv/bin/pytest tests/test_index_compaction_window.py -q -k maintenance_attempts_once_when_no_index_writer_holds_the_table
1 failed, 4 deselected
got {'status': 'maintained'}; expected {'status': 'attempted'}
```

Green after production edits:

```text
../../.venv/bin/pytest tests/test_index_scheduler.py tests/test_index_compaction_window.py -q
30 passed in 4.38s
```

## Verification

```text
../../.venv/bin/ruff check core/index_scheduler.py flow_index_vault.py tests/test_index_scheduler.py tests/test_index_compaction_window.py
All checks passed!

git diff --check
exit 0

make gate-fast
PASS (static, unit, integration); report: .evals/gate-runs/20260827-070437/report.md
```

GitNexus task-context pre-change impacts: LOW for scheduler/finalizer, no API
routes. `detect_changes()` found four changed files, medium aggregate scope,
and three existing queue-drain flows; no unexpected execution flows.

## Files

- `core/index_scheduler.py`
- `flow_index_vault.py`
- `tests/test_index_scheduler.py`
- `tests/test_index_compaction_window.py`
- `.superpowers/sdd/1688-index-maintenance/final-fix-report.md`
