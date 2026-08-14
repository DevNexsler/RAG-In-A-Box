# Hook delivery outbox final-fix report

## Scope

- Deleted accepted pending rows with CAS on `id`, `revision`, and `status = 'pending'`.
- Preserved `HookOutbox.complete()` non-`None` success result and delivery counters.
- Added CDS semantic acceptance contract to `config.staging.realmedia.yaml`.
- Added regression covering CDS callback config in both staging configs.
- Corrected TESTING callback example and payload note: UUID `event_id` is added.

## Root cause and code decision

`complete()` changed row state to `completed`; completed rows retained `event_json` and
payload content. It now performs `DELETE ... WHERE id = ? AND revision = ? AND
status = 'pending'` inside `BEGIN IMMEDIATE`. A successful delete returns a detached
`HookDelivery` with `status='completed'`, preserving current `drain_due()` success and
counter behavior. Stale/non-pending rows still return `None`.

## Impact

Command:

```bash
gitnexus impact complete --upstream --file core/hook_outbox.py --kind Method --include-tests
```

Result: HIGH risk; 2 direct callers (`hooks.delivery.drain_due`, outbox completion
test), 3 affected processes (`drain_hook_outbox`, `_process_doc_task`, `drain_queues`),
2 affected modules. Direct caller behavior was preserved and covered by focused suite.
No HIGH/CRITICAL result was ignored.

## RED

```bash
pytest tests/test_hook_outbox.py::test_complete_deletes_accepted_delivery_and_stale_transition_is_ignored tests/test_deployment_config.py::test_staging_cds_callbacks_require_semantic_acceptance -q
```

Result: 2 failed. Existing completion retained `event_json`; real-media CDS callback
lacked `accepted_statuses`.

## GREEN and verification

```bash
pytest tests/test_hook_outbox.py::test_complete_deletes_accepted_delivery_and_stale_transition_is_ignored tests/test_deployment_config.py::test_staging_cds_callbacks_require_semantic_acceptance -q
# 2 passed in 1.16s

pytest tests/test_hook_outbox.py tests/hooks/test_dispatcher.py tests/hooks/test_events.py tests/hooks/test_delivery.py tests/test_hook_integration.py tests/test_index_scheduler.py tests/test_targeted_index.py tests/test_deployment_config.py -q
# 95 passed in 4.16s

ruff check core/hook_outbox.py tests/test_hook_outbox.py tests/test_deployment_config.py
# All checks passed!

git diff --check
# exit 0
```

GitNexus change detection run before commit. Commit scope excludes pre-existing dirty
`AGENTS.md` and `CLAUDE.md`.

## Post-commit index refresh

```bash
node -e "const m=require('./.gitnexus/meta.json'); console.log(m.stats && m.stats.embeddings)"
# 5393

node .gitnexus/run.cjs analyze --embeddings
```

Result: failed in GitNexus embedding persistence with `Found duplicated primary key
value :0` while creating `CodeEmbedding`. Code commit is complete; index refresh remains
blocked by this GitNexus analyzer defect.

## Final review round 2: CDS callback uniqueness

Review found the prior contract used `next()`: it validated only first `cds-callback`,
allowing a duplicate callback with missing semantic acceptance to evade the test.

Impact call for existing `test_staging_cds_callbacks_require_semantic_acceptance`:

```text
mcp__gitnexus__impact({
  repo: "/home/danpark/projects/RAG-in-a-Box/.worktrees/fix-112-hook-outbox",
  target: "test_staging_cds_callbacks_require_semantic_acceptance",
  file_path: "tests/test_deployment_config.py", kind: "Function",
  direction: "upstream", includeTests: true
})
```

Result: target unavailable because post-commit embedding index refresh is blocked; no
HIGH/CRITICAL result returned.

RED used a temporary second `cds-callback` without `accepted_statuses` in
`config.staging.realmedia.yaml`:

```bash
pytest tests/test_deployment_config.py::test_staging_cds_callbacks_require_semantic_acceptance -q
# 1 failed: assert 2 == 1
```

GREEN removes that temporary fixture. Test now collects all matching hooks, requires
exactly one CDS callback in each staging config, then requires exact acceptance statuses.

```bash
pytest tests/test_deployment_config.py::test_staging_cds_callbacks_require_semantic_acceptance -q
# 1 passed in 0.90s

pytest tests/test_hook_outbox.py tests/hooks/test_dispatcher.py tests/hooks/test_events.py tests/hooks/test_delivery.py tests/test_hook_integration.py tests/test_index_scheduler.py tests/test_targeted_index.py tests/test_deployment_config.py -q
# 95 passed in 4.12s

ruff check tests/test_deployment_config.py
# All checks passed!

git diff --check
# exit 0
```
