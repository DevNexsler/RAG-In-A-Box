# Hook Delivery Outbox Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `document.indexed` callbacks durable, semantically acknowledged, retried safely, and auditable.

**Architecture:** Persist one outbox row per matching hook before sending. `hooks/http.py` returns a structured transport plus semantic result; the outbox deletes only accepted rows and retains retry/redrive rows. The existing scheduler drains target-index and hook queues on its normal short interval.

**Tech Stack:** Python 3.12, SQLite WAL, urllib, OpenTelemetry JSONL, pytest.

**Spec:** `docs/superpowers/specs/2026-08-14-hook-delivery-outbox-design.md`

## Global Constraints

- Run `gitnexus_impact({target: <symbol>, direction: "upstream"})` before editing every function, class, or method; warn and stop on HIGH/CRITICAL risk.
- Test first. Capture expected failing output before production edits.
- Persist no secret value. Never log/span event text, chunks, request headers, payload, or raw response body.
- Persist full event payload only under `index_root`; delete it after accepted delivery.
- `event_id` is immutable UUID text. One delivery row per `(event_id, hook_name)`.
- Five sends total: failures one through four retry after `1, 5, 30, 120` seconds. Fifth failure becomes `redrive_required`.
- CDS hook accepts exactly response statuses `updated` and `duplicate`; generic hook accepts every 2xx if it has no `accepted_statuses` config.
- CDS response statuses `no_match`, `ambiguous`, `correlation_mismatch` are terminal `redrive_required`; malformed/unexpected 2xx response retries under normal cap.
- Keep index success independent from downstream failure.
- Do not change disk threshold/capacity policy. Document operator remediation only.
- Baseline unrelated failure: `tests/test_store.py::test_store_open_repairs_fragment_that_overclaims_its_columns` reproducibly fails before this plan. Do not alter it.

---

### Task 1: Durable Hook Outbox

**Files:**
- Create: `core/hook_outbox.py`
- Test: `tests/test_hook_outbox.py`

**Interfaces:**
- Produces `HookDelivery` frozen dataclass: `id`, `event_id`, `hook_name`, `event`, `hook`, `status`, `attempts`, `next_attempt_at`, `last_outcome`, `last_error`, `created_at`, `updated_at`.
- Produces `HookOutbox(index_root)` with `enqueue(event, hook)`, `due(limit, now=None)`, `complete(delivery)`, `retry(delivery, outcome, error, now=None)`, and `redrive_required(delivery, outcome, error)`.
- `enqueue()` persists JSON event and sanitized hook configuration before network use; store `secret_env` name but never its resolved value.
- `retry()` increments attempts and schedules exact configured delay. It switches attempt five to `redrive_required`.
- `due()` returns only pending rows whose `next_attempt_at` is due; terminal rows never return.

- [ ] **Step 1: Write failing outbox tests**

```python
def test_outbox_reopens_and_returns_persisted_due_delivery(tmp_path):
    event = {"event": "document.indexed", "event_id": "evt-1", "doc_id": "documents::a", "text": "private"}
    hook = {"name": "cds", "url": "http://hook", "secret_env": "HOOK_SECRET"}
    HookOutbox(tmp_path).enqueue(event, hook)

    due = HookOutbox(tmp_path).due(limit=1)

    assert due[0].event_id == "evt-1"
    assert due[0].event["text"] == "private"
    assert "HOOK_SECRET" in due[0].hook["secret_env"]


def test_retry_uses_bounded_backoff_then_requires_redrive(tmp_path):
    outbox = HookOutbox(tmp_path)
    delivery = outbox.enqueue({"event_id": "evt-1"}, {"name": "cds"})

    for expected_attempt, delay in enumerate((1, 5, 30, 120), start=1):
        delivery = outbox.retry(delivery, "transport_error", "timeout", now=100)
        assert delivery.attempts == expected_attempt
        assert delivery.next_attempt_at == 100 + delay
    delivery = outbox.retry(delivery, "transport_error", "timeout", now=100)
    assert delivery.status == "redrive_required"
```

- [ ] **Step 2: Run tests; verify RED**

Run: `pytest tests/test_hook_outbox.py -q`

Expected: FAIL because `core.hook_outbox` does not exist.

- [ ] **Step 3: Implement minimal SQLite outbox**

Define `RETRY_DELAYS_SECONDS = (1, 5, 30, 120)`. Implement every interface named above with a WAL SQLite database under `index_root`, `PRAGMA synchronous=FULL`, row revision guards, and atomic `BEGIN IMMEDIATE` transitions. Never write secrets to row data or error text.

- [ ] **Step 4: Run tests; verify GREEN**

Run: `pytest tests/test_hook_outbox.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/hook_outbox.py tests/test_hook_outbox.py
git commit -m "feat: add durable hook outbox"
```

### Task 2: Semantic HTTP Hook Results

**Files:**
- Modify: `hooks/http.py:12-61`
- Modify: `hooks/dispatcher.py:12-56`
- Modify: `hooks/events.py:24-49`
- Modify: `config.yaml.example:170-178`
- Modify: `config.staging.yaml:151-170`
- Test: `tests/hooks/test_dispatcher.py`
- Test: `tests/hooks/test_events.py`

**Interfaces:**
- Produces immutable `HookSendResult` with `accepted`, `outcome`, `retryable`, `http_status`, and safe `error`.
- `send_http_event(hook, event)` returns `HookSendResult`, reads a 2xx JSON response only when `accepted_statuses` is configured, and never returns body text as error detail.
- `build_document_indexed_event()` includes `event_id` from `uuid.uuid4()` once per call.
- `matching_hooks(config, event_name)` returns validated configured HTTP hook dictionaries matching the event.

- [ ] **Step 1: Write failing semantic-ack and event-ID tests**

```python
def test_send_http_event_rejects_cds_no_match_even_with_http_200(monkeypatch):
    class Response:
        status = 200

        def read(self):
            return b'{"status":"no_match"}'

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("hooks.http.urllib.request.urlopen", lambda *args, **kwargs: Response())
    result = send_http_event(
        {"name": "cds", "url": "http://hook", "accepted_statuses": ["updated", "duplicate"]},
        {"event": "document.indexed", "event_id": "evt-1"},
    )
    assert result.accepted is False
    assert result.outcome == "no_match"
    assert result.retryable is False


def test_document_indexed_event_has_unique_opaque_event_id():
    event = build_document_indexed_event(
        doc_id="documents::000hF",
        source_name="documents",
        source_type="img",
        rel_path="email-attachments/photo@000hF@.jpg",
        abs_path="/data/documents/email-attachments/photo@000hF@.jpg",
        text="private",
        metadata={},
        chunks=[],
    )
    assert UUID(event["event_id"]).version == 4
```

- [ ] **Step 2: Run tests; verify RED**

Run: `pytest tests/hooks/test_dispatcher.py tests/hooks/test_events.py -q`

Expected: FAIL because result object and `event_id` do not exist.

- [ ] **Step 3: Implement structured sender and hook selection**

```python
@dataclass(frozen=True)
class HookSendResult:
    accepted: bool
    outcome: str
    retryable: bool
    http_status: int | None = None
    error: str | None = None
```

Read response JSON only for a narrowed `accepted_statuses` contract. Map `no_match`, `ambiguous`, and `correlation_mismatch` to non-retryable semantic rejection; map malformed/unexpected 2xx body and transport/5xx failure to retryable failure. Preserve generic any-2xx behavior. Add `accepted_statuses: ["updated", "duplicate"]` to the CDS examples only. Keep dispatcher compatibility tests by adapting them to result objects rather than warning strings.

- [ ] **Step 4: Run tests; verify GREEN**

Run: `pytest tests/hooks/test_dispatcher.py tests/hooks/test_events.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hooks/http.py hooks/dispatcher.py hooks/events.py config.yaml.example config.staging.yaml tests/hooks/test_dispatcher.py tests/hooks/test_events.py
git commit -m "feat: require semantic hook acceptance"
```

### Task 3: Delivery Service and Safe Outcomes

**Files:**
- Create: `hooks/delivery.py`
- Test: `tests/hooks/test_delivery.py`

**Interfaces:**
- Produces `queue_event(config, event, outbox) -> int`; returns matching hooks persisted for one event.
- Produces `drain_due(outbox, *, limit, sender=send_http_event, logger=None, now=None) -> dict[str, int]`; returned keys are `accepted`, `retry_pending`, and `redrive_required`.
- `drain_due()` passes the stored hook and event to the injected sender. It never exposes request payload or response body in logs.

- [ ] **Step 1: Write failing delivery tests**

```python
def test_timeout_stays_durable_then_updated_response_completes(tmp_path):
    outbox = HookOutbox(tmp_path)
    config = {"enabled": True, "hooks": [{"name": "cds", "events": ["document.indexed"]}]}
    event = {"event": "document.indexed", "event_id": "evt-1", "doc_id": "documents::a", "text": "do-not-log"}
    queue_event(config, event, outbox)

    first = drain_due(outbox, limit=1, sender=lambda hook, item: HookSendResult(False, "timeout", True), now=100)
    second = drain_due(
        outbox,
        limit=1,
        sender=lambda hook, item: HookSendResult(True, "updated", False, 200),
        now=101,
    )

    assert first == {"accepted": 0, "retry_pending": 1, "redrive_required": 0}
    assert second == {"accepted": 1, "retry_pending": 0, "redrive_required": 0}
    assert outbox.due(limit=1, now=101) == []


def test_no_match_becomes_safe_terminal_redrive(tmp_path, caplog):
    outbox = HookOutbox(tmp_path)
    event = {"event": "document.indexed", "event_id": "evt-1", "doc_id": "documents::a", "text": "do-not-log"}
    outbox.enqueue(event, {"name": "cds", "secret_env": "HOOK_SECRET"})

    drain_due(outbox, limit=1, sender=lambda hook, item: HookSendResult(False, "no_match", False, 200))

    assert "evt-1" in caplog.text
    assert "no_match" in caplog.text
    assert "do-not-log" not in caplog.text
    assert "HOOK_SECRET" not in caplog.text
```

- [ ] **Step 2: Run tests; verify RED**

Run: `pytest tests/hooks/test_delivery.py -q`

Expected: FAIL because `hooks.delivery` does not exist.

- [ ] **Step 3: Implement delivery state transitions**

Persist only matching HTTP hooks. For each due row, `complete()` accepted results; `retry()` retryable results; and `redrive_required()` deterministic semantic results. Emit one safe log record containing event ID, doc ID, hook name, attempt, outcome, HTTP status, and retry state. Do not use the event dictionary, hook URL, headers, payload, or raw response in log formatting.

- [ ] **Step 4: Run tests; verify GREEN**

Run: `pytest tests/hooks/test_delivery.py tests/test_hook_outbox.py tests/hooks/test_dispatcher.py tests/hooks/test_events.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hooks/delivery.py tests/hooks/test_delivery.py
git commit -m "feat: deliver hooks through durable outbox"
```

### Task 4: Pipeline and Scheduler Integration

**Files:**
- Modify: `flow_index_vault.py:116-117,2589-2603,4405-4446`
- Modify: `mcp_server.py:2712-2733`
- Modify: `tests/test_hook_integration.py`
- Modify: `tests/test_targeted_index.py`
- Modify: `tests/test_index_scheduler.py`
- Modify: `docs/TESTING.md`

**Interfaces:**
- Produces `drain_hook_outbox(config_path="config.yaml", *, limit=None) -> dict[str, int]` from `flow_index_vault.py`.
- `process_doc_task()` queues matching hook rows after successful Lance write, then immediately drains due rows. A failed callback never changes indexing outcome.
- Scheduler's existing short-interval callback returns a dictionary with `index_requests` and `hook_deliveries` keys after invoking both drains.

- [ ] **Step 1: Write failing pipeline and scheduler tests**

```python
def test_successful_upsert_keeps_failed_callback_for_later_retry(tmp_path, monkeypatch):
    store = MagicMock()
    _setup_runtime(
        store,
        index_root=tmp_path,
        event_hooks={"enabled": True, "hooks": [{"name": "cds", "events": ["document.indexed"]}]},
    )
    monkeypatch.setattr(
        "flow_index_vault.drain_due",
        lambda outbox, **kwargs: {"accepted": 0, "retry_pending": 1, "redrive_required": 0},
    )

    process_doc_task.fn(_doc())

    assert HookOutbox(tmp_path).due(limit=1)[0].event["doc_id"] == "documents::000hF"


def test_scheduler_short_drain_runs_index_and_hook_queues(monkeypatch):
    monkeypatch.setattr("flow_index_vault.drain_index_queue", lambda path: {"status": "empty"})
    monkeypatch.setattr("flow_index_vault.drain_hook_outbox", lambda path: {"accepted": 0})

    scheduler = build_index_scheduler({"scheduler": {"enabled": True}}, "test-config.yaml")

    assert scheduler.tick(0)[0][1] == {"index_requests": {"status": "empty"}, "hook_deliveries": {"accepted": 0}}
```

Extend `_setup_runtime()` to accept `index_root` and `event_hooks`; preserve defaults used by existing tests.

- [ ] **Step 2: Run tests; verify RED**

Run: `pytest tests/test_hook_integration.py tests/test_index_scheduler.py tests/test_targeted_index.py -q`

Expected: FAIL because pipeline has no outbox queue or hook drain.

- [ ] **Step 3: Implement pipeline integration**

After Lance write, build the event once, instantiate `HookOutbox(config["index_root"])`, and call `queue_event()`. Immediately invoke the same `drain_due()` path used by `drain_hook_outbox()`. Keep task warning aggregation only for safe outcome text. In `build_index_scheduler()`, compose both existing drain functions into one callback without modifying `IndexScheduler`. Update `docs/TESTING.md` with forced targeted re-index redrive and disk-capacity runbook steps.

- [ ] **Step 4: Run tests; verify GREEN**

Run: `pytest tests/test_hook_outbox.py tests/hooks/test_dispatcher.py tests/hooks/test_events.py tests/hooks/test_delivery.py tests/test_hook_integration.py tests/test_index_scheduler.py tests/test_targeted_index.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add flow_index_vault.py mcp_server.py tests/test_hook_integration.py tests/test_targeted_index.py tests/test_index_scheduler.py docs/TESTING.md
git commit -m "feat: retry document indexed callbacks"
```

### Task 5: Verification

- [ ] **Step 1: Run focused callback suite**

Run: `pytest tests/test_hook_outbox.py tests/hooks/test_dispatcher.py tests/hooks/test_events.py tests/hooks/test_delivery.py tests/test_hook_integration.py tests/test_index_scheduler.py tests/test_targeted_index.py -q`

Expected: PASS.

- [ ] **Step 2: Run development gate**

Run: `make gate-fast`

Expected: all Issue #112 tests PASS; separate baseline Lance repair failure stays reported if it still fails.
