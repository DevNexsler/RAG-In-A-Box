from core.hook_outbox import HookOutbox


def test_outbox_reopens_and_returns_persisted_due_delivery(tmp_path):
    """Fails if event data is not durably persisted under index root."""
    event = {
        "event": "document.indexed",
        "event_id": "evt-1",
        "doc_id": "documents::a",
        "text": "private",
    }
    hook = {"name": "cds", "url": "http://hook", "secret_env": "HOOK_SECRET"}
    HookOutbox(tmp_path).enqueue(event, hook)

    due = HookOutbox(tmp_path).due(limit=1)

    assert due[0].event_id == "evt-1"
    assert due[0].event["text"] == "private"
    assert due[0].hook["secret_env"] == "HOOK_SECRET"


def test_outbox_keeps_one_delivery_per_event_and_hook(tmp_path):
    """Fails if duplicate enqueue creates duplicate delivery for same hook."""
    outbox = HookOutbox(tmp_path)
    event = {"event_id": "evt-1"}
    hook = {"name": "cds", "secret_env": "HOOK_SECRET", "secret": "never-store"}

    first = outbox.enqueue(event, hook)
    second = outbox.enqueue(event, hook)

    assert second.id == first.id
    assert len(outbox.due(limit=2)) == 1
    assert "secret" not in second.hook
    assert b"never-store" not in (tmp_path / "hook-outbox.sqlite3").read_bytes()


def test_outbox_does_not_persist_nested_secret_config(tmp_path):
    """Fails if persisted hook configuration retains nested secret material."""
    delivery = HookOutbox(tmp_path).enqueue(
        {"event_id": "evt-1"},
        {"name": "cds", "headers": {"authorization": "Bearer never-store"}},
    )

    assert delivery.hook == {"name": "cds"}
    assert b"never-store" not in (tmp_path / "hook-outbox.sqlite3").read_bytes()


def test_outbox_does_not_persist_header_values(tmp_path):
    """Fails if an arbitrary header value reaches persisted hook config."""
    delivery = HookOutbox(tmp_path).enqueue(
        {"event_id": "evt-1"},
        {
            "name": "cds",
            "url": "http://hook",
            "headers": {"X-Signature": "header-secret"},
        },
    )

    assert delivery.hook == {"name": "cds", "url": "http://hook"}
    persisted = (tmp_path / "hook-outbox.sqlite3").read_bytes()
    assert b"header-secret" not in persisted


def test_outbox_does_not_persist_nested_secret_env_names(tmp_path, monkeypatch):
    """Fails if a nested secret environment reference reaches persisted config."""
    monkeypatch.setenv("NESTED_HOOK_SECRET", "environment-secret")
    delivery = HookOutbox(tmp_path).enqueue(
        {"event_id": "evt-1"},
        {"name": "cds", "auth": {"secret_env": "NESTED_HOOK_SECRET"}},
    )

    assert delivery.hook == {"name": "cds"}
    persisted = (tmp_path / "hook-outbox.sqlite3").read_bytes()
    assert b"environment-secret" not in persisted
    assert b"NESTED_HOOK_SECRET" not in persisted


def test_outbox_persists_generic_error_without_secret_text(tmp_path):
    """Fails if untrusted error text is persisted after a failed delivery."""
    outbox = HookOutbox(tmp_path)
    delivery = outbox.enqueue({"event_id": "evt-1"}, {"name": "cds"})

    retried = outbox.retry(delivery, "transport_error", "runtime-secret", now=100)

    assert retried.last_error == "delivery_error"
    assert b"runtime-secret" not in (tmp_path / "hook-outbox.sqlite3").read_bytes()


def test_retry_uses_bounded_backoff_then_requires_redrive(tmp_path):
    """Fails if retries use wrong delay or deliver past fifth failure."""
    outbox = HookOutbox(tmp_path)
    delivery = outbox.enqueue({"event_id": "evt-1"}, {"name": "cds"})

    for expected_attempt, delay in enumerate((1, 5, 30, 120), start=1):
        delivery = outbox.retry(delivery, "transport_error", "timeout", now=100)
        assert delivery.attempts == expected_attempt
        assert delivery.next_attempt_at == 100 + delay
    delivery = outbox.retry(delivery, "transport_error", "timeout", now=100)

    assert delivery.status == "redrive_required"
    assert outbox.due(limit=1, now=1_000) == []


def test_complete_hides_terminal_delivery_and_stale_transition_is_ignored(tmp_path):
    """Fails if completed or stale deliveries can transition again."""
    outbox = HookOutbox(tmp_path)
    delivery = outbox.enqueue({"event_id": "evt-1"}, {"name": "cds"})

    completed = outbox.complete(delivery)

    assert completed.status == "completed"
    assert outbox.due(limit=1, now=1_000) == []
    assert outbox.retry(delivery, "transport_error", "timeout", now=100) is None
