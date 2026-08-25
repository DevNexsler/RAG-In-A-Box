from core.hook_outbox import HookOutbox
from hooks.http import HookSendResult


def test_timeout_stays_durable_then_updated_response_completes(tmp_path, monkeypatch):
    from hooks.delivery import drain_due, queue_event
    import core.hook_outbox

    monkeypatch.setattr(core.hook_outbox.time, "time", lambda: 100)
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
    from hooks.delivery import drain_due

    outbox = HookOutbox(tmp_path)
    event = {"event": "document.indexed", "event_id": "evt-1", "doc_id": "documents::a", "text": "do-not-log"}
    outbox.enqueue(
        event,
        {"name": "cds", "url": "https://private-hook.example/delivery", "secret_env": "HOOK_SECRET"},
    )
    caplog.set_level("INFO")

    result = drain_due(outbox, limit=1, sender=lambda hook, item: HookSendResult(False, "no_match", False, 200))

    assert result == {"accepted": 0, "retry_pending": 0, "redrive_required": 1}
    assert "evt-1" in caplog.text
    assert "no_match" in caplog.text
    assert "do-not-log" not in caplog.text
    assert "HOOK_SECRET" not in caplog.text
    assert "private-hook.example" not in caplog.text


def test_queue_event_persists_only_matching_http_hooks_and_preserves_event_id(tmp_path):
    from hooks.delivery import drain_due, queue_event

    outbox = HookOutbox(tmp_path)
    event = {"event": "document.indexed", "event_id": "evt-immutable", "doc_id": "documents::a"}
    config = {
        "enabled": True,
        "hooks": [
            {"name": "match", "type": "http", "events": ["document.indexed"]},
            {"name": "other-event", "events": ["other.event"]},
            {"name": "non-http", "type": "queue"},
        ],
    }
    sent = []

    assert queue_event(config, event, outbox) == 1
    assert drain_due(outbox, limit=2, sender=lambda hook, item: sent.append((hook, item)) or HookSendResult(True, "accepted", False)) == {
        "accepted": 1,
        "retry_pending": 0,
        "redrive_required": 0,
    }
    assert sent == [({"name": "match", "type": "http", "events": ["document.indexed"]}, event)]


def test_retryable_failures_redrive_after_five_sends(tmp_path, monkeypatch):
    from hooks.delivery import drain_due
    import core.hook_outbox

    monkeypatch.setattr(core.hook_outbox.time, "time", lambda: 100)
    outbox = HookOutbox(tmp_path)
    outbox.enqueue({"event_id": "evt-1", "doc_id": "documents::a"}, {"name": "cds"})

    for now in (100, 101, 106, 136):
        assert drain_due(outbox, limit=1, sender=lambda hook, item: HookSendResult(False, "timeout", True), now=now) == {
            "accepted": 0,
            "retry_pending": 1,
            "redrive_required": 0,
        }
    assert drain_due(outbox, limit=1, sender=lambda hook, item: HookSendResult(False, "timeout", True), now=256) == {
        "accepted": 0,
        "retry_pending": 0,
        "redrive_required": 1,
    }
    assert outbox.due(limit=1, now=1_000) == []


def test_overlapping_drains_send_one_delivery_once(tmp_path):
    """Fails if a drain sends a delivery another drain already has in flight."""
    from hooks.delivery import drain_due

    outbox = HookOutbox(tmp_path)
    outbox.enqueue({"event_id": "evt-1", "doc_id": "documents::a"}, {"name": "cds"})
    sent: list[str] = []

    def sender(hook, event):
        sent.append(event["event_id"])
        if len(sent) == 1:
            # A second drainer (another index worker, or the scheduler tick)
            # starts while this send is still in flight.
            drain_due(HookOutbox(tmp_path), limit=64, sender=sender)
        return HookSendResult(True, "accepted", False, 200)

    result = drain_due(outbox, limit=64, sender=sender)

    assert sent == ["evt-1"]
    assert result == {"accepted": 1, "retry_pending": 0, "redrive_required": 0}
