"""CDS HTTP outcomes and durable delivery remain consistent across restarts."""

import subprocess
import sys

import pytest

from core.hook_outbox import HookOutbox
from hooks.delivery import drain_due, queue_event


@pytest.fixture
def queued_delivery(tmp_path, http_peer, monkeypatch):
    monkeypatch.setenv("CONTRACT_CDS_URL", http_peer.url + "/document-indexed")
    monkeypatch.setenv("CONTRACT_CDS_SECRET", "contract-only-secret")
    hook = {"name": "cds", "type": "http", "events": ["document.indexed"],
            "url": "${CONTRACT_CDS_URL}", "secret_env": "CONTRACT_CDS_SECRET",
            "accepted_statuses": ["updated", "duplicate"]}
    event = {"event": "document.indexed", "event_id": "evt-contract-1",
             "doc_id": "documents::one", "rel_path": "one.md", "metadata": {},
             "text": "Invoice total is 42 dollars.",
             "chunks": [{"text": "Invoice total is 42 dollars.", "loc": "c:0"}]}
    [delivery] = queue_event({"enabled": True, "hooks": [hook]}, event, HookOutbox(tmp_path))
    return event, hook, delivery.created_at


def test_callback_retry_after_restart_preserves_payload_and_event_identity(tmp_path, http_peer, queued_delivery):
    event, _, now = queued_delivery
    http_peer.respond = lambda request: (503, {"status": "unavailable"})
    assert drain_due(HookOutbox(tmp_path), limit=1, now=now) == {
        "accepted": 0, "retry_pending": 1, "redrive_required": 0,
    }
    http_peer.respond = lambda request: (200, {"status": "updated"})
    assert drain_due(HookOutbox(tmp_path), limit=1, now=now + 2) == {
        "accepted": 1, "retry_pending": 0, "redrive_required": 0,
    }
    assert drain_due(HookOutbox(tmp_path), limit=1, now=now + 100)["accepted"] == 0
    assert len(http_peer.requests) == 2, "acknowledged event must not be redelivered after restart"
    for request in http_peer.requests:
        assert request["body"] == event
        assert request["path"] == "/document-indexed"
        assert request["headers"]["X-Rag-Hook-Secret"] == "contract-only-secret"


@pytest.mark.parametrize("status", ["no_match", "ambiguous", "correlation_mismatch"])
def test_http_200_cannot_acknowledge_wrong_cds_identity(tmp_path, http_peer, queued_delivery, status):
    _, _, now = queued_delivery
    http_peer.respond = lambda request: (200, {"status": status})
    assert drain_due(HookOutbox(tmp_path), limit=1, now=now) == {
        "accepted": 0, "retry_pending": 0, "redrive_required": 1,
    }
    assert HookOutbox(tmp_path).due(limit=1, now=now + 1000) == []


@pytest.mark.parametrize("body", [b"not-json", {"status": "unknown"}, []])
def test_unrecognized_ack_remains_durable(tmp_path, http_peer, queued_delivery, body):
    event, _, now = queued_delivery
    http_peer.respond = lambda request: (200, body)
    assert drain_due(HookOutbox(tmp_path), limit=1, now=now)["retry_pending"] == 1
    [pending] = HookOutbox(tmp_path).due(limit=1, now=now + 2)
    assert pending.event == event
    assert pending.attempts == 1


def test_crash_after_receiver_accepts_replays_same_event_then_finishes(tmp_path, http_peer, queued_delivery):
    event, _, now = queued_delivery
    http_peer.respond = lambda request: (
        200, {"status": "updated" if len(http_peer.requests) == 1 else "duplicate"}
    )
    process = subprocess.run([sys.executable, "-c", """
import os, sys
from core.hook_outbox import HookOutbox
from hooks.http import send_http_event
box = HookOutbox(sys.argv[1])
[pending] = box.due(limit=1)
claimed = box.claim(pending)
assert claimed is not None
assert send_http_event(claimed.hook, claimed.event).accepted
os._exit(17)  # receiver committed; sender never recorded acknowledgment
""", str(tmp_path)], timeout=15, capture_output=True, text=True)
    assert process.returncode == 17, process.stderr
    assert drain_due(HookOutbox(tmp_path), limit=1, now=now + 1000) == {
        "accepted": 1, "retry_pending": 0, "redrive_required": 0,
    }
    assert [request["body"] for request in http_peer.requests] == [event, event]
    assert HookOutbox(tmp_path).due(limit=1, now=now + 2000) == []
