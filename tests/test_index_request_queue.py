"""Durability and coalescing contract for targeted index requests."""

import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor

from core.index_request_queue import IndexRequestQueue


def _enqueue_in_child(index_root, target, force=False):
    IndexRequestQueue(index_root).enqueue(
        "chunks", "documents", target, force=force
    )


def _snapshot_then_crash(index_root, started):
    queue = IndexRequestQueue(index_root)
    assert queue.pending("chunks", limit=1)
    started.set()
    os._exit(17)


def test_enqueue_normalizes_and_coalesces_with_force_escalation(tmp_path):
    queue = IndexRequestQueue(tmp_path)

    first = queue.enqueue(
        "chunks", "documents", r"email-attachments\\team//report.pdf"
    )
    second = queue.enqueue(
        "chunks", "documents", "email-attachments/team/report.pdf", force=True
    )
    third = queue.enqueue(
        "chunks", "documents", "email-attachments/team/report.pdf", force=False
    )

    assert first.target == "email-attachments/team/report.pdf"
    assert [item.id for item in queue.pending("chunks", limit=10)] == [first.id]
    assert second.id == third.id == first.id
    assert (first.revision, second.revision, third.revision) == (1, 2, 3)
    assert third.force is True
    assert third.status == "pending"


def test_completion_and_failure_are_revision_guarded(tmp_path):
    queue = IndexRequestQueue(tmp_path)
    old = queue.enqueue("chunks", "documents", "a.pdf")
    current = queue.enqueue("chunks", "documents", "a.pdf", force=True)

    assert queue.complete(old) is False
    assert queue.fail(old, "old failure") is False
    assert queue.pending("chunks", limit=10) == [current]

    assert queue.fail(current, "provider offline") is True
    failed = queue.pending("chunks", limit=10)[0]
    assert failed.attempts == 1
    assert failed.last_error == "provider offline"
    assert failed.revision == current.revision
    assert queue.complete(failed) is True
    assert queue.pending("chunks", limit=10) == []


def test_pending_prioritizes_current_key_and_honors_limit(tmp_path):
    queue = IndexRequestQueue(tmp_path)
    queue.enqueue("chunks", "documents", "a.pdf")
    queue.enqueue("chunks", "documents", "b.pdf")
    current = queue.enqueue("chunks", "mail", "c.pdf")

    pending = queue.pending(
        "chunks",
        limit=2,
        prioritize=(current.source_name, current.target),
    )

    assert [(item.source_name, item.target) for item in pending] == [
        ("mail", "c.pdf"),
        ("documents", "a.pdf"),
    ]


def test_queue_survives_writer_process_exit(tmp_path):
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_enqueue_in_child,
        args=(str(tmp_path), "persist.pdf", True),
    )
    process.start()
    process.join(10)

    assert process.exitcode == 0
    [request] = IndexRequestQueue(tmp_path).pending("chunks", limit=10)
    assert request.target == "persist.pdf"
    assert request.force is True


def test_queue_survives_crash_after_drain_snapshot(tmp_path):
    queue = IndexRequestQueue(tmp_path)
    original = queue.enqueue("chunks", "documents", "crash.pdf")
    context = multiprocessing.get_context("spawn")
    started = context.Event()
    process = context.Process(
        target=_snapshot_then_crash,
        args=(str(tmp_path), started),
    )
    process.start()
    assert started.wait(10)
    process.join(10)

    assert process.exitcode == 17
    assert IndexRequestQueue(tmp_path).pending("chunks", limit=10) == [original]


def test_concurrent_enqueue_is_atomic_and_never_downgrades_force(tmp_path):
    def enqueue(index: int):
        return IndexRequestQueue(tmp_path).enqueue(
            "chunks",
            "documents",
            "same.pdf",
            force=index == 7,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(enqueue, range(24)))

    [request] = IndexRequestQueue(tmp_path).pending("chunks", limit=10)
    assert {item.id for item in results} == {request.id}
    assert request.revision == 24
    assert request.force is True
