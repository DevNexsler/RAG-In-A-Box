"""Generated operation histories exercise the durable queue's public contract."""

from core.index_request_queue import IndexRequestQueue


def test_completed_snapshot_cannot_ack_a_later_incarnation(tmp_path):
    queue = IndexRequestQueue(tmp_path)
    old = queue.enqueue("chunks", "documents", "invoice.md")
    assert queue.complete(old)
    current = queue.enqueue("chunks", "documents", "invoice.md", force=True)
    assert not queue.complete(old), "a replayed ACK must not delete new work"
    assert not queue.fail(old, "stale worker"), "a replayed failure must not alter new work"
    assert queue.pending("chunks", limit=10) == [current]


import random
import sqlite3

import pytest


@pytest.mark.parametrize("seed", range(16))
def test_generated_queue_histories_preserve_latest_work(tmp_path, seed):
    rng = random.Random(seed)
    queue = IndexRequestQueue(tmp_path)
    model, snapshots, history = {}, [], []
    generation = 0
    for _ in range(160):
        operation = rng.choice(["enqueue", "enqueue", "complete", "fail", "reopen"])
        key = (rng.choice(["chunks", "other"]), rng.choice(["mail", "documents"]), rng.choice(["a.md", "b.md"]))
        history.append((operation, key))
        if operation == "enqueue":
            force = rng.choice([False, True])
            prior = model.get(key, {"force": False, "attempts": 0})
            generation += 1
            model[key] = {"generation": generation, "force": prior["force"] or force, "attempts": prior["attempts"]}
            snapshots.append((key, generation, queue.enqueue(*key, force=force)))
        elif operation == "reopen":
            queue = IndexRequestQueue(tmp_path)
        elif snapshots:
            key, observed_generation, request = rng.choice(snapshots)
            current = model.get(key)
            owns_work = current is not None and current["generation"] == observed_generation
            if operation == "complete":
                assert queue.complete(request) == owns_work, (seed, history)
                if owns_work:
                    del model[key]
            else:
                assert queue.fail(request, "provider offline") == owns_work, (seed, history)
                if owns_work:
                    current["attempts"] += 1
        actual = {}
        for table in ("chunks", "other"):
            for row in queue.pending(table, limit=100):
                actual[(table, row.source_name, row.target)] = (row.force, row.attempts)
        assert actual == {key: (value["force"], value["attempts"]) for key, value in model.items()}, (seed, history)


def test_legacy_queue_upgrade_preserves_work_and_rejects_old_snapshots(tmp_path):
    # Historical on-disk schema is external input to the upgrade boundary.
    with sqlite3.connect(tmp_path / "index-requests.sqlite3") as db:
        db.execute("""CREATE TABLE index_requests (
            id INTEGER PRIMARY KEY, table_name TEXT NOT NULL, source_name TEXT NOT NULL,
            target TEXT NOT NULL, force INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0, revision INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL, last_error TEXT,
            UNIQUE(table_name, source_name, target))""")
        db.execute("INSERT INTO index_requests VALUES (1,'chunks','documents','legacy.md',1,'pending',3,7,'old','old','offline')")
    queue = IndexRequestQueue(tmp_path)
    [legacy] = queue.pending("chunks", limit=10)
    assert (legacy.target, legacy.force, legacy.attempts, legacy.revision, legacy.last_error) == ("legacy.md", True, 3, 7, "offline")
    assert queue.complete(legacy)
    current = queue.enqueue("chunks", "documents", "legacy.md")
    assert not queue.complete(legacy)
    assert IndexRequestQueue(tmp_path).pending("chunks", limit=10) == [current]
