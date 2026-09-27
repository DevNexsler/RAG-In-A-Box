"""Delayed callback workers cannot mutate a newer delivery incarnation."""
from core.hook_outbox import HookOutbox


def test_delayed_delivery_cannot_ack_replayed_event(tmp_path):
    outbox = HookOutbox(tmp_path)
    event = {"event_id": "same-event", "event_type": "document.indexed"}
    hook = {"name": "cds", "url": "http://localhost/callback"}
    old = outbox.claim(outbox.enqueue(event, hook))
    assert outbox.complete(old)
    current = outbox.claim(outbox.enqueue(event, hook))
    assert outbox.complete(old) is None
    assert outbox.retry(old, "http_error", "stale failure") is None
    assert outbox.complete(current)


def test_legacy_outbox_upgrade_preserves_delivery_and_claims(tmp_path):
    import json
    import sqlite3
    from concurrent.futures import ThreadPoolExecutor
    event = {'event_id': 'legacy', 'event': 'document.indexed'}
    hook = {'name': 'cds', 'url': 'http://localhost/callback'}
    with sqlite3.connect(tmp_path / 'hook-outbox.sqlite3') as db:
        db.execute('''CREATE TABLE hook_deliveries (
            id INTEGER PRIMARY KEY, event_id TEXT NOT NULL, hook_name TEXT NOT NULL,
            event_json TEXT NOT NULL, hook_json TEXT NOT NULL, status TEXT NOT NULL,
            attempts INTEGER NOT NULL, next_attempt_at REAL NOT NULL, last_outcome TEXT,
            last_error TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL,
            revision INTEGER NOT NULL, UNIQUE(event_id,hook_name))''')
        db.execute("INSERT INTO hook_deliveries VALUES (1,'legacy','cds',?,?,'pending',2,0,'http_error','delivery_error',1,2,7)",
                   (json.dumps(event), json.dumps(hook)))
    with ThreadPoolExecutor(max_workers=4) as pool:
        snapshots = list(pool.map(lambda _: HookOutbox(tmp_path).due(10), range(8)))
    assert all(snapshot == snapshots[0] for snapshot in snapshots)
    [legacy] = snapshots[0]
    assert (legacy.event, legacy.attempts, legacy.revision) == (event, 2, 7)
    outbox = HookOutbox(tmp_path)
    owned = outbox.claim(legacy)
    assert owned is not None
    assert outbox.claim(legacy) is None
    assert outbox.complete(owned)
    assert outbox.due(10) == []


def test_concurrent_process_startup_keeps_all_callbacks(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import subprocess
    import sys

    code = '''
import sys
from core.hook_outbox import HookOutbox
box = HookOutbox(sys.argv[1])
box.enqueue({'event_id':sys.argv[2], 'event':'document.indexed'}, {'name':'cds','url':'http://localhost/callback'})
'''
    def start(index):
        return subprocess.run([sys.executable, '-c', code, str(tmp_path), str(index)],
                              capture_output=True, text=True, timeout=20)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(start, range(8)))
    assert all(result.returncode == 0 for result in results), [result.stderr for result in results]
    assert {delivery.event_id for delivery in HookOutbox(tmp_path).due(100)} == {str(i) for i in range(8)}
