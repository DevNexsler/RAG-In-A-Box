"""Durable queue operations must not depend on garbage collection for capacity."""
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX descriptor limits")
def test_request_queue_releases_connections_between_operations(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import gc
            import resource
            import sys
            from core.index_request_queue import IndexRequestQueue

            _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            limit = 128 if hard == resource.RLIM_INFINITY else min(128, hard)
            resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))
            gc.disable()
            queue = IndexRequestQueue(sys.argv[1])
            for _ in range(200):
                request = queue.enqueue("chunks", "documents", "retry.md")
                assert queue.pending("chunks", limit=10) == [request]
                assert queue.fail(request, "provider offline")
                [retry] = queue.pending("chunks", limit=10)
                assert retry.attempts == 1
                assert queue.complete(retry)
                assert queue.pending("chunks", limit=10) == []
        """), str(tmp_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX descriptor limits")
def test_hook_outbox_releases_connections_between_delivery_attempts(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import gc
            import resource
            import sys
            from core.hook_outbox import HookOutbox

            _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            limit = 128 if hard == resource.RLIM_INFINITY else min(128, hard)
            resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))
            gc.disable()
            outbox = HookOutbox(sys.argv[1])
            for _ in range(200):
                delivery = outbox.enqueue({"event_id": "retry"}, {"name": "cds"})
                assert outbox.due(limit=10) == [delivery]
                claimed = outbox.claim(delivery, now=100)
                assert claimed is not None
                assert outbox.claim(delivery, now=100) is None
                retried = outbox.retry(claimed, "transport_error", "offline", now=200)
                assert retried.attempts == 1
                assert outbox.due(limit=10, now=202) == [retried]
                assert outbox.complete(retried) is not None
                assert outbox.due(limit=10, now=202) == []
        """), str(tmp_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
