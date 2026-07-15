"""Bounded submission and opt-in indexer memory observability contracts."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch


class RecordingLogger:
    def __init__(self):
        self.messages: list[tuple[str, tuple]] = []

    def info(self, message, *args):
        self.messages.append((message, args))


def test_disabled_memory_observer_has_zero_measurement_work():
    from memory_observer import MemoryObserver

    calls: list[str] = []
    observer = MemoryObserver(
        enabled=False,
        logger=RecordingLogger(),
        rss_reader=lambda: calls.append("rss") or 1,
        arrow_reader=lambda: calls.append("arrow") or 2,
    )

    observer.sample("phase_start", phase="scan")

    assert calls == []


def test_enabled_memory_observer_logs_rss_arrow_and_deltas():
    from memory_observer import MemoryObserver

    logger = RecordingLogger()
    rss = iter([100, 140])
    arrow = iter([10, 25])
    observer = MemoryObserver(
        enabled=True,
        logger=logger,
        rss_reader=lambda: next(rss),
        arrow_reader=lambda: next(arrow),
    )

    observer.sample("doc_start", phase="process", doc_id="documents::abc")
    observer.sample("doc_finish", phase="process", doc_id="documents::abc", outcome="ok")

    payloads = [json.loads(args[0]) for message, args in logger.messages]
    assert payloads[0]["rss_bytes"] == 100
    assert payloads[0]["arrow_allocated_bytes"] == 10
    assert payloads[1]["rss_delta_bytes"] == 40
    assert payloads[1]["arrow_delta_bytes"] == 15
    assert payloads[1]["doc_id"] == "documents::abc"
    assert payloads[1]["outcome"] == "ok"


def test_concurrent_doc_deltas_use_each_docs_own_start_baseline():
    from memory_observer import MemoryObserver

    logger = RecordingLogger()
    rss = iter([100, 200, 150, 260])
    arrow = iter([10, 20, 15, 30])
    observer = MemoryObserver(
        enabled=True,
        logger=logger,
        rss_reader=lambda: next(rss),
        arrow_reader=lambda: next(arrow),
    )

    observer.sample("doc_start", phase="process", doc_id="documents::a")
    observer.sample("doc_start", phase="process", doc_id="documents::b")
    observer.sample("doc_finish", phase="process", doc_id="documents::a", outcome="ok")
    observer.sample("doc_finish", phase="process", doc_id="documents::b", outcome="ok")

    payloads = [json.loads(args[0]) for _, args in logger.messages]
    assert payloads[2]["rss_delta_bytes"] == 50
    assert payloads[2]["arrow_delta_bytes"] == 5
    assert payloads[3]["rss_delta_bytes"] == 60
    assert payloads[3]["arrow_delta_bytes"] == 10


def test_measure_emits_subphase_boundaries_with_one_baseline():
    from memory_observer import MemoryObserver

    logger = RecordingLogger()
    rss = iter([100, 175])
    arrow = iter([10, 30])
    observer = MemoryObserver(
        enabled=True,
        logger=logger,
        rss_reader=lambda: next(rss),
        arrow_reader=lambda: next(arrow),
    )

    with observer.measure("extract", doc_id="documents::abc"):
        pass

    payloads = [json.loads(args[0]) for _, args in logger.messages]
    assert [payload["event"] for payload in payloads] == [
        "subphase_start",
        "subphase_finish",
    ]
    assert {payload["subphase"] for payload in payloads} == {"extract"}
    assert payloads[1]["rss_delta_bytes"] == 75
    assert payloads[1]["arrow_delta_bytes"] == 20
    assert payloads[1]["outcome"] == "ok"


def test_bounded_executor_map_consumes_at_most_window_before_completion():
    from flow_index_vault import _bounded_executor_map

    consumed: list[int] = []
    started = threading.Barrier(3)
    release = threading.Event()
    result: list[int] = []

    def items():
        for value in range(20):
            consumed.append(value)
            yield value

    def work(value):
        if value < 2:
            started.wait(timeout=5)
            release.wait(timeout=5)
        return value

    def run():
        with ThreadPoolExecutor(max_workers=2) as executor:
            result.extend(_bounded_executor_map(executor, work, items(), max_pending=2))

    runner = threading.Thread(target=run)
    runner.start()
    started.wait(timeout=5)
    try:
        assert consumed == [0, 1]
    finally:
        release.set()
        runner.join(timeout=5)

    assert runner.is_alive() is False
    assert result == list(range(20))


def test_process_docs_emits_per_doc_memory_samples():
    import flow_index_vault as flow

    events: list[tuple[str, dict]] = []

    class Observer:
        def sample(self, event, **fields):
            events.append((event, fields))

    docs = [{"doc_id": "documents::abc", "mtime": 1, "rel_path": "a.md"}]
    old_runtime = dict(flow._RUNTIME)
    flow._RUNTIME.clear()
    flow._RUNTIME.update(
        {
            "memory_observer": Observer(),
            "degraded_lock": threading.Lock(),
            "degraded_now": {},
            "degraded_clean": set(),
            "skip_now": {},
            "skip_clean": set(),
        }
    )
    try:
        with patch("flow_index_vault.process_doc_task", return_value=None):
            assert flow._process_docs(docs, concurrency=1) == []
    finally:
        flow._RUNTIME.clear()
        flow._RUNTIME.update(old_runtime)

    assert events == [
        ("doc_start", {"phase": "process", "doc_id": "documents::abc"}),
        (
            "doc_finish",
            {"phase": "process", "doc_id": "documents::abc", "outcome": "ok"},
        ),
    ]
