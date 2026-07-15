"""Durable background-index run supervision contracts (#0325)."""

from __future__ import annotations

import json
import signal
import subprocess
import threading
import time
from pathlib import Path


class FakeProcess:
    def __init__(self, pid: int = 4242):
        self.pid = pid
        self.returncode: int | None = None
        self._done = threading.Event()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if not self._done.wait(timeout):
            raise subprocess.TimeoutExpired(["indexer"], timeout)
        return self.returncode

    def finish(self, returncode: int):
        self.returncode = returncode
        self._done.set()


def _wait_for(predicate, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("condition was not met before timeout")


def _supervisor(tmp_path: Path, process: FakeProcess, **kwargs):
    from index_run_supervisor import IndexRunSupervisor

    return IndexRunSupervisor(
        tmp_path,
        process_matches=lambda pid: pid == process.pid,
        pid_alive=lambda pid: pid == process.pid and process.poll() is None,
        monitor_interval=0.01,
        **kwargs,
    )


def test_atomic_single_flight_launches_only_one_process(tmp_path):
    """Concurrent MCP calls must perform one check-and-launch transaction."""
    process = FakeProcess()
    launches: list[list[str]] = []

    def popen(command, **kwargs):
        launches.append(list(command))
        return process

    supervisor = _supervisor(tmp_path, process, popen_factory=popen)
    barrier = threading.Barrier(3)
    results: list[dict] = []

    def launch():
        barrier.wait()
        results.append(
            supervisor.start(
                ["python", "-c", "index_vault_flow=True"],
                log_path=tmp_path / "indexer.log",
                source_name=None,
            )
        )

    threads = [threading.Thread(target=launch) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=2)

    assert len(launches) == 1
    assert sorted(result["status"] for result in results) == ["already_running", "started"]
    state = json.loads((tmp_path / "index_run_state.json").read_text())
    assert state["current"]["status"] == "running"
    assert state["current"]["pid"] == process.pid
    assert (tmp_path / "indexer.pid").read_text() == str(process.pid)

    process.finish(0)
    _wait_for(lambda: supervisor.snapshot().get("current") is None)


def test_monitor_records_success_and_peak_rss_atomically(tmp_path):
    process = FakeProcess()
    samples = iter([10_000, 30_000, 20_000])
    last = {"rss": 20_000}

    def rss_reader(_pid):
        try:
            last["rss"] = next(samples)
        except StopIteration:
            pass
        return last["rss"]

    supervisor = _supervisor(
        tmp_path,
        process,
        popen_factory=lambda *a, **k: process,
        rss_reader=rss_reader,
    )
    result = supervisor.start(["python", "indexer"], log_path=tmp_path / "indexer.log")
    assert result["status"] == "started"

    _wait_for(
        lambda: supervisor.snapshot()["current"].get("peak_rss_bytes", 0) >= 30_000
    )
    process.finish(0)
    state = _wait_for(
        lambda: (
            snapshot
            if (snapshot := supervisor.snapshot()).get("current") is None
            else None
        )
    )

    assert state["last_attempt"]["status"] == "succeeded"
    assert state["last_attempt"]["exit_code"] == 0
    assert state["last_attempt"]["termination_signal"] is None
    assert state["last_attempt"]["peak_rss_bytes"] >= 30_000
    assert state["last_success"]["run_id"] == state["last_attempt"]["run_id"]
    assert not (tmp_path / "indexer.pid").exists()


def test_monitor_records_signal_as_terminal_failure(tmp_path):
    process = FakeProcess()
    supervisor = _supervisor(
        tmp_path,
        process,
        popen_factory=lambda *a, **k: process,
        rss_reader=lambda _pid: 1234,
    )
    supervisor.start(["python", "indexer"], log_path=tmp_path / "indexer.log")
    process.finish(-signal.SIGKILL)

    state = _wait_for(
        lambda: (
            snapshot
            if (snapshot := supervisor.snapshot()).get("current") is None
            else None
        )
    )
    assert state["last_attempt"]["status"] == "signaled"
    assert state["last_attempt"]["exit_code"] is None
    assert state["last_attempt"]["termination_signal"] == signal.SIGKILL
    assert state["last_success"] is None


def test_startup_reconciles_dead_active_run_to_lost_terminal_state(tmp_path):
    """A SIGKILL leaves no finally block; startup must preserve that failure."""
    active = {
        "run_id": "run-dead",
        "status": "running",
        "pid": 919191,
        "pgid": 919191,
        "source_name": None,
        "started_at": "2026-07-15T12:00:00+00:00",
        "peak_rss_bytes": 99,
    }
    (tmp_path / "index_run_state.json").write_text(
        json.dumps(
            {
                "version": 1,
                "current": active,
                "last_attempt": active,
                "last_success": None,
            }
        )
    )
    (tmp_path / "indexer.pid").write_text("919191")

    from index_run_supervisor import IndexRunSupervisor

    supervisor = IndexRunSupervisor(
        tmp_path,
        pid_alive=lambda _pid: False,
        process_matches=lambda _pid: True,
        monitor_interval=0.01,
    )
    state = supervisor.snapshot()

    assert state["current"] is None
    assert state["last_attempt"]["status"] == "lost"
    assert state["last_attempt"]["terminal_reason"] == "process_missing_on_reconcile"
    assert state["last_attempt"]["finished_at"]
    assert not (tmp_path / "indexer.pid").exists()


def test_shutdown_terminates_process_group_and_records_terminal_signal(tmp_path):
    process = FakeProcess()
    signals: list[tuple[int, int]] = []

    def killpg(pgid, sig):
        signals.append((pgid, sig))
        if sig == signal.SIGTERM:
            process.finish(-signal.SIGTERM)

    supervisor = _supervisor(
        tmp_path,
        process,
        popen_factory=lambda *a, **k: process,
        rss_reader=lambda _pid: 100,
        killpg=killpg,
    )
    supervisor.start(["python", "indexer"], log_path=tmp_path / "indexer.log")

    supervisor.shutdown(grace_seconds=0.2)
    state = supervisor.snapshot()

    assert signals == [(process.pid, signal.SIGTERM)]
    assert state["current"] is None
    assert state["last_attempt"]["status"] == "terminated"
    assert state["last_attempt"]["termination_signal"] == signal.SIGTERM


def test_shutdown_escalates_to_sigkill_when_group_ignores_term(tmp_path):
    process = FakeProcess()
    signals: list[tuple[int, int]] = []

    def killpg(pgid, sig):
        signals.append((pgid, sig))
        if sig == signal.SIGKILL:
            process.finish(-signal.SIGKILL)

    supervisor = _supervisor(
        tmp_path,
        process,
        popen_factory=lambda *a, **k: process,
        rss_reader=lambda _pid: 100,
        killpg=killpg,
    )
    supervisor.start(["python", "indexer"], log_path=tmp_path / "indexer.log")

    supervisor.shutdown(grace_seconds=0.01)

    assert signals == [
        (process.pid, signal.SIGTERM),
        (process.pid, signal.SIGKILL),
    ]
    assert supervisor.snapshot()["last_attempt"]["status"] == "terminated"


def test_launch_failure_is_durable_terminal_attempt(tmp_path):
    process = FakeProcess()

    def fail_launch(*_args, **_kwargs):
        raise OSError("cannot fork")

    supervisor = _supervisor(tmp_path, process, popen_factory=fail_launch)
    result = supervisor.start(["python", "indexer"], log_path=tmp_path / "indexer.log")
    state = supervisor.snapshot()

    assert result["status"] == "launch_failed"
    assert "cannot fork" in result["error"]
    assert state["current"] is None
    assert state["last_attempt"]["status"] == "launch_failed"
    assert state["last_attempt"]["finished_at"]
    assert state["last_success"] is None


def test_status_summary_marks_latest_terminal_failure_unresolved(tmp_path):
    process = FakeProcess()
    supervisor = _supervisor(tmp_path, process, popen_factory=lambda *a, **k: process)
    supervisor.start(["python", "indexer"], log_path=tmp_path / "indexer.log")
    process.finish(-signal.SIGKILL)

    summary = _wait_for(
        lambda: (
            status
            if (status := supervisor.status_summary()).get("latest_terminal")
            else None
        )
    )

    assert summary["latest_terminal"]["status"] == "signaled"
    assert summary["unresolved_failure"] is True
    assert summary["last_success"] is None
