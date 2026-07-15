"""OS-level index supervisor process-group and signal integration tests."""

from __future__ import annotations

import os
import signal
import sys
import time

from index_run_supervisor import IndexRunSupervisor, process_is_alive


def _wait_for(predicate, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("condition was not met before timeout")


def test_real_sigkill_records_signal_and_sampled_peak_rss(tmp_path):
    supervisor = IndexRunSupervisor(
        tmp_path,
        process_matches=lambda _pid: True,
        monitor_interval=0.01,
    )
    launch = supervisor.start(
        [
            sys.executable,
            "-c",
            "payload = bytearray(8 * 1024 * 1024); import time; time.sleep(30)",
        ],
        log_path=tmp_path / "indexer.log",
    )
    pid = launch["pid"]
    _wait_for(
        lambda: supervisor.status_summary()["current"]["peak_rss_bytes"] > 0
    )

    os.killpg(pid, signal.SIGKILL)
    terminal = _wait_for(
        lambda: (
            attempt
            if (attempt := supervisor.status_summary()["last_attempt"])["status"]
            == "signaled"
            else None
        )
    )

    assert terminal["termination_signal"] == signal.SIGKILL
    assert terminal["peak_rss_bytes"] > 0
    assert supervisor.status_summary()["unresolved_failure"] is True


def test_shutdown_terminates_real_process_group_descendants(tmp_path):
    grandchild_pid_path = tmp_path / "grandchild.pid"
    script = (
        "import pathlib, subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        f"pathlib.Path({str(grandchild_pid_path)!r}).write_text(str(child.pid)); "
        "time.sleep(30)"
    )
    supervisor = IndexRunSupervisor(
        tmp_path,
        process_matches=lambda _pid: True,
        monitor_interval=0.01,
    )
    launch = supervisor.start(
        [sys.executable, "-c", script],
        log_path=tmp_path / "indexer.log",
    )
    leader_pid = launch["pid"]
    _wait_for(grandchild_pid_path.exists)
    grandchild_pid = int(grandchild_pid_path.read_text())

    supervisor.shutdown(grace_seconds=0.5)

    assert not process_is_alive(leader_pid)
    assert not process_is_alive(grandchild_pid)
    assert supervisor.status_summary()["last_attempt"]["status"] == "terminated"
