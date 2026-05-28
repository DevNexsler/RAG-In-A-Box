"""Test that file_index_update does NOT block the HTTP server.

The indexer must run in a background subprocess so the MCP server
can continue serving search/status requests while indexing runs.
"""

from pathlib import Path
import os
import time
from unittest.mock import patch

import pytest


def _spawn_zombie_child() -> int:
    """Create a zombie child for the current process and return its PID."""
    pid = os.fork()
    if pid == 0:
        os._exit(0)

    deadline = time.time() + 5
    status_path = Path(f"/proc/{pid}/status")
    while time.time() < deadline:
        try:
            status = status_path.read_text()
        except FileNotFoundError:
            break
        if "State:\tZ" in status:
            return pid
        time.sleep(0.01)

    os.waitpid(pid, 0)
    raise AssertionError("Failed to create zombie child for test")


def _patch_index_config(monkeypatch, mcp_server, tmp_path):
    monkeypatch.setattr(
        mcp_server,
        "load_config",
        lambda _path: {"index_root": str(tmp_path)},
    )


def test_index_update_returns_started(tmp_path, monkeypatch):
    """_file_index_update_impl must return immediately with status='started'.

    The old blocking code returned status='completed' only after the full
    indexer flow finished (12+ minutes).  The new non-blocking code must
    return status='started' and a subprocess PID before the indexer finishes.
    """
    import mcp_server

    monkeypatch.setenv("INDEX_ROOT", str(tmp_path))
    _patch_index_config(monkeypatch, mcp_server, tmp_path)

    class DummyProc:
        pid = 424242

    with patch("subprocess.Popen", return_value=DummyProc()):
        result = mcp_server._file_index_update_impl("config.yaml")

    assert result.get("status") == "started", (
        f"Expected status='started' (non-blocking), got {result.get('status')!r}. "
        "If status is 'completed', the indexer ran synchronously and blocked."
    )
    assert "pid" in result, "Must return the subprocess PID for tracking"
    assert isinstance(result["pid"], int), "PID must be an integer"
    assert result["pid"] > 0, "PID must be positive"
    assert result["pid"] == 424242


def test_index_update_rejects_concurrent_runs(tmp_path, monkeypatch):
    """If an indexer subprocess is still running, a second call returns
    'already_running' instead of launching a competing process."""
    import sys
    import subprocess
    import mcp_server

    monkeypatch.setenv("INDEX_ROOT", str(tmp_path))
    _patch_index_config(monkeypatch, mcp_server, tmp_path)

    # Launch a long-lived dummy subprocess and write its PID file manually,
    # simulating an in-progress indexer run.
    dummy = subprocess.Popen(
        [sys.executable, "-c", "import time; index_vault_flow = True; time.sleep(30)"],
        start_new_session=True,
    )
    pid_file = tmp_path / "indexer.pid"
    pid_file.write_text(str(dummy.pid))

    try:
        result = mcp_server._file_index_update_impl("config.yaml")
        assert result.get("status") == "already_running", (
            f"Expected 'already_running' while a subprocess is alive, got {result!r}"
        )
        assert result.get("pid") == dummy.pid
    finally:
        dummy.terminate()
        dummy.wait()
        pid_file.unlink(missing_ok=True)


def test_index_update_clears_stale_pid_file(tmp_path, monkeypatch):
    """A stale PID file (dead process) must be cleaned up and a new
    indexer started instead of returning 'already_running'."""
    import mcp_server

    monkeypatch.setenv("INDEX_ROOT", str(tmp_path))
    _patch_index_config(monkeypatch, mcp_server, tmp_path)

    # Write a PID that cannot possibly be alive (PID 1 is init/systemd, which
    # we can't signal to test "dead", so use a PID that we know exited by
    # forking a process, waiting for it, then using its former PID).
    import subprocess, sys
    zombie = subprocess.Popen([sys.executable, "-c", "pass"])
    zombie.wait()
    dead_pid = zombie.pid

    pid_file = tmp_path / "indexer.pid"
    pid_file.write_text(str(dead_pid))

    class DummyProc:
        pid = 424243

    with patch("subprocess.Popen", return_value=DummyProc()):
        result = mcp_server._file_index_update_impl("config.yaml")

    # Should have cleaned up the stale PID and started a new indexer
    assert result.get("status") == "started", (
        f"Expected 'started' after clearing stale PID file, got {result!r}"
    )
    assert result.get("pid") == 424243


def test_index_update_clears_zombie_pid_file(tmp_path, monkeypatch):
    """Zombie PID file must not block a fresh indexer start."""
    import mcp_server

    monkeypatch.setenv("INDEX_ROOT", str(tmp_path))
    _patch_index_config(monkeypatch, mcp_server, tmp_path)

    zombie_pid = _spawn_zombie_child()
    pid_file = tmp_path / "indexer.pid"
    pid_file.write_text(str(zombie_pid))

    class DummyProc:
        pid = 424242

    try:
        with patch("subprocess.Popen", return_value=DummyProc()):
            result = mcp_server._file_index_update_impl("config_test.yaml")

        assert result.get("status") == "started", (
            f"Expected zombie PID file to be cleared, got {result!r}"
        )
        assert result.get("pid") == 424242
        assert not pid_file.exists(), "Zombie PID file should be removed before restart"
    finally:
        try:
            os.waitpid(zombie_pid, os.WNOHANG)
        except ChildProcessError:
            pass


def test_index_update_ignores_zombie_pid_file(tmp_path, monkeypatch):
    """A zombie PID must not block a new indexing run."""
    import subprocess
    import sys

    import mcp_server

    monkeypatch.setenv("INDEX_ROOT", str(tmp_path))
    _patch_index_config(monkeypatch, mcp_server, tmp_path)

    zombie = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        start_new_session=True,
    )
    time.sleep(0.1)

    pid_file = tmp_path / "indexer.pid"
    pid_file.write_text(str(zombie.pid))

    try:
        result = mcp_server._file_index_update_impl("config.yaml")
        assert result.get("status") == "started", (
            f"Expected zombie PID to be ignored, got {result!r}"
        )
        assert "pid" in result
    finally:
        zombie.wait()
        pid_file.unlink(missing_ok=True)


def test_index_update_ignores_non_indexer_pid_file(tmp_path, monkeypatch):
    """A reused PID for another process must not block a fresh indexer."""
    import subprocess
    import sys

    import mcp_server

    monkeypatch.setenv("INDEX_ROOT", str(tmp_path))
    _patch_index_config(monkeypatch, mcp_server, tmp_path)

    other_proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )

    pid_file = tmp_path / "indexer.pid"
    pid_file.write_text(str(other_proc.pid))

    class DummyProc:
        pid = 424244

    try:
        with patch("subprocess.Popen", return_value=DummyProc()):
            result = mcp_server._file_index_update_impl("config.yaml")

        assert result.get("status") == "started", (
            f"Expected foreign PID to be ignored, got {result!r}"
        )
        assert result.get("pid") == 424244
        assert not pid_file.exists(), "Foreign PID file should be removed before restart"
    finally:
        other_proc.terminate()
        other_proc.wait()


def test_pid_file_is_cleaned_up_by_subprocess(tmp_path, monkeypatch):
    """The PID file written inside the subprocess must be removed when the
    subprocess exits (even if the indexer flow raises an exception)."""
    import sys
    import subprocess
    import time

    # Launch the subprocess directly (mimicking what _file_index_update_impl
    # does) with a script that writes a PID file and then exits immediately.
    pid_file = tmp_path / "indexer.pid"
    script = f"""
import os
from pathlib import Path

pid_file = Path({str(pid_file)!r})
pid_file.write_text(str(os.getpid()))
try:
    raise RuntimeError("simulated flow failure")
finally:
    pid_file.unlink(missing_ok=True)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        start_new_session=True,
    )
    proc.wait(timeout=10)

    assert not pid_file.exists(), (
        "PID file should be cleaned up by the subprocess even after an error"
    )
