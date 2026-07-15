"""Durable, single-flight supervision for background index runs.

The MCP request stays non-blocking, but launch and terminal outcome no longer
disappear behind a best-effort PID file.  A versioned state document records the
active run, last attempt, last success, exit signal, and sampled peak RSS.  The
launch lock is both process-local and ``flock``-backed so concurrent MCP calls
cannot pass the liveness check together.
"""

from __future__ import annotations

import copy
import fcntl
import json
import logging
import os
import signal
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator


logger = logging.getLogger(__name__)

_STATE_VERSION = 1
_ACTIVE_STATUSES = {"starting", "running", "terminating"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def process_is_alive(pid: int) -> bool:
    """Return true for a live, non-zombie Linux process."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return True
    return not any(
        line.startswith("State:") and ("\tZ" in line or "(zombie)" in line.lower())
        for line in status.splitlines()
    )


def process_peak_rss_bytes(pid: int) -> int:
    """Read Linux VmHWM/VmRSS without adding a process-inspection dependency."""
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
    except OSError:
        return 0
    values: dict[str, int] = {}
    for line in lines:
        key, sep, raw = line.partition(":")
        if not sep or key not in {"VmHWM", "VmRSS"}:
            continue
        parts = raw.split()
        if not parts:
            continue
        try:
            values[key] = int(parts[0]) * 1024
        except ValueError:
            continue
    return max(values.values(), default=0)


class IndexRunSupervisor:
    """Own one index run for one index root."""

    def __init__(
        self,
        index_root: str | Path,
        *,
        process_matches: Callable[[int], bool] | None = None,
        pid_alive: Callable[[int], bool] = process_is_alive,
        popen_factory: Callable[..., Any] | None = None,
        rss_reader: Callable[[int], int] = process_peak_rss_bytes,
        killpg: Callable[[int, int], None] = os.killpg,
        monitor_interval: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.index_root = Path(index_root)
        self.state_path = self.index_root / "index_run_state.json"
        self.lock_path = self.index_root / "index_run.lock"
        self.pid_path = self.index_root / "indexer.pid"
        self._process_matches = process_matches or (lambda _pid: True)
        self._pid_alive = pid_alive
        # Resolve at construction time so callers/tests can replace Popen
        # without fighting a function default bound at module import.
        self._popen = popen_factory or subprocess.Popen
        self._rss_reader = rss_reader
        self._killpg = killpg
        self._monitor_interval = max(0.001, float(monitor_interval))
        self._clock = clock
        self._sleep = sleep
        self._thread_lock = threading.RLock()
        self._monitor_lock = threading.Lock()
        self._monitors: dict[str, threading.Thread] = {}
        self.index_root.mkdir(parents=True, exist_ok=True)
        self.reconcile()

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "version": _STATE_VERSION,
            "current": None,
            "last_attempt": None,
            "last_success": None,
        }

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.index_root.mkdir(parents=True, exist_ok=True)
        with self._thread_lock:
            with self.lock_path.open("a+") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _read_state_locked(self) -> dict[str, Any]:
        try:
            state = json.loads(self.state_path.read_text())
        except FileNotFoundError:
            return self._empty_state()
        except (OSError, ValueError, TypeError) as exc:
            logger.error("Cannot read index run state %s: %s", self.state_path, exc)
            return self._empty_state()
        if not isinstance(state, dict) or state.get("version") != _STATE_VERSION:
            logger.error("Unsupported index run state at %s", self.state_path)
            return self._empty_state()
        normalized = self._empty_state()
        normalized.update(state)
        return normalized

    def _write_state_locked(self, state: dict[str, Any]) -> None:
        state = copy.deepcopy(state)
        state["version"] = _STATE_VERSION
        tmp_path = self.state_path.with_name(
            f".{self.state_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        with tmp_path.open("w") as handle:
            json.dump(state, handle, sort_keys=True, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, self.state_path)
        try:
            dir_fd = os.open(self.index_root, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass

    def _write_pid_locked(self, pid: int) -> None:
        tmp_path = self.pid_path.with_name(f".{self.pid_path.name}.{os.getpid()}.tmp")
        tmp_path.write_text(str(pid))
        os.replace(tmp_path, self.pid_path)

    def _clear_pid_locked(self, expected_pid: int | None = None) -> None:
        if expected_pid is not None:
            try:
                existing = int(self.pid_path.read_text().strip())
            except (OSError, ValueError):
                existing = None
            if existing not in {None, expected_pid}:
                return
        self.pid_path.unlink(missing_ok=True)

    def _active_is_live(self, current: dict[str, Any]) -> bool:
        try:
            pid = int(current.get("pid") or 0)
        except (TypeError, ValueError):
            return False
        return pid > 0 and self._pid_alive(pid) and self._process_matches(pid)

    def _monitor_is_alive(self, run_id: object) -> bool:
        if not run_id:
            return False
        with self._monitor_lock:
            monitor = self._monitors.get(str(run_id))
            return bool(monitor and monitor.is_alive())

    def _terminal_lost(self, current: dict[str, Any], reason: str) -> dict[str, Any]:
        terminal = copy.deepcopy(current)
        terminal.update(
            {
                "status": "lost",
                "finished_at": _utc_now(),
                "exit_code": None,
                "termination_signal": None,
                "terminal_reason": reason,
            }
        )
        return terminal

    def _reconcile_locked(self, state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        current = state.get("current")
        if isinstance(current, dict) and current.get("status") in _ACTIVE_STATUSES:
            if self._active_is_live(current):
                self._write_pid_locked(int(current["pid"]))
                return state, current
            # A child-backed monitor owns the authoritative return code.  Give
            # it the short race between process exit and its next poll instead
            # of degrading a clean/signalled exit to an ambiguous lost state.
            if self._monitor_is_alive(current.get("run_id")):
                return state, current
            terminal = self._terminal_lost(current, "process_missing_on_reconcile")
            state["current"] = None
            state["last_attempt"] = terminal
            self._clear_pid_locked(current.get("pid"))
            self._write_state_locked(state)
            return state, None

        # Upgrade the legacy PID-only contract without losing an in-progress run.
        try:
            legacy_pid = int(self.pid_path.read_text().strip())
        except (OSError, ValueError):
            legacy_pid = 0
        if legacy_pid and self._pid_alive(legacy_pid) and self._process_matches(legacy_pid):
            try:
                started_at = datetime.fromtimestamp(
                    self.pid_path.stat().st_mtime, tz=timezone.utc
                ).isoformat()
            except OSError:
                started_at = _utc_now()
            adopted = {
                "run_id": f"legacy-{legacy_pid}",
                "status": "running",
                "pid": legacy_pid,
                "pgid": legacy_pid,
                "source_name": None,
                "started_at": started_at,
                "peak_rss_bytes": self._rss_reader(legacy_pid),
                "adopted": True,
            }
            state["current"] = adopted
            state["last_attempt"] = adopted
            self._write_state_locked(state)
            return state, adopted
        if legacy_pid:
            self._clear_pid_locked(legacy_pid)
        return state, None

    def reconcile(self) -> dict[str, Any]:
        with self._locked():
            state, active = self._reconcile_locked(self._read_state_locked())
            snapshot = copy.deepcopy(state)
        if active:
            self._ensure_monitor(active, process=None)
        return snapshot

    def snapshot(self) -> dict[str, Any]:
        return self.reconcile()

    def status_summary(self) -> dict[str, Any]:
        """Expose active, latest terminal, and success freshness."""
        state = self.snapshot()
        attempt = state.get("last_attempt")
        success = state.get("last_success")
        latest_terminal = None
        if isinstance(attempt, dict) and attempt.get("status") not in _ACTIVE_STATUSES:
            latest_terminal = attempt
        unresolved_failure = bool(
            latest_terminal
            and latest_terminal.get("status") in {"launch_failed", "lost", "failed", "signaled"}
        )
        return {
            "current": copy.deepcopy(state.get("current")),
            "last_attempt": copy.deepcopy(attempt),
            "last_success": copy.deepcopy(success),
            "latest_terminal": copy.deepcopy(latest_terminal),
            "unresolved_failure": unresolved_failure,
        }

    def start(
        self,
        command: list[str],
        *,
        log_path: str | Path,
        source_name: str | None = None,
    ) -> dict[str, Any]:
        """Atomically reject overlap or launch one detached process group."""
        process = None
        attempt: dict[str, Any]
        with self._locked():
            state, active = self._reconcile_locked(self._read_state_locked())
            if active:
                return {
                    "status": "already_running",
                    "pid": int(active["pid"]),
                    "run_id": active["run_id"],
                    "source_name": active.get("source_name"),
                }

            attempt = {
                "run_id": uuid.uuid4().hex,
                "status": "starting",
                "pid": None,
                "pgid": None,
                "source_name": source_name,
                "started_at": _utc_now(),
                "peak_rss_bytes": 0,
            }
            state["current"] = attempt
            state["last_attempt"] = attempt
            self._write_state_locked(state)

            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            previous = path.with_suffix(path.suffix + ".prev")
            if path.exists():
                try:
                    os.replace(path, previous)
                except OSError:
                    pass

            log_handle = None
            try:
                log_handle = path.open("a", buffering=1)
                process = self._popen(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                pid = int(process.pid)
                attempt = {
                    **attempt,
                    "status": "running",
                    "pid": pid,
                    "pgid": pid,
                }
                state["current"] = attempt
                state["last_attempt"] = attempt
                self._write_pid_locked(pid)
                self._write_state_locked(state)
            except OSError as exc:
                terminal = {
                    **attempt,
                    "status": "launch_failed",
                    "finished_at": _utc_now(),
                    "exit_code": None,
                    "termination_signal": None,
                    "terminal_reason": str(exc),
                }
                state["current"] = None
                state["last_attempt"] = terminal
                self._clear_pid_locked()
                self._write_state_locked(state)
                return {
                    "status": "launch_failed",
                    "run_id": attempt["run_id"],
                    "source_name": source_name,
                    "error": str(exc),
                }
            finally:
                if log_handle is not None:
                    log_handle.close()

        self._ensure_monitor(attempt, process=process)
        return {
            "status": "started",
            "pid": int(attempt["pid"]),
            "run_id": attempt["run_id"],
            "source_name": source_name,
        }

    def _ensure_monitor(self, attempt: dict[str, Any], process: Any | None) -> None:
        run_id = str(attempt["run_id"])
        with self._monitor_lock:
            existing = self._monitors.get(run_id)
            if existing and existing.is_alive():
                return
            thread = threading.Thread(
                target=self._monitor,
                args=(run_id, int(attempt["pid"]), process),
                name=f"index-run-monitor-{run_id[:8]}",
                daemon=True,
            )
            self._monitors[run_id] = thread
            thread.start()

    def _record_peak(self, run_id: str, rss_bytes: int) -> None:
        if rss_bytes <= 0:
            return
        with self._locked():
            state = self._read_state_locked()
            current = state.get("current")
            if not isinstance(current, dict) or current.get("run_id") != run_id:
                return
            if rss_bytes <= int(current.get("peak_rss_bytes") or 0):
                return
            current = {**current, "peak_rss_bytes": int(rss_bytes)}
            state["current"] = current
            state["last_attempt"] = current
            self._write_state_locked(state)

    def _finish(self, run_id: str, returncode: int) -> None:
        with self._locked():
            state = self._read_state_locked()
            current = state.get("current")
            if not isinstance(current, dict) or current.get("run_id") != run_id:
                return
            terminating = current.get("status") == "terminating"
            if terminating:
                status = "terminated"
            elif returncode == 0:
                status = "succeeded"
            elif returncode < 0:
                status = "signaled"
            else:
                status = "failed"
            terminal = {
                **current,
                "status": status,
                "finished_at": _utc_now(),
                "exit_code": returncode if returncode >= 0 else None,
                "termination_signal": -returncode if returncode < 0 else None,
                "terminal_reason": (
                    "shutdown_requested"
                    if terminating
                    else ("clean_exit" if returncode == 0 else "process_exit")
                ),
            }
            state["current"] = None
            state["last_attempt"] = terminal
            if status == "succeeded":
                state["last_success"] = terminal
            self._clear_pid_locked(current.get("pid"))
            self._write_state_locked(state)

    def _finish_lost(self, run_id: str, reason: str) -> None:
        with self._locked():
            state = self._read_state_locked()
            current = state.get("current")
            if not isinstance(current, dict) or current.get("run_id") != run_id:
                return
            terminal = self._terminal_lost(current, reason)
            state["current"] = None
            state["last_attempt"] = terminal
            self._clear_pid_locked(current.get("pid"))
            self._write_state_locked(state)

    def _monitor(self, run_id: str, pid: int, process: Any | None) -> None:
        try:
            while True:
                self._record_peak(run_id, self._rss_reader(pid))
                if process is not None:
                    returncode = process.poll()
                    if returncode is not None:
                        self._finish(run_id, int(returncode))
                        return
                elif not self._pid_alive(pid) or not self._process_matches(pid):
                    self._finish_lost(run_id, "process_disappeared_after_adoption")
                    return
                self._sleep(self._monitor_interval)
        except Exception as exc:  # monitoring must not take down the MCP server
            logger.exception("Index run monitor failed for %s: %s", run_id, exc)
        finally:
            with self._monitor_lock:
                self._monitors.pop(run_id, None)

    def shutdown(self, grace_seconds: float = 10.0) -> None:
        """Terminate the active index process group before the server exits."""
        with self._locked():
            state, active = self._reconcile_locked(self._read_state_locked())
            if not active:
                return
            current = {
                **active,
                "status": "terminating",
                "shutdown_requested_at": _utc_now(),
            }
            state["current"] = current
            state["last_attempt"] = current
            self._write_state_locked(state)
            run_id = str(current["run_id"])
            pid = int(current["pid"])
            pgid = int(current.get("pgid") or pid)

        if not self._pid_alive(pid) or not self._process_matches(pid):
            self._finish_lost(run_id, "process_missing_during_shutdown")
            return

        sent_signal = signal.SIGTERM
        try:
            self._killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            self._finish_lost(run_id, "process_missing_during_shutdown")
            return

        deadline = self._clock() + max(0.0, grace_seconds)
        while self._pid_alive(pid) and self._clock() < deadline:
            self._sleep(min(0.05, max(0.0, deadline - self._clock())))

        if self._pid_alive(pid):
            sent_signal = signal.SIGKILL
            try:
                self._killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        settle_deadline = self._clock() + 1.0
        while self._pid_alive(pid) and self._clock() < settle_deadline:
            self._sleep(0.01)
        # The monitor normally records the real return code. If it lost the race
        # with shutdown, preserve the signal outcome directly.
        self._finish(run_id, -int(sent_signal))
