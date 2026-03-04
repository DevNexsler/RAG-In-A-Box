"""Generic llama-server process manager with shared file-based heartbeat.

Used by both the embedding provider and the reranker to auto-start/stop
llama-server processes on demand with idle timeout.

Each named instance (e.g. "reranker", "embedder") gets its own singleton,
its own port, and its own model file. Thread-safe and multi-process safe.

Heartbeat coordination:
  Any local process (our code or external services) can keep the server alive
  by touching {heartbeat_dir}/{name}-{port}.heartbeat. The watchdog checks
  the heartbeat file's mtime to determine idle time, so it respects activity
  from ALL clients — not just our process.

  A file lock ({heartbeat_dir}/{name}-{port}.lock) prevents two processes
  from racing to start the server simultaneously.
"""

from __future__ import annotations

import fcntl
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_HEARTBEAT_DIR = "/Users/Shared/Shared-Services/llama-server"


class LlamaServerManager:
    """Manages a llama-server process lifecycle with shared heartbeat.

    - Auto-starts on first request if not already running.
    - Auto-shuts down after `idle_timeout` seconds of no activity from ANY client.
    - Thread-safe: multiple threads can call ensure_running() safely.
    - Multi-process safe: file lock prevents startup races; heartbeat file
      tracks activity from all local clients.
    - Keyed singletons: one per name (e.g. "reranker", "embedder").
    """

    _instances: dict[str, LlamaServerManager] = {}
    _class_lock = threading.Lock()

    def __init__(
        self,
        name: str,
        model_path: str,
        port: int,
        server_flags: list[str] | None = None,
        idle_timeout: float = 300.0,
        startup_timeout: float = 120.0,
        heartbeat_dir: str = DEFAULT_HEARTBEAT_DIR,
    ):
        self.name = name
        self.model_path = model_path
        self.port = port
        self.server_flags = server_flags or []
        self.idle_timeout = idle_timeout
        self.startup_timeout = startup_timeout
        self.heartbeat_dir = Path(heartbeat_dir)
        self._process: subprocess.Popen | None = None
        self._watchdog: threading.Thread | None = None
        self._stopping = False
        self._lock = threading.Lock()
        self._log_fh = None

        self.heartbeat_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _heartbeat_path(self) -> Path:
        return self.heartbeat_dir / f"{self.name}-{self.port}.heartbeat"

    @property
    def _lockfile_path(self) -> Path:
        return self.heartbeat_dir / f"{self.name}-{self.port}.lock"

    @classmethod
    def get_instance(
        cls,
        name: str,
        model_path: str = "",
        port: int = 8787,
        server_flags: list[str] | None = None,
        idle_timeout: float = 300.0,
        heartbeat_dir: str = DEFAULT_HEARTBEAT_DIR,
    ) -> LlamaServerManager:
        """Get or create a named singleton instance."""
        with cls._class_lock:
            if name not in cls._instances:
                cls._instances[name] = cls(
                    name=name,
                    model_path=model_path,
                    port=port,
                    server_flags=server_flags,
                    idle_timeout=idle_timeout,
                    heartbeat_dir=heartbeat_dir,
                )
            return cls._instances[name]

    def _is_healthy(self) -> bool:
        """Check if the server is responding and the model is fully loaded.

        llama-server returns {"status": "ok"} when ready, but
        {"status": "loading model"} during startup — both with HTTP 200.
        """
        try:
            import httpx
            resp = httpx.get(
                f"http://localhost:{self.port}/health",
                timeout=2.0,
            )
            if resp.status_code != 200:
                return False
            body = resp.json()
            return body.get("status") == "ok"
        except Exception:
            return False

    def _start_server(self) -> None:
        """Spawn the llama-server process."""
        llama_server = shutil.which("llama-server")
        if not llama_server:
            raise RuntimeError(
                "llama-server not found on PATH. Install with: brew install llama.cpp"
            )

        if not os.path.isfile(self.model_path):
            raise RuntimeError(f"Model not found: {self.model_path}")

        cmd = [
            llama_server,
            "-m", self.model_path,
            "--port", str(self.port),
            *self.server_flags,
        ]
        log_path = self.heartbeat_dir / f"{self.name}-{self.port}.log"
        self._log_fh = open(log_path, "a")
        logger.info("Starting llama-server [%s]: %s (log: %s)", self.name, " ".join(cmd), log_path)
        self._process = subprocess.Popen(
            cmd,
            stdout=self._log_fh,
            stderr=self._log_fh,
        )

    def _wait_for_healthy(self) -> bool:
        """Poll the health endpoint until the server is ready."""
        deadline = time.monotonic() + self.startup_timeout
        interval = 0.5
        while time.monotonic() < deadline:
            if self._process and self._process.poll() is not None:
                logger.error(
                    "llama-server [%s] exited during startup (code=%d)",
                    self.name, self._process.returncode,
                )
                return False
            if self._is_healthy():
                return True
            time.sleep(interval)
            interval = min(interval * 1.5, 3.0)
        logger.error(
            "llama-server [%s] failed to become healthy within %ds",
            self.name, self.startup_timeout,
        )
        return False

    def _get_idle_seconds(self) -> float:
        """Read idle time from the shared heartbeat file's mtime."""
        try:
            mtime = self._heartbeat_path.stat().st_mtime
            return time.time() - mtime
        except FileNotFoundError:
            return float("inf")

    def _watchdog_loop(self) -> None:
        """Background thread: shuts down the server after idle_timeout.

        Reads the shared heartbeat file's mtime so activity from ANY local
        client (not just our process) keeps the server alive.
        """
        while not self._stopping:
            time.sleep(10)
            if self._stopping:
                break
            if self._process is None or self._process.poll() is not None:
                break
            idle = self._get_idle_seconds()
            if idle >= self.idle_timeout:
                logger.info(
                    "[%s] idle for %.0fs — shutting down llama-server (pid=%d)",
                    self.name, idle, self._process.pid,
                )
                self.stop()
                break

    def ensure_running(self) -> bool:
        """Ensure the server is running and healthy. Starts it if needed.

        Uses a file lock so multiple processes don't race to start the server.
        Returns True if the server is ready, False if it could not be started.
        """
        self.touch()

        if self._is_healthy():
            return True

        # Acquire cross-process file lock to prevent startup races
        lock_fd = None
        try:
            lock_fd = open(self._lockfile_path, "w")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

            # Double-check after acquiring lock (another process may have started it)
            if self._is_healthy():
                return True

            with self._lock:
                if self._process and self._process.poll() is None:
                    try:
                        self._process.terminate()
                        self._process.wait(timeout=5)
                    except Exception:
                        self._process.kill()

                self._start_server()
                if not self._wait_for_healthy():
                    return False

                logger.info(
                    "llama-server [%s] ready (pid=%d, port=%d, idle_timeout=%ds)",
                    self.name, self._process.pid, self.port, self.idle_timeout,
                )

                self._stopping = False
                self._watchdog = threading.Thread(
                    target=self._watchdog_loop, daemon=True,
                    name=f"{self.name}-watchdog",
                )
                self._watchdog.start()
                return True
        finally:
            if lock_fd:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()

    def touch(self) -> None:
        """Reset the idle timer by updating the shared heartbeat file.

        Any local process can call this (or simply `touch` the file from
        the command line) to keep the server alive.
        """
        try:
            self._heartbeat_path.touch()
        except OSError as e:
            logger.debug("Could not touch heartbeat file %s: %s", self._heartbeat_path, e)

    def stop(self) -> None:
        """Gracefully stop the server."""
        self._stopping = True
        if self._process and self._process.poll() is None:
            logger.info(
                "Stopping llama-server [%s] (pid=%d)", self.name, self._process.pid,
            )
            try:
                self._process.terminate()
                self._process.wait(timeout=10)
            except Exception:
                self._process.kill()
        self._process = None
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None
