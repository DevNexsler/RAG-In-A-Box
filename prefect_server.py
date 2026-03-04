"""Auto-start/stop Prefect server so flow runs persist in SQLite for dashboard debugging."""

import atexit
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

PREFECT_HOST = "127.0.0.1"
PREFECT_PORT = 4200
HEALTH_URL = f"http://{PREFECT_HOST}:{PREFECT_PORT}/api/health"
API_URL = f"http://{PREFECT_HOST}:{PREFECT_PORT}/api"
STARTUP_TIMEOUT = 30  # seconds


class PrefectServer:
    """Context manager that ensures a Prefect server is running.

    - If a server is already healthy, reuses it (won't stop on exit).
    - Otherwise, starts one as a subprocess and stops it on exit.
    - Sets PREFECT_API_URL so flows log to the persistent server.
    """

    def __init__(self) -> None:
        self._process: Optional[subprocess.Popen] = None
        self._we_started = False
        self._original_api_url: Optional[str] = None
        self._log_fh = None

    # -- context manager --

    def __enter__(self) -> "PrefectServer":
        self._original_api_url = os.environ.get("PREFECT_API_URL")

        if self._is_healthy():
            logger.info("Prefect server already running at %s — reusing", API_URL)
        else:
            self._start()

        os.environ["PREFECT_API_URL"] = API_URL
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._stop()
        # Restore original env
        if self._original_api_url is None:
            os.environ.pop("PREFECT_API_URL", None)
        else:
            os.environ["PREFECT_API_URL"] = self._original_api_url
        return None

    # -- internal --

    def _is_healthy(self) -> bool:
        try:
            r = httpx.get(HEALTH_URL, timeout=2)
            return r.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def _start(self) -> None:
        logger.info("Starting Prefect server on %s:%s …", PREFECT_HOST, PREFECT_PORT)
        log_path = os.path.join(tempfile.gettempdir(), "prefect_server.log")
        self._log_fh = open(log_path, "a")
        logger.info("Prefect server log: %s", log_path)
        self._process = subprocess.Popen(
            [
                sys.executable, "-m", "prefect", "server", "start",
                "--host", PREFECT_HOST,
                "--port", str(PREFECT_PORT),
                "--no-services",
            ],
            stdout=self._log_fh,
            stderr=self._log_fh,
        )
        self._we_started = True
        atexit.register(self._stop)

        # Handle SIGTERM so cleanup runs if process is killed
        prev_handler = signal.getsignal(signal.SIGTERM)

        def _on_sigterm(signum, frame):
            self._stop()
            if callable(prev_handler) and prev_handler not in (signal.SIG_DFL, signal.SIG_IGN):
                prev_handler(signum, frame)
            else:
                sys.exit(128 + signum)

        signal.signal(signal.SIGTERM, _on_sigterm)

        # Poll until healthy
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"Prefect server exited with code {self._process.returncode}"
                )
            if self._is_healthy():
                logger.info("Prefect server ready at %s", API_URL)
                return
            time.sleep(0.5)

        self._stop()
        raise TimeoutError(
            f"Prefect server did not become healthy within {STARTUP_TIMEOUT}s"
        )

    def _stop(self) -> None:
        if not self._we_started or self._process is None:
            return
        if self._process.poll() is None:
            logger.info("Stopping Prefect server (pid %s)", self._process.pid)
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)
        self._process = None
        self._we_started = False
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None
