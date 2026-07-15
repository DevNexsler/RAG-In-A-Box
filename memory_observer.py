"""Opt-in, low-overhead process and Arrow memory checkpoints."""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable


def current_rss_bytes() -> int | None:
    """Read current Linux RSS without loading a process-inspection package."""
    try:
        lines = Path(f"/proc/{os.getpid()}/status").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        if not line.startswith("VmRSS:"):
            continue
        try:
            return int(line.split()[1]) * 1024
        except (IndexError, ValueError):
            return None
    return None


def arrow_allocated_bytes() -> int | None:
    """Return Arrow allocator bytes when pyarrow exposes the metric."""
    try:
        import pyarrow

        return int(pyarrow.total_allocated_bytes())
    except (ImportError, AttributeError, TypeError, ValueError):
        return None


class MemoryObserver:
    """Emit structured memory checkpoints only when explicitly enabled."""

    def __init__(
        self,
        *,
        enabled: bool,
        logger: logging.Logger,
        rss_reader: Callable[[], int | None] = current_rss_bytes,
        arrow_reader: Callable[[], int | None] = arrow_allocated_bytes,
    ) -> None:
        self.enabled = bool(enabled)
        self._logger = logger
        self._rss_reader = rss_reader
        self._arrow_reader = arrow_reader
        self._lock = threading.Lock()
        self._last_rss: int | None = None
        self._last_arrow: int | None = None

    @classmethod
    def from_config(cls, config: dict, logger: logging.Logger) -> "MemoryObserver":
        settings = config.get("memory_observability", {})
        return cls(enabled=bool(settings.get("enabled", False)), logger=logger)

    def sample(self, event: str, **fields: Any) -> None:
        if not self.enabled:
            return
        try:
            with self._lock:
                rss = self._rss_reader()
                arrow = self._arrow_reader()
                payload: dict[str, Any] = {"event": event, **fields}
                if rss is not None:
                    payload["rss_bytes"] = rss
                    if self._last_rss is not None:
                        payload["rss_delta_bytes"] = rss - self._last_rss
                    self._last_rss = rss
                if arrow is not None:
                    payload["arrow_allocated_bytes"] = arrow
                    if self._last_arrow is not None:
                        payload["arrow_delta_bytes"] = arrow - self._last_arrow
                    self._last_arrow = arrow
                self._logger.info("index-memory %s", json.dumps(payload, sort_keys=True))
        except Exception:
            # Observability must never interrupt indexing.
            self._logger.debug("index memory checkpoint failed", exc_info=True)
