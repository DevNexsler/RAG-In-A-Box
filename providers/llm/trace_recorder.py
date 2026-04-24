from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

logger = logging.getLogger(__name__)

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "unknown"


class LLMTraceRecorder:
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        enabled: bool = False,
        directory: str | Path = ".evals/llm-traces",
    ) -> None:
        self.provider = provider
        self.model = model
        self.enabled = enabled
        self.directory = Path(directory)

    def record(
        self,
        *,
        request: dict[str, Any],
        success: bool,
        latency_ms: float,
        response: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return

        ts = datetime.now(timezone.utc)
        payload = {
            "ts": ts.isoformat(),
            "provider": self.provider,
            "model": self.model,
            "request": request,
            "response": response,
            "latency_ms": latency_ms,
            "success": success,
            "error": error,
        }

        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / self._build_filename(ts)
            with path.open("a", encoding="utf-8") as fh:
                if fcntl is not None:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
                if fcntl is not None:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            logger.warning("Failed to write LLM trace: %s", exc)

    def _build_filename(self, ts: datetime) -> str:
        return f"{ts.date().isoformat()}-{_slugify(self.provider)}-{_slugify(self.model)}.jsonl"
