"""OCR via local DeepSeek-OCR2 service (mlx-vlm on Apple Silicon)."""

import logging
from pathlib import Path
from typing import Optional

import httpx

from core.resilience import call_with_retry
from providers.ocr.base import OCRProvider

logger = logging.getLogger(__name__)


class DeepSeekOCR2Local(OCRProvider):
    """Send images to the local DeepSeek-OCR2 HTTP service for text extraction."""

    def __init__(self, base_url: str = "http://localhost:8790", timeout: float = 120.0,
                 attempts: int = 2, backoff: tuple[float, ...] = (3.0, 8.0)):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Retry transient blips (504 gateway-timeout when the model is cold-loading or
        # the host is momentarily busy, connection resets) via the shared resilience
        # layer. Kept low (1 retry) because OCR is slow — a genuinely-too-slow page
        # would 504 again, so we degrade it to the ledger rather than burn minutes.
        self.attempts = attempts
        self.backoff = backoff
        logger.info("DeepSeekOCR2Local: %s", self.base_url)

    def _send(self, endpoint: str, file_path: Path) -> str:
        image_bytes = file_path.read_bytes()
        mime = "image/png" if file_path.suffix.lower() == ".png" else "image/jpeg"

        def _do() -> str:
            resp = httpx.post(
                f"{self.base_url}{endpoint}",
                files={"file": (file_path.name, image_bytes, mime)},
                timeout=self.timeout,
            )
            resp.raise_for_status()  # 504/5xx -> HTTPStatusError -> retried by the layer
            return resp.json().get("text", "")

        return call_with_retry(
            _do, attempts=self.attempts, backoff=self.backoff,
            label=f"deepseek-ocr {endpoint}",
        )

    def extract(self, file_path: str | Path, page: Optional[int] = None) -> str:
        file_path = Path(file_path)
        if not file_path.exists():
            return ""
        return self._send("/extract", file_path)

    def describe(self, file_path: str | Path) -> str:
        file_path = Path(file_path)
        if not file_path.exists():
            return ""
        return self._send("/describe", file_path)
