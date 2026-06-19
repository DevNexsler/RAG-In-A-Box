"""OCR via Ollama vision-language models (e.g. qwen3-vl:8b)."""

import base64
import logging
import os
import threading
from pathlib import Path
from typing import Optional

import httpx

from providers.ocr.base import OCRProvider

logger = logging.getLogger(__name__)

_EXTRACT_PROMPT = (
    "Extract ALL text from this image, preserving the original layout "
    "as closely as possible.\n\n"
    "After the extracted text, if the image contains any non-text visual elements "
    "(charts, graphs, diagrams, tables, maps, photos, signatures, stamps, logos), "
    "add a section starting with '--- Visual Elements ---' and briefly describe each one.\n\n"
    "If there is no text and no visual elements, return an empty string."
)

_DESCRIBE_PROMPT = (
    "Analyze this image thoroughly and provide:\n\n"
    "1. **TEXT**: Extract ALL visible text exactly as it appears, preserving layout. "
    "If no text is visible, write 'No visible text.'\n\n"
    "2. **DESCRIPTION**: Provide a detailed description of the image contents including:\n"
    "   - What type of image this is (photo, screenshot, diagram, map, chart, "
    "document scan, handwritten note, receipt, invoice, floor plan, etc.)\n"
    "   - Key subjects, objects, people, or scenes depicted\n"
    "   - Any data, measurements, or quantities shown\n"
    "   - Spatial layout and relationships between elements\n"
    "   - Colors, labels, annotations, or markings\n"
    "   - Context clues about what this image relates to\n\n"
    "Format your response as:\n"
    "--- Text ---\n"
    "[extracted text here]\n\n"
    "--- Description ---\n"
    "[detailed description here]"
)


# Ollama processes one vision request at a time per model. With the indexer
# running 8-wide, naive concurrent calls queue server-side and expire against
# the HTTP timeout (901 describe timeouts in one run). Serialize app-side so
# waiting happens without burning the request timeout.
_VISION_GATE = threading.Semaphore(int(os.environ.get("OLLAMA_VISION_CONCURRENCY", "1")))

_MAX_IMAGE_DIM = 1024

# Output token caps (a ceiling, not a target — images that finish early stop
# early and pay nothing extra, so this does not slow or pad normal images).
# qwen3-vl emits a hidden reasoning trace even with think=False; on busier
# images that trace alone overran the old 800-token cap, leaving content empty
# and the image with only a metadata stub (~8.5% of images). A generous describe
# cap lets the answer land after even a long reasoning trace — the worst
# observed image needed ~4.5k tokens. The only images that take longer are the
# ones that would otherwise have produced no description at all.
_EXTRACT_NUM_PREDICT = 800
_DESCRIBE_NUM_PREDICT = 10240


def _downscale(image_bytes: bytes) -> bytes:
    """Resize large images before sending — vision prefill cost scales with
    pixels and attachment photos are often full-resolution camera shots."""
    try:
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(image_bytes))
        if max(img.size) <= _MAX_IMAGE_DIM:
            return image_bytes
        img.thumbnail((_MAX_IMAGE_DIM, _MAX_IMAGE_DIM))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception:
        return image_bytes


class OllamaVisionOCR(OCRProvider):
    """Use an Ollama vision model for text extraction and image description."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3-vl:8b",
        timeout: float = 300.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        logger.info("OllamaVisionOCR: %s model=%s", self.base_url, self.model)

    def _call(self, file_path: Path, prompt: str, num_predict: int) -> str:
        image_bytes = _downscale(file_path.read_bytes())
        b64 = base64.b64encode(image_bytes).decode("ascii")
        with _VISION_GATE:
            return self._request(b64, prompt, num_predict)

    def _request(self, b64: str, prompt: str, num_predict: int) -> str:

        resp = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [b64],
                    }
                ],
                "stream": False,
                # qwen3-vl defaults to thinking mode, which burned ~2 minutes
                # per image and made most describe calls hit the 120s timeout
                # (901 timeouts in one indexing run). Descriptions don't need
                # deliberation; cap the output so generation stays bounded.
                "think": False,
                "options": {"num_predict": num_predict},
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()

    def extract(self, file_path: str | Path, page: Optional[int] = None) -> str:
        file_path = Path(file_path)
        if not file_path.exists():
            return ""
        return self._call(file_path, _EXTRACT_PROMPT, _EXTRACT_NUM_PREDICT)

    def describe(self, file_path: str | Path) -> str:
        file_path = Path(file_path)
        if not file_path.exists():
            return ""
        text = self._call(file_path, _DESCRIBE_PROMPT, _DESCRIBE_NUM_PREDICT)
        if not text:
            # Empty content despite a successful call: qwen3-vl's hidden
            # reasoning trace overran the token cap before emitting an answer.
            # Surface it (instead of silently storing a metadata-only stub) so
            # the gap is visible in logs without triggering an endless retry.
            logger.warning(
                "Vision describe returned empty (reasoning overran cap) for %s",
                file_path,
            )
        return text
