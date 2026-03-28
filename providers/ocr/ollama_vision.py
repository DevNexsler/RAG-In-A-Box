"""OCR via Ollama vision-language models (e.g. qwen3-vl:8b)."""

import base64
import logging
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


class OllamaVisionOCR(OCRProvider):
    """Use an Ollama vision model for text extraction and image description."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3-vl:8b",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        logger.info("OllamaVisionOCR: %s model=%s", self.base_url, self.model)

    def _call(self, file_path: Path, prompt: str) -> str:
        image_bytes = file_path.read_bytes()
        b64 = base64.b64encode(image_bytes).decode("ascii")

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
        return self._call(file_path, _EXTRACT_PROMPT)

    def describe(self, file_path: str | Path) -> str:
        file_path = Path(file_path)
        if not file_path.exists():
            return ""
        return self._call(file_path, _DESCRIBE_PROMPT)
