"""OCR via Gemini Vision API. Uses google-genai to send images to Gemini and extract text."""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Optional

from providers.ocr.base import OCRProvider

logger = logging.getLogger(__name__)

_MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}

# Text-focused OCR for PDF pages (preserves layout, notes visual elements briefly)
_PDF_PAGE_PROMPT = (
    "Extract ALL text from this document page, preserving the original layout "
    "as closely as possible.\n\n"
    "After the extracted text, if the page contains any non-text visual elements "
    "(charts, graphs, diagrams, tables, maps, photos, signatures, stamps, logos), "
    "add a section starting with '--- Visual Elements ---' and briefly describe each one "
    "(e.g. 'Bar chart showing quarterly revenue 2023-2024', "
    "'Site photo of a two-story residential building', "
    "'Table with 5 columns: Date, Description, Amount, Balance, Status').\n\n"
    "If there is no text and no visual elements, return an empty string."
)

# Rich description for standalone images: text + detailed visual description
_IMAGE_DESCRIBE_PROMPT = (
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


class GeminiOCR(OCRProvider):
    """Use Google Gemini's vision model to extract text and describe images.

    extract()  — PDF page OCR: text extraction + brief visual element notes.
    describe() — Standalone image: full text extraction + detailed visual description.

    Requires GEMINI_API_KEY environment variable.
    """

    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        self.model = model
        self._client = self._build_client()

    def _build_client(self):
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set — required for Gemini OCR")
        return genai.Client(api_key=api_key)

    _CALL_TIMEOUT = 240  # seconds per API call (dense tax/table pages need longer)

    def _call_gemini(self, image_bytes: bytes, mime_type: str, prompt: str) -> str:
        """Send image + prompt to Gemini with retry on rate limits.

        Each API call is wrapped in a thread with a hard timeout to prevent
        indefinite hangs (the google-genai SDK's built-in timeout is unreliable).
        """
        from google.genai import types

        contents = [
            types.Content(
                parts=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    types.Part.from_text(text=prompt),
                ]
            )
        ]

        def _do_call():
            resp = self._client.models.generate_content(
                model=self.model,
                contents=contents,
            )
            return resp.text.strip() if resp.text else ""

        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(_do_call)
                    return future.result(timeout=self._CALL_TIMEOUT)
            except FuturesTimeout:
                logger.warning(
                    "Gemini call timed out after %ds (attempt %d/%d)",
                    self._CALL_TIMEOUT, attempt + 1, max_retries + 1,
                )
                if attempt < max_retries:
                    continue
                raise RuntimeError(
                    f"Gemini OCR timed out after {max_retries + 1} attempts "
                    f"({self._CALL_TIMEOUT}s each)"
                )
            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                if is_rate_limit and attempt < max_retries:
                    wait = 10 * (2 ** attempt)  # 10s, 20s, 40s
                    logger.info(
                        "Gemini rate limited, retrying in %ds (attempt %d/%d)",
                        wait, attempt + 1, max_retries,
                    )
                    time.sleep(wait)
                    continue
                raise RuntimeError(
                    f"Gemini OCR failed after {attempt + 1} attempts: {e}"
                ) from e

    def _read_image(self, file_path: str | Path) -> tuple[bytes, str] | None:
        """Read image bytes and determine MIME type. Returns None if file missing."""
        file_path = Path(file_path)
        if not file_path.exists():
            return None
        image_bytes = file_path.read_bytes()
        mime_type = _MIME_MAP.get(file_path.suffix.lower(), "image/png")
        return image_bytes, mime_type

    def extract(self, file_path: str | Path, page: Optional[int] = None) -> str:
        """Text-focused OCR for PDF pages. Extracts text and briefly notes visual elements."""
        result = self._read_image(file_path)
        if result is None:
            return ""
        return self._call_gemini(result[0], result[1], _PDF_PAGE_PROMPT)

    def describe(self, file_path: str | Path) -> str:
        """Rich extraction for standalone images: text + detailed visual description."""
        result = self._read_image(file_path)
        if result is None:
            return ""
        return self._call_gemini(result[0], result[1], _IMAGE_DESCRIBE_PROMPT)
