"""OCR provider interface. Implementations: deepseek_ocr2_local, gemini_ocr, none."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class OCRProvider(ABC):
    """Interface for vision-based extraction.

    extract()  — text-focused OCR for PDF pages (preserves layout).
    describe() — rich extraction for standalone images: text + visual description.
    """

    @abstractmethod
    def extract(self, file_path: str | Path, page: Optional[int] = None) -> str:
        """Extract text from an image or a single PDF page. Returns empty string if nothing to return."""
        ...

    def describe(self, file_path: str | Path) -> str:
        """Extract text AND describe visual content of a standalone image.

        Default implementation falls back to extract().
        Providers with vision models should override with a richer prompt.
        """
        return self.extract(file_path)
