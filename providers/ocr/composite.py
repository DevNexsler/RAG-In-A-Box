"""Composite OCR provider — delegates extract() and describe() to separate providers."""

from pathlib import Path
from typing import Optional

from providers.ocr.base import OCRProvider


class CompositeOCRProvider(OCRProvider):
    """Routes extract() and describe() to potentially different providers.

    Used when config specifies separate providers for PDF text extraction
    vs image description (e.g. DeepSeek OCR2 for PDFs, Ollama VL for images).
    """

    def __init__(self, extract_provider: OCRProvider, describe_provider: OCRProvider):
        self._extract = extract_provider
        self._describe = describe_provider

    def extract(self, file_path: str | Path, page: Optional[int] = None) -> str:
        return self._extract.extract(file_path, page)

    def describe(self, file_path: str | Path) -> str:
        return self._describe.describe(file_path)
