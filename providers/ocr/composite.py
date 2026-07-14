"""Composite OCR provider — delegates extract() and describe() to separate providers."""

import logging
from pathlib import Path
from typing import Optional

from core.resilience import is_transient
from providers.ocr.base import OCRProvider

logger = logging.getLogger(__name__)


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
        try:
            return self._describe.describe(file_path)
        except Exception as exc:
            if self._extract is self._describe or not is_transient(exc):
                raise
            logger.warning(
                "Dedicated OCR describe backend failed transiently for %s; "
                "falling back to extract backend describe: %s",
                file_path,
                exc,
            )
            return self._extract.describe(file_path)
