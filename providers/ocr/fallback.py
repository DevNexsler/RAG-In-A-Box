"""OCRProvider that adds a per-method LiteLLM fallback via the shared decision rule."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from core.fallback import resolve_with_fallback
from providers.ocr.base import OCRProvider

FallbackRun = Callable[[str], str]  # path -> text; raises transient on unreachable


class FallbackOCRProvider(OCRProvider):
    def __init__(
        self,
        primary: OCRProvider,
        describe_fallback: Optional[FallbackRun] = None,
        extract_fallback: Optional[FallbackRun] = None,
    ):
        self._primary = primary
        self._describe_fb = describe_fallback
        self._extract_fb = extract_fallback

    def describe(self, file_path: str | Path) -> str:
        p = str(file_path)
        return resolve_with_fallback(
            lambda: self._primary.describe(p),
            (lambda: self._describe_fb(p)) if self._describe_fb else None,
        )

    def extract(self, file_path: str | Path, page: Optional[int] = None) -> str:
        p = str(file_path)
        return resolve_with_fallback(
            lambda: self._primary.extract(p, page),
            (lambda: self._extract_fb(p)) if self._extract_fb else None,
            # An empty OCR-extract page is normally just textless (not an outage);
            # in dark mode it indexes clean rather than re-OCRing every run. An
            # unreachable extract still raises transient (handled in the primary call).
            empty_is_clean=True,
        )
