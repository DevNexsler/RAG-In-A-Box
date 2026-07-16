"""First-class OCR and image description through LiteLLM model aliases."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from providers.fallback.litellm_fallback import LiteLLMFallback, image_encoder
from providers.ocr.base import OCRProvider

logger = logging.getLogger(__name__)

_EXTRACT_PROMPT = "Transcribe all text in this image verbatim."
_DESCRIBE_PROMPT = "Describe this image in detail for document search."


class LiteLLMOCR(OCRProvider):
    def __init__(
        self,
        endpoint: str,
        extract_model: str,
        describe_model: str,
        timeout: float = 300.0,
        *,
        api_key: str | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.extract_model = extract_model
        self.describe_model = describe_model
        self.timeout = timeout
        self._extract_client = LiteLLMFallback(
            self.endpoint,
            extract_model,
            _EXTRACT_PROMPT,
            image_encoder,
            api_key=api_key,
            timeout=timeout,
        )
        self._describe_client = LiteLLMFallback(
            self.endpoint,
            describe_model,
            _DESCRIBE_PROMPT,
            image_encoder,
            api_key=api_key,
            timeout=timeout,
        )
        logger.info(
            "LiteLLMOCR: %s extract_model=%s describe_model=%s",
            self.endpoint,
            extract_model,
            describe_model,
        )

    def extract(self, file_path: str | Path, page: Optional[int] = None) -> str:
        return self._extract_client.run(file_path)

    def describe(self, file_path: str | Path) -> str:
        return self._describe_client.run(file_path)
