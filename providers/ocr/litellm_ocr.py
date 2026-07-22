"""First-class OCR and image description through LiteLLM model aliases."""
from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Optional

from providers.fallback.litellm_fallback import LiteLLMFallback, image_encoder
from providers.ocr.base import OCRProvider

logger = logging.getLogger(__name__)

_EXTRACT_PROMPT = "Transcribe all text in this image verbatim."
_DESCRIBE_PROMPT = "Describe this image in detail for document search."

# Vision-model prefill cost scales with image resolution (image tokens): a
# 4000px image is ~4-5x slower to describe than ~1024px, with no describe-quality
# loss (measured against qwen3-vl:8b). We downscale ONLY for the describe call;
# the extract (OCR text) path keeps full resolution so small text stays legible.
_DESCRIBE_MAX_EDGE = 1024


def _image_encoder(path: Path, prompt: str) -> list:
    content = image_encoder(path, prompt)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        encoded = content[1]["image_url"]["url"].split(",", 1)[1]
        content[1]["image_url"]["url"] = f"data:image/jpeg;base64,{encoded}"
    return content


def _downscaled_describe_encoder(path: Path, prompt: str) -> list:
    """Encode an image for the vision DESCRIBE call, downscaled to
    ~_DESCRIBE_MAX_EDGE on its longest edge. Falls back to the full-resolution
    encoder if the file is not a decodable image."""
    path = Path(path)
    try:
        from PIL import Image

        with Image.open(path) as img:
            img = img.convert("RGB")
            width, height = img.size
            scale = max(width, height) / _DESCRIBE_MAX_EDGE
            if scale > 1.0:
                img = img.resize(
                    (max(1, round(width / scale)), max(1, round(height / scale))),
                    Image.LANCZOS,
                )
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
        b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        return [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]
    except Exception as exc:  # noqa: BLE001 — any decode/resize issue -> full-res
        logger.debug("describe downscale failed for %s (%s); sending full-res", path, exc)
        return _image_encoder(path, prompt)


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
            _image_encoder,
            api_key=api_key,
            timeout=timeout,
        )
        self._describe_client = LiteLLMFallback(
            self.endpoint,
            describe_model,
            _DESCRIBE_PROMPT,
            _downscaled_describe_encoder,
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
