"""First-class OCR and image description through LiteLLM model aliases."""
from __future__ import annotations

import base64
import io
import logging
import math
import threading
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
# Re-measured 2026-10-16 on `vision` = omni, 20 real receipt/document photos,
# dollar amounts read: 1024px 18/20, 2048px 17/20, full 16/20, at 809 vs 1941
# median image tokens. The 2026-10-16 sideways-receipt misread was LiteLLM's
# orientation hook turning the photo the wrong way, not this cap.
_DESCRIBE_MAX_EDGE = 1024

# The `ocr` alias falls back to `ocr-cloud`, which refuses an image needing more
# than 30000 patches of 32x32 px, identically on every retry. A 200-dpi render
# of a phone-scanned PDF page (stored at 72 dpi, so 25x33 in) needs more:
# #4514's pages needed 32085-44772 and were never OCR'd while the local hops
# were down, while the 29492-patch page of the same PDF was read. The alias can
# only accept what its narrowest hop does, so extract keeps full resolution up
# to that budget and shrinks past it.
_EXTRACT_PATCH_EDGE = 32
_EXTRACT_MAX_PATCHES = 30000


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
        from PIL import Image, ImageOps

        with Image.open(path) as img:
            # Re-encoding below drops the EXIF orientation tag, so apply it to
            # the pixels first; otherwise a phone photo reaches the model sideways.
            img = ImageOps.exif_transpose(img).convert("RGB")
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


def _extract_encoder(path: Path, prompt: str) -> list:
    """Encode an image for the OCR EXTRACT call: unchanged when it fits the
    `ocr` route's patch budget, otherwise shrunk just enough to fit it. Sent
    unchanged too if the file is not a decodable image."""
    from PIL import Image, ImageOps

    path = Path(path)
    try:
        with Image.open(path) as img:
            columns = math.ceil(img.width / _EXTRACT_PATCH_EDGE)
            rows = math.ceil(img.height / _EXTRACT_PATCH_EDGE)
            if columns * rows <= _EXTRACT_MAX_PATCHES:
                img = None
            else:
                # Re-encoding drops the EXIF orientation tag (see describe).
                img = ImageOps.exif_transpose(img).convert("RGB")
    except (OSError, Image.DecompressionBombError) as exc:
        logger.debug("extract patch fit skipped for %s (%s); sending as-is", path, exc)
        img = None
    if img is None:
        return _image_encoder(path, prompt)

    # The largest patch grid of the image's shape inside the budget, then the
    # scale that fits the image into that grid.
    width, height = img.size
    grid_scale = math.sqrt(
        _EXTRACT_MAX_PATCHES * _EXTRACT_PATCH_EDGE**2 / (width * height)
    )
    scale = min(
        math.floor(width * grid_scale / _EXTRACT_PATCH_EDGE) * _EXTRACT_PATCH_EDGE / width,
        math.floor(height * grid_scale / _EXTRACT_PATCH_EDGE) * _EXTRACT_PATCH_EDGE / height,
    )
    img = img.resize(
        (max(1, math.floor(width * scale)), max(1, math.floor(height * scale))),
        Image.LANCZOS,
    )
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]


class LiteLLMOCR(OCRProvider):
    def __init__(
        self,
        endpoint: str,
        extract_model: str,
        describe_model: str,
        timeout: float = 300.0,
        *,
        api_key: str | None = None,
        concurrency: int = 1,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.extract_model = extract_model
        self.describe_model = describe_model
        self.timeout = timeout
        if (
            isinstance(concurrency, bool)
            or not isinstance(concurrency, int)
            or concurrency < 1
        ):
            raise ValueError("ocr concurrency must be a positive integer")
        self.concurrency = concurrency
        self._concurrency_gate = threading.BoundedSemaphore(self.concurrency)
        self._extract_client = LiteLLMFallback(
            self.endpoint,
            extract_model,
            _EXTRACT_PROMPT,
            _extract_encoder,
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
            "LiteLLMOCR: %s extract_model=%s describe_model=%s concurrency=%d",
            self.endpoint,
            extract_model,
            describe_model,
            self.concurrency,
        )

    def extract(self, file_path: str | Path, page: Optional[int] = None) -> str:
        with self._concurrency_gate:
            return self._extract_client.run(file_path)

    def describe(self, file_path: str | Path) -> str:
        with self._concurrency_gate:
            return self._describe_client.run(file_path)
