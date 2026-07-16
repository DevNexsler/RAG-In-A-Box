"""Real LiteLLM OCR and image-description routing checks."""

import re
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from core.config import load_config
from providers.ocr import build_ocr_provider
from providers.ocr.litellm_ocr import LiteLLMOCR


_IMAGE_SIZE = (1920, 1080)
_TEST_TEXT = "RAGBOX LIVE OCR 7319"


def _load_large_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=112)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=112)
    except TypeError as exc:
        raise RuntimeError(
            "Pillow cannot provide a readable 112-pixel fallback font"
        ) from exc


@pytest.fixture(scope="module")
def litellm_live_image(tmp_path_factory: pytest.TempPathFactory) -> Path:
    image_path = (
        tmp_path_factory.mktemp("litellm-live-ocr") / "litellm-live-ocr.png"
    )
    image = Image.new("RGB", _IMAGE_SIZE, "white")
    draw = ImageDraw.Draw(image)
    font = _load_large_font()

    left, _, right, _ = draw.textbbox((0, 0), _TEST_TEXT, font=font)
    text_x = (_IMAGE_SIZE[0] - (right - left)) // 2 - left
    draw.text((text_x, 80), _TEST_TEXT, fill="black", font=font)
    draw.ellipse(
        (160, 390, 760, 990),
        fill=(220, 0, 0),
        outline="black",
        width=12,
    )
    draw.rectangle(
        (1160, 390, 1760, 990),
        fill=(0, 70, 230),
        outline="black",
        width=12,
    )
    image.save(image_path, format="PNG")
    return image_path


@pytest.fixture(scope="module")
def litellm_live_provider() -> LiteLLMOCR:
    provider = build_ocr_provider(load_config())
    assert isinstance(provider, LiteLLMOCR)
    assert provider.extract_model == "ocr"
    assert provider.describe_model == "vision"
    return provider


@pytest.mark.live
def test_litellm_extract_alias_reads_generated_image(
    litellm_live_provider: LiteLLMOCR,
    litellm_live_image: Path,
) -> None:
    extracted = litellm_live_provider.extract(litellm_live_image)
    normalized = "".join(character for character in extracted.upper() if character.isalnum())

    assert "RAGBOXLIVEOCR7319" in normalized


@pytest.mark.live
def test_litellm_vision_alias_describes_generated_image(
    litellm_live_provider: LiteLLMOCR,
    litellm_live_image: Path,
) -> None:
    description = litellm_live_provider.describe(litellm_live_image).lower()

    assert re.search(r"\bred\b", description)
    assert re.search(r"\bblue\b", description)
    assert re.search(r"\b(?:circle|round)\b", description)
    assert re.search(r"\b(?:square|box)\b", description)
