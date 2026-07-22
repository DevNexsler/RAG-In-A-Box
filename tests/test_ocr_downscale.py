"""Vision-describe image downscaling (speeds up qwen3-vl prefill ~4-5x)."""
import base64
import io
from pathlib import Path

from PIL import Image

from providers.ocr.litellm_ocr import _DESCRIBE_MAX_EDGE, _downscaled_describe_encoder


def _decoded(content) -> Image.Image:
    url = content[1]["image_url"]["url"]
    data = base64.b64decode(url.split(",", 1)[1])
    return Image.open(io.BytesIO(data))


def test_downscaled_describe_encoder_caps_longest_edge(tmp_path: Path):
    p = tmp_path / "big.png"
    Image.new("RGB", (4000, 2000), "blue").save(p)
    img = _decoded(_downscaled_describe_encoder(p, "describe"))
    assert max(img.size) <= _DESCRIBE_MAX_EDGE
    assert img.size == (1024, 512)  # aspect ratio preserved


def test_downscaled_describe_encoder_does_not_upscale_small_images(tmp_path: Path):
    p = tmp_path / "small.png"
    Image.new("RGB", (400, 300), "red").save(p)
    img = _decoded(_downscaled_describe_encoder(p, "describe"))
    assert img.size == (400, 300)


def test_downscaled_describe_encoder_handles_rgba(tmp_path: Path):
    p = tmp_path / "alpha.png"
    Image.new("RGBA", (2048, 2048), (0, 128, 255, 128)).save(p)
    img = _decoded(_downscaled_describe_encoder(p, "describe"))
    assert img.size == (1024, 1024)
    assert img.mode == "RGB"  # flattened for JPEG


def test_downscaled_describe_encoder_non_image_falls_back(tmp_path: Path):
    p = tmp_path / "notimage.jpg"
    p.write_text("this is not an image")
    content = _downscaled_describe_encoder(p, "describe")  # must not raise
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:")
