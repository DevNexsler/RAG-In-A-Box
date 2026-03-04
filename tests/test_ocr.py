"""Tests for DeepSeek-OCR2 OCR provider (local mlx-vlm service).

Unit tests mock HTTP calls (always run).
Live tests require the service at localhost:8790 (skipped otherwise).

Run with:  pytest tests/test_ocr.py -v
"""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import httpx

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from providers.ocr.deepseek_ocr2_local import DeepSeekOCR2Local


# -----------------------------------------------------------------------
# Unit tests (mocked HTTP, always run)
# -----------------------------------------------------------------------

class TestDeepSeekOCR2Unit:
    """Unit tests with mocked httpx.post — no service needed."""

    def test_extract_returns_text(self, tmp_path):
        """extract() sends file to /extract and returns the 'text' field."""
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "Extracted OCR text"}
        mock_resp.raise_for_status = MagicMock()

        with patch("providers.ocr.deepseek_ocr2_local.httpx.post", return_value=mock_resp) as mock_post:
            provider = DeepSeekOCR2Local(base_url="http://localhost:8790")
            result = provider.extract(img)

        assert result == "Extracted OCR text"
        mock_post.assert_called_once()
        call_url = mock_post.call_args[0][0]
        assert "/extract" in call_url

    def test_describe_returns_text(self, tmp_path):
        """describe() sends file to /describe and returns the 'text' field."""
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "A meeting whiteboard with notes"}
        mock_resp.raise_for_status = MagicMock()

        with patch("providers.ocr.deepseek_ocr2_local.httpx.post", return_value=mock_resp) as mock_post:
            provider = DeepSeekOCR2Local(base_url="http://localhost:8790")
            result = provider.describe(img)

        assert result == "A meeting whiteboard with notes"
        call_url = mock_post.call_args[0][0]
        assert "/describe" in call_url

    def test_extract_missing_file_returns_empty(self):
        """extract() returns '' for a non-existent file path (no HTTP call)."""
        provider = DeepSeekOCR2Local()
        result = provider.extract("/nonexistent/path/image.png")
        assert result == ""

    def test_describe_missing_file_returns_empty(self):
        """describe() returns '' for a non-existent file path (no HTTP call)."""
        provider = DeepSeekOCR2Local()
        result = provider.describe("/nonexistent/path/image.png")
        assert result == ""

    def test_extract_http_error_raises(self, tmp_path):
        """extract() propagates HTTP errors (no silent swallowing)."""
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        with patch(
            "providers.ocr.deepseek_ocr2_local.httpx.post",
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            provider = DeepSeekOCR2Local()
            with pytest.raises(httpx.ConnectError):
                provider.extract(img)

    def test_describe_http_error_raises(self, tmp_path):
        """describe() propagates timeout errors (no silent swallowing)."""
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        with patch(
            "providers.ocr.deepseek_ocr2_local.httpx.post",
            side_effect=httpx.TimeoutException("timed out"),
        ):
            provider = DeepSeekOCR2Local()
            with pytest.raises(httpx.TimeoutException):
                provider.describe(img)

    def test_correct_mime_type_png(self, tmp_path):
        """extract() uses image/png MIME type for .png files."""
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "ok"}
        mock_resp.raise_for_status = MagicMock()

        with patch("providers.ocr.deepseek_ocr2_local.httpx.post", return_value=mock_resp) as mock_post:
            provider = DeepSeekOCR2Local()
            provider.extract(img)

        files_arg = mock_post.call_args[1]["files"]
        mime = files_arg["file"][2]
        assert mime == "image/png"

    def test_correct_mime_type_jpeg(self, tmp_path):
        """extract() uses image/jpeg MIME type for .jpg files."""
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "ok"}
        mock_resp.raise_for_status = MagicMock()

        with patch("providers.ocr.deepseek_ocr2_local.httpx.post", return_value=mock_resp) as mock_post:
            provider = DeepSeekOCR2Local()
            provider.extract(img)

        files_arg = mock_post.call_args[1]["files"]
        mime = files_arg["file"][2]
        assert mime == "image/jpeg"

    def test_custom_base_url_and_timeout(self):
        """Constructor accepts custom base_url and timeout."""
        provider = DeepSeekOCR2Local(base_url="http://custom:9999/", timeout=30.0)
        assert provider.base_url == "http://custom:9999"
        assert provider.timeout == 30.0

    def test_build_ocr_provider_deepseek(self):
        """build_ocr_provider creates DeepSeekOCR2Local for provider='deepseek_ocr2'."""
        from providers.ocr import build_ocr_provider

        config = {
            "ocr": {
                "enabled": True,
                "provider": "deepseek_ocr2",
                "base_url": "http://localhost:8790",
                "timeout": 60.0,
            }
        }
        provider = build_ocr_provider(config)
        assert isinstance(provider, DeepSeekOCR2Local)
        assert provider.base_url == "http://localhost:8790"
        assert provider.timeout == 60.0


# -----------------------------------------------------------------------
# Live integration tests (require DeepSeek-OCR2 at localhost:8790)
# -----------------------------------------------------------------------

def _deepseek_ocr2_running() -> bool:
    try:
        resp = httpx.get("http://localhost:8790/health", timeout=3.0)
        return resp.status_code < 500
    except Exception:
        return False


_TEST_IMAGE = Path(__file__).parent.parent / "test_vault" / "meeting_notes.png"


@pytest.mark.live
@pytest.mark.skipif(
    not _deepseek_ocr2_running(),
    reason="DeepSeek-OCR2 not running at localhost:8790",
)
class TestDeepSeekOCR2Live:
    """Live integration tests against real DeepSeek-OCR2 service."""

    @pytest.fixture(scope="class")
    def provider(self):
        return DeepSeekOCR2Local(base_url="http://localhost:8790", timeout=120.0)

    def test_extract_returns_nonempty_text(self, provider):
        """Real OCR should return non-empty text for a real image."""
        assert _TEST_IMAGE.exists(), f"Test image not found: {_TEST_IMAGE}"
        text = provider.extract(_TEST_IMAGE)
        assert len(text) > 10, f"Expected substantial OCR text, got {len(text)} chars"

    def test_describe_returns_nonempty_text(self, provider):
        """Real describe should return non-empty text for a real image."""
        assert _TEST_IMAGE.exists(), f"Test image not found: {_TEST_IMAGE}"
        text = provider.describe(_TEST_IMAGE)
        assert len(text) > 10, f"Expected substantial describe text, got {len(text)} chars"

    def test_extract_pdf_page_via_ocr(self, provider, tmp_path):
        """Extract text from a PDF page rendered to PNG (simulates pipeline)."""
        import fitz

        pdf_path = tmp_path / "test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 100), "Invoice #12345: Total $500.00", fontsize=14)
        doc.save(str(pdf_path))
        doc.close()

        doc = fitz.open(str(pdf_path))
        pix = doc[0].get_pixmap(dpi=200)
        png_path = tmp_path / "page0.png"
        pix.save(str(png_path))
        doc.close()

        text = provider.extract(png_path)
        assert "12345" in text or "500" in text, f"Expected invoice content, got: {text[:200]}"
