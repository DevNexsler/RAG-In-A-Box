"""Tests for DeepSeek-OCR2 OCR provider (local mlx-vlm service).

Unit tests mock HTTP calls (always run).
Live tests hit the real service at the URL resolved from, in order:
OCR_BASE_URL env var, ocr.base_url in config_test.yaml / config.yaml, or
http://localhost:8790. They are skipped if unreachable.

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

from extractors import begin_degradation_capture, collect_degradations, extract_image
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
        from providers.ocr.fallback import FallbackOCRProvider
        assert isinstance(provider, FallbackOCRProvider)
        assert isinstance(provider._primary, DeepSeekOCR2Local)
        assert provider._primary.base_url == "http://localhost:8790"
        assert provider._primary.timeout == 60.0

    def test_build_ocr_provider_routes_images_to_ollama_describe(self, tmp_path):
        """Split OCR config keeps PDF OCR on DeepSeek and routes images to Ollama VL."""
        from providers.ocr import build_ocr_provider
        from providers.ocr.composite import CompositeOCRProvider

        from PIL import Image as _Img
        img = tmp_path / "photo.png"
        _Img.new("RGB", (8, 8), "white").save(str(img))  # real PNG for the VL downscale
        config = {
            "ocr": {
                "enabled": True,
                "provider": "deepseek_ocr2",
                "base_url": "http://deepseek:8790",
                "timeout": 60.0,
                "describe": {
                    "provider": "ollama_vision",
                    "base_url": "http://ollama:11434",
                    "model": "qwen3-vl:8b",
                    "timeout": 90.0,
                },
            }
        }

        provider = build_ocr_provider(config)

        from providers.ocr.fallback import FallbackOCRProvider
        assert isinstance(provider, FallbackOCRProvider)
        assert isinstance(provider._primary, CompositeOCRProvider)
        with patch("providers.ocr.deepseek_ocr2_local.httpx.post") as deepseek_post:
            deepseek_resp = MagicMock()
            deepseek_resp.json.return_value = {"text": "pdf page text"}
            deepseek_resp.raise_for_status = MagicMock()
            deepseek_post.return_value = deepseek_resp
            assert provider._primary.extract(img) == "pdf page text"

        # ollama_vision streams via httpx.Client().stream() (not httpx.post) —
        # capture the stream call to assert routing + payload.
        import json as _json
        captured = {}

        class _OllamaResp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def raise_for_status(self): pass
            def iter_lines(self):
                return iter([_json.dumps(
                    {"message": {"content": "image description"}, "done": True})])

        class _OllamaClient:
            def __init__(self, *a, **k): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def stream(self, method, url, json=None):
                captured["url"], captured["payload"] = url, json
                return _OllamaResp()

        with patch("providers.ocr.ollama_vision.httpx.Client", _OllamaClient):
            assert provider._primary.describe(img) == "image description"

        assert deepseek_post.call_args.args[0] == "http://deepseek:8790/extract"
        assert captured["url"] == "http://ollama:11434/api/chat"
        assert captured["payload"]["model"] == "qwen3-vl:8b"

    def test_split_provider_describe_falls_back_to_extract_backend_on_transient_error(self, tmp_path):
        """If the dedicated image-describe backend is transiently down, reuse the
        extract backend's describe path instead of degrading the image to metadata only."""
        from providers.ocr import build_ocr_provider
        from providers.ocr.composite import CompositeOCRProvider

        from PIL import Image as _Img
        img = tmp_path / "photo.png"
        _Img.new("RGB", (8, 8), "white").save(str(img))
        config = {
            "ocr": {
                "enabled": True,
                "provider": "deepseek_ocr2",
                "base_url": "http://deepseek:8790",
                "timeout": 60.0,
                "describe": {
                    "provider": "ollama_vision",
                    "base_url": "http://ollama:11434",
                    "model": "qwen3-vl:8b",
                    "timeout": 90.0,
                },
            }
        }

        provider = build_ocr_provider(config)

        # build_ocr_provider now ALWAYS wraps in FallbackOCRProvider; the composite
        # (with #62's local sibling-failover) is the wrapper's primary. This asserts
        # #62's failover works correctly INSIDE the fallback wrapper: an outage on the
        # dedicated describe backend is recovered locally via the extract backend before
        # the (dark / unconfigured) cloud fallback would ever see an empty.
        from providers.ocr.fallback import FallbackOCRProvider

        assert isinstance(provider, FallbackOCRProvider)
        composite = provider._primary
        assert isinstance(composite, CompositeOCRProvider)
        with patch.object(
            composite._describe, "describe", side_effect=httpx.ConnectError("vision down")
        ) as describe_call:
            with patch.object(
                composite._extract, "describe", return_value="deepseek backup description"
            ) as backup_call:
                assert provider.describe(img) == "deepseek backup description"

        # FallbackOCRProvider stringifies the path before delegating to the primary.
        describe_call.assert_called_once_with(str(img))
        backup_call.assert_called_once_with(str(img))

    def test_split_provider_cooldown_preserves_ocr_output_without_degradation(self, tmp_path):
        """A failed dedicated describe call and its cooldown both use extract describe."""
        from PIL import Image as _Img

        from providers.ocr import build_ocr_provider
        from providers.ocr.composite import CompositeOCRProvider
        from providers.ocr.fallback import FallbackOCRProvider
        from providers.ocr.ollama_vision import OllamaVisionOCR

        img = tmp_path / "photo.png"
        _Img.new("RGB", (8, 8), "white").save(str(img))
        provider = build_ocr_provider(
            {
                "ocr": {
                    "enabled": True,
                    "provider": "deepseek_ocr2",
                    "base_url": "http://deepseek:8790",
                    "describe": {
                        "provider": "ollama_vision",
                        "base_url": "http://ollama:11434",
                    },
                }
            }
        )

        assert isinstance(provider, FallbackOCRProvider)
        composite = provider._primary
        assert isinstance(composite, CompositeOCRProvider)
        assert isinstance(composite._describe, OllamaVisionOCR)

        begin_degradation_capture()
        with patch.object(
            composite._describe,
            "_call",
            side_effect=httpx.ConnectError("vision down"),
        ) as vision_call:
            with patch.object(
                composite._extract,
                "describe",
                return_value="deepseek backup description",
            ) as backup_call:
                first = extract_image(img, provider)
                second = extract_image(img, provider)

        assert "deepseek backup description" in first.full_text
        assert "deepseek backup description" in second.full_text
        assert collect_degradations() == []
        vision_call.assert_called_once()
        assert backup_call.call_count == 2


class TestOllamaVisionBudget:
    """qwen3-vl's hidden reasoning trace starved the old 800-token cap, leaving
    ~8.5% of images with empty descriptions. describe() now gets a larger cap
    and surfaces the rare still-empty result instead of storing a silent stub."""

    def _img(self, tmp_path):
        from PIL import Image
        img = tmp_path / "photo.jpg"
        Image.new("RGB", (8, 8), "white").save(str(img))  # real image for downscale
        return img

    @staticmethod
    def _fake_client(contents):
        """Build a fake httpx.Client whose .stream() yields contents[i] on the
        i-th call (last value repeats). _request streams via httpx.Client, not
        httpx.post, so we patch the client and record each call's payload/count."""
        import json as _json
        state = {"payloads": [], "calls": 0}
        seq = list(contents)

        class _Resp:
            def __init__(self, content): self._c = content
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def raise_for_status(self): pass
            def iter_lines(self):
                return iter([_json.dumps({"message": {"content": self._c}, "done": True})])

        class _Client:
            def __init__(self, *a, **k): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def stream(self, method, url, json=None):
                i = state["calls"]; state["calls"] += 1
                state["payloads"].append(json)
                return _Resp(seq[i] if i < len(seq) else seq[-1])

        return _Client, state

    def test_describe_uses_larger_num_predict(self, tmp_path):
        from providers.ocr.ollama_vision import OllamaVisionOCR, _DESCRIBE_NUM_PREDICT

        Client, state = self._fake_client(["a description"])
        with patch("providers.ocr.ollama_vision.httpx.Client", Client):
            OllamaVisionOCR(base_url="http://ollama:11434").describe(self._img(tmp_path))
        opts = state["payloads"][0]["options"]
        assert opts["num_predict"] == _DESCRIBE_NUM_PREDICT
        assert _DESCRIBE_NUM_PREDICT >= 2048
        # greedy decoding — deterministic, faithful descriptions
        assert opts["temperature"] == 0
        # keep-alive is intentionally NOT sent from the client — the vision host (Mac Mini)
        # owns the pin-resident policy via OLLAMA_KEEP_ALIVE=-1; a per-request value here
        # would override that server default, so we omit it.
        assert "keep_alive" not in state["payloads"][0]

    def test_extract_keeps_default_num_predict(self, tmp_path):
        from providers.ocr.ollama_vision import OllamaVisionOCR, _EXTRACT_NUM_PREDICT

        Client, state = self._fake_client(["some text"])
        with patch("providers.ocr.ollama_vision.httpx.Client", Client):
            OllamaVisionOCR(base_url="http://ollama:11434").extract(self._img(tmp_path))
        assert state["payloads"][0]["options"]["num_predict"] == _EXTRACT_NUM_PREDICT

    def test_describe_empty_returns_empty_and_warns(self, tmp_path, caplog):
        import logging
        from providers.ocr.ollama_vision import OllamaVisionOCR, _DESCRIBE_EMPTY_RETRIES

        Client, state = self._fake_client([""])  # always empty
        with patch("providers.ocr.ollama_vision.httpx.Client", Client):
            with patch("providers.ocr.ollama_vision.time.sleep"):
                with caplog.at_level(logging.WARNING, logger="providers.ocr.ollama_vision"):
                    result = OllamaVisionOCR(base_url="http://ollama:11434").describe(self._img(tmp_path))
        assert result == ""
        assert any("empty" in r.message.lower() for r in caplog.records)
        # a persistently-empty describe is retried before giving up
        assert state["calls"] == 1 + _DESCRIBE_EMPTY_RETRIES

    def test_describe_retries_empty_then_succeeds(self, tmp_path):
        """An empty describe is usually transient (timeout/contention under
        load), so describe() retries and returns the recovered description."""
        from providers.ocr.ollama_vision import OllamaVisionOCR

        Client, state = self._fake_client(["", "a real description"])
        with patch("providers.ocr.ollama_vision.httpx.Client", Client):
            with patch("providers.ocr.ollama_vision.time.sleep"):
                result = OllamaVisionOCR(base_url="http://ollama:11434").describe(self._img(tmp_path))
        assert result == "a real description"
        assert state["calls"] == 2

    def test_describe_connect_error_raises_transient_and_enters_cooldown(self, tmp_path):
        from core.resilience import TransientError, is_transient
        from providers.ocr.ollama_vision import OllamaVisionOCR

        state = {"calls": 0}

        class _Client:
            def __init__(self, *a, **k): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def stream(self, method, url, json=None):
                state["calls"] += 1
                raise httpx.ConnectError("Connection refused")

        with patch("providers.ocr.ollama_vision.httpx.Client", _Client):
            provider = OllamaVisionOCR(base_url="http://ollama:11434")
            with pytest.raises(Exception) as first:
                provider.describe(self._img(tmp_path))
            assert is_transient(first.value)          # outage is transient
            # second call is short-circuited by cooldown -> raises WITHOUT hitting client
            with pytest.raises(TransientError):
                provider.describe(self._img(tmp_path))
        assert state["calls"] == 1                     # cooldown suppressed the 2nd real call


# -----------------------------------------------------------------------
# Live integration tests (require DeepSeek-OCR2 reachable at its configured
# base_url). Resolution order:
#   1. OCR_BASE_URL env var
#   2. ocr.base_url from config_test.yaml (or config.yaml as fallback)
#   3. http://localhost:8790
# -----------------------------------------------------------------------

def _resolve_ocr_base_url() -> str:
    env_url = os.environ.get("OCR_BASE_URL")
    if env_url:
        return env_url.rstrip("/")
    repo_root = Path(__file__).parent.parent
    for cfg_name in ("config_test.yaml", "config.yaml"):
        cfg_path = repo_root / cfg_name
        if not cfg_path.exists():
            continue
        try:
            import yaml
            with open(cfg_path) as f:
                raw = yaml.safe_load(f) or {}
            url = (raw.get("ocr") or {}).get("base_url")
            if url:
                return url.rstrip("/")
        except Exception:
            continue
    return "http://localhost:8790"


_OCR_BASE_URL = _resolve_ocr_base_url()


def _deepseek_ocr2_running() -> bool:
    try:
        resp = httpx.get(f"{_OCR_BASE_URL}/health", timeout=3.0)
        return resp.status_code < 500
    except Exception:
        return False


def _minimal_png() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xff\xff?\x00\x05\xfe\x02\xfeA\xe2%\xb5"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def _deepseek_ocr2_model_ready() -> bool:
    if not _deepseek_ocr2_running():
        return False
    for endpoint in ("/extract", "/describe"):
        try:
            resp = httpx.post(
                f"{_OCR_BASE_URL}{endpoint}",
                files={"file": ("test.png", _minimal_png(), "image/png")},
                timeout=10.0,
            )
        except Exception:
            return False
        if resp.status_code >= 500:
            return False
    return True


_TEST_IMAGE = Path(__file__).parent.parent / "test_vault" / "meeting_notes@00004@.png"


@pytest.mark.live
@pytest.mark.skipif(
    not _deepseek_ocr2_model_ready(),
    reason=f"DeepSeek-OCR2 model endpoints not ready at {_OCR_BASE_URL}",
)
class TestDeepSeekOCR2Live:
    """Live integration tests against real DeepSeek-OCR2 service."""

    @pytest.fixture(scope="class")
    def provider(self):
        return DeepSeekOCR2Local(base_url=_OCR_BASE_URL, timeout=120.0)

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


def test_ollama_request_prepends_no_reasoning_system_message(monkeypatch):
    """_request must send a system message suppressing qwen3-vl's reasoning trace.

    qwen3-vl:8b ignores the API's think=False and emits a long <think> trace that
    eats the whole num_predict budget, returning empty content on dense images.
    A plain-language system instruction is what actually forces a direct answer;
    this guards that it stays in the request payload (regression for the
    empty-describe / done_reason=length bug)."""
    import json as _json
    from providers.ocr import ollama_vision as ov

    captured = {}

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def raise_for_status(self): pass
        def iter_lines(self):
            return iter([_json.dumps({"message": {"content": "OCR text"}, "done": True})])

    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def stream(self, method, url, json=None):
            captured["payload"] = json
            return FakeResp()

    monkeypatch.setattr("providers.ocr.ollama_vision.httpx.Client", FakeClient)
    prov = ov.OllamaVisionOCR(base_url="http://x:11434", model="qwen3-vl:8b")
    out = prov._request("B64DATA", ov._DESCRIBE_PROMPT, 5120)

    assert out == "OCR text"                      # still parses the stream
    msgs = captured["payload"]["messages"]
    assert msgs[0] == {"role": "system", "content": ov._NO_REASONING_SYSTEM}
    assert msgs[1]["role"] == "user" and msgs[1]["images"] == ["B64DATA"]
