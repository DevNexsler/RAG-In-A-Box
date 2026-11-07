import base64
import io
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from PIL import Image

from core.resilience import is_transient
from providers.ocr import build_ocr_provider
from providers.ocr.fallback import FallbackOCRProvider
from providers.ocr.litellm_ocr import LiteLLMOCR


def _response(url: str, content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": content}}]},
        request=httpx.Request("POST", url),
    )


def test_extract_routes_to_ocr_alias(monkeypatch, tmp_path):
    image = tmp_path / "page.png"
    image.write_bytes(b"png-bytes")
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return _response(url, "invoice 123")

    monkeypatch.setattr(
        "providers.fallback.litellm_fallback.httpx.post", fake_post
    )
    provider = LiteLLMOCR(
        "http://lite/v1/", "ocr", "vision", timeout=41, api_key="unit-key"
    )

    assert provider.extract(image, page=7) == "invoice 123"
    assert provider.endpoint == "http://lite/v1"
    assert provider.extract_model == "ocr"
    assert provider.describe_model == "vision"
    assert provider.timeout == 41
    assert captured["url"] == "http://lite/v1/chat/completions"
    assert captured["json"] == {
        "model": "ocr",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Transcribe all text in this image verbatim.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,cG5nLWJ5dGVz"
                        },
                    },
                ],
            }
        ],
        "temperature": 0.0,
    }
    assert captured["timeout"] == 41
    assert captured["headers"]["Authorization"] == "Bearer unit-key"


def test_extract_uses_jpeg_mime_for_jpg(monkeypatch, tmp_path):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"image")
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return _response(url, "text")

    monkeypatch.setattr(
        "providers.fallback.litellm_fallback.httpx.post", fake_post
    )
    provider = LiteLLMOCR(
        "http://lite/v1", "ocr", "vision", api_key="unit-key"
    )

    assert provider.extract(image) == "text"
    content = captured["json"]["messages"][0]["content"]
    assert content[1]["image_url"]["url"] == (
        "data:image/jpeg;base64,aW1hZ2U="
    )


def _ocr_cloud_patches(width: int, height: int) -> int:
    """Patches `ocr-cloud` counts for an image: 32x32 px tiles, partial tiles
    included. Matched the counts in its 400s exactly (#4514)."""
    return -(-width // 32) * -(-height // 32)


def _posted_image(captured) -> tuple[str, bytes]:
    url = captured["json"]["messages"][0]["content"][1]["image_url"]["url"]
    header, encoded = url.split(",", 1)
    return header, base64.b64decode(encoded)


def test_extract_shrinks_a_page_over_the_ocr_cloud_patch_budget(
    monkeypatch, tmp_path
):
    """#4514: a phone-scanned PDF page (1755x2443 pt at 72 dpi) renders at 200
    dpi to 4875x6787 px = 32589 patches, which `ocr-cloud` refuses (limit
    30000) on every retry, so the page was never OCR'd while the local hops
    were down. The page sent must fit the alias's narrowest hop."""
    page = tmp_path / "page.png"
    Image.new("RGB", (4875, 6787), "white").save(page)
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return _response(url, "page text")

    monkeypatch.setattr(
        "providers.fallback.litellm_fallback.httpx.post", fake_post
    )
    provider = LiteLLMOCR("http://lite/v1", "ocr", "vision", api_key="k")

    assert provider.extract(page) == "page text"
    header, data = _posted_image(captured)
    assert header == "data:image/png;base64"
    sent = Image.open(io.BytesIO(data))
    assert _ocr_cloud_patches(*sent.size) <= 30000
    # Shrunk only as far as the budget needs, aspect ratio kept.
    assert _ocr_cloud_patches(*sent.size) > 29000
    assert abs(sent.width / sent.height - 4875 / 6787) < 0.01


def test_extract_sends_a_page_within_the_budget_byte_for_byte(
    monkeypatch, tmp_path
):
    """Page 1 of the same PDF (4645x6439 px, 29492 patches) was read fine;
    a page that fits keeps its full resolution and its exact bytes."""
    page = tmp_path / "page.png"
    Image.new("RGB", (4645, 6439), "white").save(page)
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return _response(url, "text")

    monkeypatch.setattr(
        "providers.fallback.litellm_fallback.httpx.post", fake_post
    )
    provider = LiteLLMOCR("http://lite/v1", "ocr", "vision", api_key="k")

    assert provider.extract(page) == "text"
    assert _posted_image(captured) == ("data:image/png;base64", page.read_bytes())


def test_describe_routes_to_vision_alias(monkeypatch, tmp_path):
    image = tmp_path / "photo.png"
    image.write_bytes(b"image")
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return _response(url, "a detailed description")

    monkeypatch.setattr(
        "providers.fallback.litellm_fallback.httpx.post", fake_post
    )
    provider = LiteLLMOCR(
        "http://lite/v1", "ocr", "vision", api_key="unit-key"
    )

    assert provider.describe(image) == "a detailed description"
    assert captured["json"]["model"] == "vision"
    prompt = captured["json"]["messages"][0]["content"][0]["text"]
    assert "Describe this image in detail" in prompt


def test_reachable_whitespace_returns_empty(monkeypatch, tmp_path):
    image = tmp_path / "blank.png"
    image.write_bytes(b"image")

    monkeypatch.setattr(
        "providers.fallback.litellm_fallback.httpx.post",
        lambda url, **kwargs: _response(url, "  \n\t "),
    )
    provider = LiteLLMOCR(
        "http://lite/v1", "ocr", "vision", api_key="unit-key"
    )

    assert provider.extract(image) == ""


def test_uses_environment_auth_without_explicit_key(monkeypatch, tmp_path):
    image = tmp_path / "page.png"
    image.write_bytes(b"image")
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return _response(url, "text")

    monkeypatch.setenv("LITELLM_API_KEY", "environment-key")
    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    monkeypatch.setattr(
        "providers.fallback.litellm_fallback.httpx.post", fake_post
    )
    provider = LiteLLMOCR("http://lite/v1", "ocr", "vision")

    assert provider.extract(image) == "text"
    assert captured["headers"]["Authorization"] == "Bearer environment-key"


def test_unauthorized_response_remains_transient(monkeypatch, tmp_path):
    image = tmp_path / "page.png"
    image.write_bytes(b"image")

    def fake_post(url, **kwargs):
        return httpx.Response(
            401,
            json={"error": "bad key"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(
        "providers.fallback.litellm_fallback.httpx.post", fake_post
    )
    provider = LiteLLMOCR(
        "http://lite/v1", "ocr", "vision", api_key="bad-key"
    )

    with pytest.raises(Exception) as caught:
        provider.extract(image)

    assert is_transient(caught.value)


def test_startup_log_names_endpoint_and_aliases_but_not_key(caplog):
    with caplog.at_level(logging.INFO, logger="providers.ocr.litellm_ocr"):
        LiteLLMOCR(
            "http://lite/v1/",
            "ocr",
            "vision",
            api_key="must-not-appear",
        )

    messages = " ".join(caplog.messages)
    assert "http://lite/v1" in messages
    assert "ocr" in messages
    assert "vision" in messages
    assert "must-not-appear" not in messages


def test_factory_builds_litellm_provider_directly(monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "environment-key")
    config = {
        "ocr": {
            "enabled": True,
            "provider": "litellm",
            "endpoint": "http://lite/v1",
            "extract_model": "ocr",
            "describe_model": "vision",
            "timeout": 41,
            "concurrency": 3,
            "api_key": "yaml-key-must-be-ignored",
        }
    }

    provider = build_ocr_provider(config)

    assert type(provider) is LiteLLMOCR
    assert not isinstance(provider, FallbackOCRProvider)
    assert provider.endpoint == "http://lite/v1"
    assert provider.extract_model == "ocr"
    assert provider.describe_model == "vision"
    assert provider.timeout == 41
    assert provider.concurrency == 3
    assert provider._extract_client.api_key == "environment-key"
    assert provider._describe_client.api_key == "environment-key"


def test_factory_enforces_shared_ocr_concurrency_limit():
    provider = build_ocr_provider(
        {
            "ocr": {
                "enabled": True,
                "provider": "litellm",
                "endpoint": "http://lite/v1",
                "extract_model": "ocr",
                "describe_model": "vision",
                "concurrency": 1,
            }
        }
    )
    extract_started = threading.Event()
    release_extract = threading.Event()
    describe_started = threading.Event()
    second_gate_entry = threading.Event()

    class RecordingGate:
        def __init__(self, gate):
            self._gate = gate
            self._entries = 0
            self._lock = threading.Lock()

        def __enter__(self):
            with self._lock:
                self._entries += 1
                if self._entries == 2:
                    second_gate_entry.set()
            self._gate.acquire()

        def __exit__(self, *_exc_info):
            self._gate.release()

    class BlockingExtractClient:
        def run(self, _file_path):
            extract_started.set()
            assert release_extract.wait(timeout=2)
            return "text"

    class RecordingDescribeClient:
        def run(self, _file_path):
            describe_started.set()
            return "description"

    provider._extract_client = BlockingExtractClient()
    provider._describe_client = RecordingDescribeClient()
    provider._concurrency_gate = RecordingGate(provider._concurrency_gate)

    with ThreadPoolExecutor(max_workers=2) as executor:
        extract_future = executor.submit(provider.extract, "page.png")
        assert extract_started.wait(timeout=1)
        describe_future = executor.submit(provider.describe, "photo.png")
        try:
            assert second_gate_entry.wait(timeout=1)
            assert not describe_started.is_set()
        finally:
            release_extract.set()

        assert extract_future.result(timeout=1) == "text"
        assert describe_future.result(timeout=1) == "description"
        assert describe_started.is_set()


@pytest.mark.parametrize("concurrency", [0, -1, True, 1.5, "2"])
def test_litellm_ocr_requires_positive_integer_concurrency(concurrency):
    with pytest.raises(ValueError, match="positive integer"):
        LiteLLMOCR(
            "http://lite/v1",
            "ocr",
            "vision",
            api_key="unit-key",
            concurrency=concurrency,
        )


@pytest.mark.parametrize("missing", ["endpoint", "extract_model", "describe_model"])
def test_factory_requires_litellm_field(missing):
    ocr_config = {
        "enabled": True,
        "provider": "litellm",
        "endpoint": "http://lite/v1",
        "extract_model": "ocr",
        "describe_model": "vision",
    }
    del ocr_config[missing]

    with pytest.raises(ValueError, match=missing):
        build_ocr_provider({"ocr": ocr_config})
