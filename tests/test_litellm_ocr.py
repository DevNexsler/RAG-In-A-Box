import logging

import httpx
import pytest

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
    assert provider._extract_client.api_key == "environment-key"
    assert provider._describe_client.api_key == "environment-key"


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
