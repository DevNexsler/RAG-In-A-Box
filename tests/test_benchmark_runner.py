from unittest.mock import patch

import httpx
import pytest

from providers.llm.openrouter_llm import OpenRouterGenerator


def test_generate_with_metadata_returns_content_usage_and_latency(tmp_path):
    response_json = {
        "id": "chatcmpl-123",
        "choices": [{"message": {"content": '{"summary":"ok"}'}}],
        "usage": {"total_tokens": 42},
    }
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(200, json=response_json, request=request)

    with patch("providers.llm.openrouter_llm.httpx.post", return_value=response):
        generator = OpenRouterGenerator(
            model="openai/gpt-4.1-mini",
            api_key="secret-key",
            trace_capture={"enabled": True, "directory": str(tmp_path)},
        )
        generated = generator.generate("hello", max_tokens=77)
        result = generator.generate_with_metadata("hello", max_tokens=77)

    assert isinstance(generated, str)
    assert generated == '{"summary":"ok"}'
    assert result["content"] == '{"summary":"ok"}'
    assert result["response"]["usage"]["total_tokens"] == 42
    assert result["latency_ms"] >= 0


def test_generate_with_metadata_preserves_timeout_exception(tmp_path):
    with patch(
        "providers.llm.openrouter_llm.httpx.post",
        side_effect=httpx.TimeoutException("timed out"),
    ), patch("providers.llm.openrouter_llm.time.sleep", return_value=None):
        generator = OpenRouterGenerator(
            model="openai/gpt-4.1-mini",
            api_key="secret-key",
            trace_capture={"enabled": True, "directory": str(tmp_path)},
        )

        with pytest.raises(httpx.TimeoutException, match="timed out"):
            generator.generate_with_metadata("hello", max_tokens=77)


def test_generate_with_metadata_returns_isolated_request_and_response(tmp_path):
    response_json = {
        "id": "chatcmpl-123",
        "choices": [{"message": {"content": '{"summary":"ok"}'}}],
        "usage": {"total_tokens": 42},
    }
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(200, json=response_json, request=request)
    captured: dict[str, object] = {}

    with patch("providers.llm.openrouter_llm.httpx.post", return_value=response):
        generator = OpenRouterGenerator(
            model="openai/gpt-4.1-mini",
            api_key="secret-key",
            trace_capture={"enabled": True, "directory": str(tmp_path)},
        )
        original_record = generator.trace_recorder.record

        def record_spy(*args, **kwargs):
            captured["request"] = kwargs["request"]
            captured["response"] = kwargs["response"]
            return original_record(*args, **kwargs)

        generator.trace_recorder.record = record_spy  # type: ignore[method-assign]
        result = generator.generate_with_metadata("hello", max_tokens=77)

    result["request"]["payload"]["messages"][1]["content"] = "mutated"
    result["response"]["usage"]["total_tokens"] = -1

    assert captured["request"]["payload"]["messages"][1]["content"] == "hello"
    assert captured["response"]["usage"]["total_tokens"] == 42
