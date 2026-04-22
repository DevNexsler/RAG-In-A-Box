from unittest.mock import patch

import httpx

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
        result = generator.generate_with_metadata("hello", max_tokens=77)

    assert result["content"] == '{"summary":"ok"}'
    assert result["response"]["usage"]["total_tokens"] == 42
    assert result["latency_ms"] >= 0
