import json
import logging
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from core.resilience import TransientError
from providers.llm import build_llm_provider
from providers.llm.litellm_llm import LiteLLMGenerator


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _completion_response(
    content: str,
    *,
    completion_tokens: int,
    reasoning: str = "",
) -> httpx.Response:
    request = httpx.Request("POST", "http://litellm.local/v1/chat/completions")
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-litellm",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": content,
                        "reasoning_content": reasoning,
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": completion_tokens,
                "total_tokens": 12 + completion_tokens,
            },
        },
        request=request,
    )


def test_litellm_generator_uses_configurable_openai_compatible_endpoint(tmp_path):
    response_json = {
        "id": "chatcmpl-litellm",
        "choices": [{"message": {"content": '{"summary":"ok"}'}}],
        "usage": {"total_tokens": 12},
    }
    request = httpx.Request("POST", "http://litellm.local/v1/chat/completions")
    response = httpx.Response(200, json=response_json, request=request)

    with patch("providers.llm.litellm_llm.httpx.post", return_value=response) as post:
        generator = LiteLLMGenerator(
            model="ollama-deepseek-v4-pro",
            base_url="http://litellm.local/v1/",
            api_key="secret-key",
            trace_capture={"enabled": True, "directory": str(tmp_path)},
        )
        result = generator.generate("hello world", max_tokens=77)

    assert result == '{"summary":"ok"}'
    post.assert_called_once()
    assert post.call_args.args[0] == "http://litellm.local/v1/chat/completions"
    payload = post.call_args.kwargs["json"]
    assert payload["model"] == "ollama-deepseek-v4-pro"
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["temperature"] == 0.0
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer secret-key"

    rows = _read_jsonl(next(tmp_path.glob("*.jsonl")))
    assert rows[0]["provider"] == "litellm"
    assert rows[0]["model"] == "ollama-deepseek-v4-pro"
    assert "secret-key" not in json.dumps(rows[0])


def test_build_llm_provider_supports_litellm(monkeypatch):
    monkeypatch.setenv("LITELLM_MASTER_KEY", "secret-key")
    generator = build_llm_provider(
        {
            "enrichment": {
                "enabled": True,
                "provider": "litellm",
                "model": "ollama-deepseek-v4-pro",
                "base_url": "http://host.docker.internal:4000/v1",
                "timeout": 600.0,
            }
        }
    )

    assert isinstance(generator, LiteLLMGenerator)
    assert generator.model == "ollama-deepseek-v4-pro"
    assert generator.base_url == "http://host.docker.internal:4000/v1"
    assert generator.timeout == 600.0


def test_litellm_generator_retries_budget_saturation_without_reasoning():
    truncated = _completion_response(
        '{"context_entities_people":["private-marker"],"summary":"',
        completion_tokens=77,
        reasoning="long hidden reasoning",
    )
    recovered = _completion_response(
        '{"summary":"Recovered","doc_type":["email"],"topics":["ops"]}',
        completion_tokens=24,
    )

    with patch(
        "providers.llm.litellm_llm.httpx.post",
        side_effect=[truncated, recovered],
    ) as post:
        generator = LiteLLMGenerator(
            model="ollama-deepseek-v4-pro",
            base_url="http://litellm.local/v1",
            api_key="secret-key",
        )
        result = generator.generate("large email", max_tokens=77)

    assert json.loads(result)["summary"] == "Recovered"
    assert post.call_count == 2
    assert "reasoning_effort" not in post.call_args_list[0].kwargs["json"]
    assert post.call_args_list[1].kwargs["json"]["reasoning_effort"] == "none"


def test_litellm_generator_retries_reasoning_only_response_without_reasoning():
    reasoning_only = _completion_response(
        "",
        completion_tokens=40,
        reasoning="reasoning consumed output",
    )
    recovered = _completion_response(
        '{"summary":"Recovered","doc_type":["email"]}',
        completion_tokens=12,
    )

    with patch(
        "providers.llm.litellm_llm.httpx.post",
        side_effect=[reasoning_only, recovered],
    ) as post:
        generator = LiteLLMGenerator(
            model="ollama-deepseek-v4-pro",
            base_url="http://litellm.local/v1",
            api_key="secret-key",
        )
        result = generator.generate("large email", max_tokens=77)

    assert json.loads(result)["summary"] == "Recovered"
    assert post.call_count == 2
    assert post.call_args_list[1].kwargs["json"]["reasoning_effort"] == "none"


def test_litellm_generator_raises_transient_after_truncation_retry_exhausted(
    caplog,
):
    private_marker = "private-customer-response-marker"
    first = _completion_response(
        f'{{"summary":"{private_marker}',
        completion_tokens=77,
        reasoning="first reasoning",
    )
    second = _completion_response(
        f'{{"summary":"{private_marker}',
        completion_tokens=77,
        reasoning="second reasoning",
    )

    with patch(
        "providers.llm.litellm_llm.httpx.post",
        side_effect=[first, second],
    ) as post:
        generator = LiteLLMGenerator(
            model="ollama-deepseek-v4-pro",
            base_url="http://litellm.local/v1",
            api_key="secret-key",
        )
        with caplog.at_level(logging.WARNING):
            with pytest.raises(TransientError, match="truncated"):
                generator.generate("large email", max_tokens=77)

    assert post.call_count == 2
    assert private_marker not in caplog.text


def test_litellm_generator_does_not_log_permanent_error_body(caplog):
    private_marker = "private-upstream-error-body"
    request = httpx.Request("POST", "http://litellm.local/v1/chat/completions")
    response = httpx.Response(
        401,
        text=private_marker,
        request=request,
    )

    with patch("providers.llm.litellm_llm.httpx.post", return_value=response):
        generator = LiteLLMGenerator(
            model="ollama-deepseek-v4-pro",
            base_url="http://litellm.local/v1",
            api_key="secret-key",
        )
        with caplog.at_level(logging.ERROR):
            with pytest.raises(httpx.HTTPStatusError):
                generator.generate("large email", max_tokens=77)

    assert private_marker not in caplog.text
