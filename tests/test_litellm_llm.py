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
    completion_tokens: int | None,
    reasoning: str = "",
    finish_reason: str = "stop",
) -> httpx.Response:
    request = httpx.Request("POST", "http://litellm.local/v1/chat/completions")
    payload = {
        "id": "chatcmpl-litellm",
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "content": content,
                    "reasoning_content": reasoning,
                },
            }
        ],
    }
    if completion_tokens is not None:
        payload["usage"] = {
            "prompt_tokens": 12,
            "completion_tokens": completion_tokens,
            "total_tokens": 12 + completion_tokens,
        }
    return httpx.Response(
        200,
        json=payload,
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


def test_litellm_generator_retries_explicit_length_without_usage():
    truncated = _completion_response(
        '{"summary":"Partial","doc_type":["email"],"topics":["lease"],"keywords":[',
        completion_tokens=None,
        finish_reason="length",
    )
    recovered = _completion_response(
        '{"summary":"Recovered","doc_type":["email"],'
        '"topics":["lease"],"keywords":["renewal"]}',
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

    assert json.loads(result)["keywords"] == ["renewal"]
    assert post.call_count == 2
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


@pytest.mark.parametrize("empty_content", ["", "   "])
def test_litellm_generator_retries_structurally_empty_success_without_reasoning(
    empty_content,
):
    empty_success = _completion_response(empty_content, completion_tokens=1)
    recovered = _completion_response(
        '{"summary":"Recovered","doc_type":["email"]}',
        completion_tokens=12,
    )

    with patch(
        "providers.llm.litellm_llm.httpx.post",
        side_effect=[empty_success, recovered],
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


def test_litellm_generator_sends_configured_reasoning_effort_on_first_attempt():
    """#1151: enrichment must be able to decline reasoning it never keeps.

    The route discards the reasoning text and keeps only the JSON body, but
    reasoning shares the completion budget — so a reasoning blow-up spends the
    whole budget and leaves nothing for the answer, and the recovery attempt
    pays for a second full prompt.  The first attempt has to be configurable.
    """
    complete = _completion_response(
        '{"summary":"Direct","doc_type":["email"],"topics":["ops"]}',
        completion_tokens=24,
    )

    with patch(
        "providers.llm.litellm_llm.httpx.post",
        return_value=complete,
    ) as post:
        generator = LiteLLMGenerator(
            model="ollama-deepseek-v4-pro",
            base_url="http://litellm.local/v1",
            api_key="secret-key",
            reasoning_effort="none",
        )
        result = generator.generate("large email", max_tokens=77)

    assert json.loads(result)["summary"] == "Direct"
    assert post.call_count == 1
    assert post.call_args_list[0].kwargs["json"]["reasoning_effort"] == "none"


def test_litellm_generator_does_not_repeat_an_identical_reasoning_disabled_request():
    """A recovery attempt that changes nothing is a guaranteed wasted call.

    Retrying with identical parameters is the #0260 trap: the retry burns the
    same way the first attempt did.  Once reasoning is already disabled there
    is nothing left to disable, so the truncation is reported straight away.
    """
    truncated = _completion_response(
        '{"summary":"Partial","doc_type":["email"],"topics":[',
        completion_tokens=77,
        finish_reason="length",
    )

    with patch(
        "providers.llm.litellm_llm.httpx.post",
        return_value=truncated,
    ) as post:
        generator = LiteLLMGenerator(
            model="ollama-deepseek-v4-pro",
            base_url="http://litellm.local/v1",
            api_key="secret-key",
            reasoning_effort="none",
        )
        with pytest.raises(TransientError, match="truncated"):
            generator.generate("large email", max_tokens=77)

    assert post.call_count == 1


def test_litellm_generator_rejects_an_unknown_reasoning_effort():
    """A typo must not fail open into a provider 400 on every document."""
    with pytest.raises(ValueError, match="reasoning_effort"):
        LiteLLMGenerator(
            model="ollama-deepseek-v4-pro",
            base_url="http://litellm.local/v1",
            api_key="secret-key",
            reasoning_effort="off",
        )


def test_build_llm_provider_passes_reasoning_effort(monkeypatch):
    monkeypatch.setenv("LITELLM_MASTER_KEY", "secret-key")
    generator = build_llm_provider(
        {
            "enrichment": {
                "enabled": True,
                "provider": "litellm",
                "model": "ollama-deepseek-v4-pro",
                "reasoning_effort": "none",
            }
        }
    )

    assert isinstance(generator, LiteLLMGenerator)
    assert generator.reasoning_effort == "none"


def test_build_llm_provider_leaves_reasoning_effort_unset_by_default(monkeypatch):
    monkeypatch.setenv("LITELLM_MASTER_KEY", "secret-key")
    generator = build_llm_provider(
        {
            "enrichment": {
                "enabled": True,
                "provider": "litellm",
                "model": "ollama-deepseek-v4-pro",
            }
        }
    )

    assert isinstance(generator, LiteLLMGenerator)
    assert generator.reasoning_effort is None
