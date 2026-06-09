import json
from pathlib import Path
from unittest.mock import patch

import httpx

from providers.llm import build_llm_provider
from providers.llm.litellm_llm import LiteLLMGenerator


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
