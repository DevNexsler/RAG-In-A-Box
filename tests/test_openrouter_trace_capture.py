import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import yaml

from providers.llm.openrouter_llm import OpenRouterGenerator
from providers.llm.trace_recorder import LLMTraceRecorder


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _trace_files(trace_dir: Path) -> list[Path]:
    return sorted(trace_dir.glob("*.jsonl"))


class TestLLMTraceRecorder:
    def test_disabled_recorder_noops(self, tmp_path):
        recorder = LLMTraceRecorder(
            provider="openrouter",
            model="openai/gpt-4.1-mini",
            enabled=False,
            directory=tmp_path,
        )

        recorder.record(
            request={"payload": {"messages": [{"role": "user", "content": "hello"}]}},
            response={"id": "resp_123"},
            success=True,
            latency_ms=12.0,
        )

        assert _trace_files(tmp_path) == []

    def test_enabled_recorder_appends_jsonl(self, tmp_path):
        recorder = LLMTraceRecorder(
            provider="openrouter",
            model="openai/gpt-4.1-mini",
            enabled=True,
            directory=tmp_path,
        )

        recorder.record(
            request={"payload": {"messages": [{"role": "user", "content": "hello"}]}},
            response={"id": "resp_123"},
            success=True,
            latency_ms=12.0,
        )

        files = _trace_files(tmp_path)
        assert len(files) == 1
        rows = _read_jsonl(files[0])
        assert len(rows) == 1
        assert rows[0]["provider"] == "openrouter"
        assert rows[0]["model"] == "openai/gpt-4.1-mini"
        assert rows[0]["success"] is True
        assert rows[0]["response"]["id"] == "resp_123"

    def test_enabled_recorder_creates_parent_directory(self, tmp_path):
        trace_dir = tmp_path / "nested" / "llm-traces"
        recorder = LLMTraceRecorder(
            provider="openrouter",
            model="openai/gpt-4.1-mini",
            enabled=True,
            directory=trace_dir,
        )

        recorder.record(
            request={"payload": {"messages": [{"role": "user", "content": "hello"}]}},
            response={"id": "resp_123"},
            success=True,
            latency_ms=12.0,
        )

        assert trace_dir.exists()
        assert len(_trace_files(trace_dir)) == 1

    def test_write_failure_is_swallowed(self, tmp_path, monkeypatch):
        recorder = LLMTraceRecorder(
            provider="openrouter",
            model="openai/gpt-4.1-mini",
            enabled=True,
            directory=tmp_path,
        )

        def _boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(Path, "open", _boom)

        recorder.record(
            request={"payload": {"messages": [{"role": "user", "content": "hello"}]}},
            response={"id": "resp_123"},
            success=True,
            latency_ms=12.0,
        )


class TestOpenRouterTraceCapture:
    def test_success_trace_written(self, tmp_path):
        response_json = {
            "id": "chatcmpl-123",
            "choices": [{"message": {"content": '{"summary":"ok"}'}}],
        }
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        response = httpx.Response(200, json=response_json, request=request)

        with patch("providers.llm.openrouter_llm.httpx.post", return_value=response):
            generator = OpenRouterGenerator(
                model="openai/gpt-4.1-mini",
                api_key="secret-key",
                trace_capture={"enabled": True, "directory": str(tmp_path)},
            )
            result = generator.generate("hello world", max_tokens=77)

        assert result == '{"summary":"ok"}'
        files = _trace_files(tmp_path)
        assert len(files) == 1
        rows = _read_jsonl(files[0])
        assert len(rows) == 1
        row = rows[0]
        assert row["success"] is True
        assert row["request"]["payload"]["max_tokens"] == 77
        assert row["request"]["payload"]["messages"][1]["content"] == "hello world"
        assert row["response"]["id"] == "chatcmpl-123"
        assert row["latency_ms"] >= 0
        assert "secret-key" not in json.dumps(row)

    def test_http_error_trace_written_before_raise(self, tmp_path):
        response = httpx.Response(429, text='{"error":"rate_limited"}')
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        error = httpx.HTTPStatusError("boom", request=request, response=response)

        class FailingResponse:
            text = response.text
            status_code = 429

            def raise_for_status(self):
                raise error

        with patch("providers.llm.openrouter_llm.httpx.post", return_value=FailingResponse()):
            generator = OpenRouterGenerator(
                model="openai/gpt-4.1-mini",
                api_key="secret-key",
                trace_capture={"enabled": True, "directory": str(tmp_path)},
            )
            with pytest.raises(httpx.HTTPStatusError):
                generator.generate("hello world", max_tokens=77)

        rows = _read_jsonl(_trace_files(tmp_path)[0])
        assert len(rows) == 1
        row = rows[0]
        assert row["success"] is False
        assert row["error"]["type"] == "HTTPStatusError"
        assert row["error"]["status_code"] == 429
        assert "rate_limited" in row["error"]["body"]

    def test_timeout_final_failure_recorded_once(self, tmp_path):
        with patch(
            "providers.llm.openrouter_llm.httpx.post",
            side_effect=httpx.TimeoutException("timed out"),
        ), patch("providers.llm.openrouter_llm.time.sleep", return_value=None):
            generator = OpenRouterGenerator(
                model="openai/gpt-4.1-mini",
                api_key="secret-key",
                trace_capture={"enabled": True, "directory": str(tmp_path)},
            )
            with pytest.raises(httpx.TimeoutException):
                generator.generate("hello world", max_tokens=77)

        rows = _read_jsonl(_trace_files(tmp_path)[0])
        assert len(rows) == 1
        row = rows[0]
        assert row["success"] is False
        assert row["error"]["type"] == "TimeoutException"
        assert row["error"]["message"] == "timed out"


class TestConfigExamples:
    @pytest.mark.parametrize(
        "config_path",
        [
            "config.yaml.example",
            "config_test.yaml.example",
            "config.local.yaml.example",
            "config.vps.yaml.example",
        ],
    )
    def test_example_configs_include_trace_capture_defaults(self, config_path):
        data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        trace_cfg = data["enrichment"]["trace_capture"]
        assert trace_cfg["enabled"] is False
        assert trace_cfg["directory"] == ".evals/llm-traces"

    def test_gitignore_ignores_evals_directory(self):
        gitignore = Path(".gitignore").read_text(encoding="utf-8")
        assert ".evals/" in gitignore
