from unittest.mock import patch

import httpx
import pytest

from core.benchmarking.runner import run_benchmark
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


class FakeReplayClient:
    def __init__(self, *, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, object]] = []

    def generate_with_metadata(self, user_prompt: str, max_tokens: int = 512):
        self.calls.append({"user_prompt": user_prompt, "max_tokens": max_tokens})
        return {
            "content": self.content,
            "request": {
                "payload": {
                    "messages": [
                        {"role": "system", "content": "system"},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": max_tokens,
                }
            },
            "response": {"usage": {"total_tokens": 42}},
            "latency_ms": 12.5,
        }


def test_run_benchmark_persists_per_case_results_and_summary(tmp_path):
    fixture_bench_dir = tmp_path / "benchmarks"
    fixture_bench_dir.mkdir(parents=True)
    case_dir = fixture_bench_dir / "cases"
    gold_dir = fixture_bench_dir / "gold"
    case_dir.mkdir()
    gold_dir.mkdir()

    (case_dir / "case_0001.json").write_text(
        """{
  "case_id": "case_0001",
  "prompt": "Prompt text",
  "baseline_response": "{}",
  "title": "Lease renewal",
  "source_type": "pdf",
  "category": "housing",
  "difficulty": "easy",
  "trace_file": "trace.jsonl",
  "trace_line": 1
}""",
        encoding="utf-8",
    )
    (gold_dir / "case_0001.json").write_text(
        """{
  "case_id": "case_0001",
  "canonical": {
    "summary": "Lease renewal request.",
    "doc_type": ["lease"],
    "entities_people": [],
    "entities_places": [],
    "entities_orgs": [],
    "entities_dates": [],
    "topics": ["lease renewal"],
    "keywords": ["renewal terms"],
    "key_facts": ["Tenant requested renewal."],
    "suggested_tags": ["lease"],
    "suggested_folder": "Housing/Leases",
    "importance": "0.8"
  },
  "alternates": {
    "suggested_tags": [],
    "suggested_folder": []
  }
}""",
        encoding="utf-8",
    )

    fake_client = FakeReplayClient(
        content='{"summary":"Lease renewal request.","doc_type":["lease"],"entities_people":[],"entities_places":[],"entities_orgs":[],"entities_dates":[],"topics":["lease renewal"],"keywords":["renewal terms"],"key_facts":["Tenant requested renewal."],"suggested_tags":["lease"],"suggested_folder":"Housing/Leases","importance":0.8}'
    )

    run = run_benchmark(
        bench_dir=fixture_bench_dir,
        model="openai/gpt-4.1-mini",
        run_id="baseline",
        replay_client=fake_client,
    )

    assert (fixture_bench_dir / "runs" / "baseline" / "per_case.jsonl").exists()
    assert (fixture_bench_dir / "runs" / "baseline" / "summary.json").exists()
    assert run.summary["case_count"] == 1
    assert run.summary["parse_failures"] == 0
    assert fake_client.calls[0]["user_prompt"] == "Prompt text"


def test_run_benchmark_records_parse_failures_per_case(tmp_path):
    fixture_bench_dir = tmp_path / "benchmarks"
    fixture_bench_dir.mkdir(parents=True)
    case_dir = fixture_bench_dir / "cases"
    gold_dir = fixture_bench_dir / "gold"
    case_dir.mkdir()
    gold_dir.mkdir()

    (case_dir / "case_0001.json").write_text(
        """{
  "case_id": "case_0001",
  "prompt": "Prompt text",
  "baseline_response": "{}",
  "title": "Lease renewal",
  "source_type": "pdf",
  "category": "housing",
  "difficulty": "easy",
  "trace_file": "trace.jsonl",
  "trace_line": 1
}""",
        encoding="utf-8",
    )
    (gold_dir / "case_0001.json").write_text(
        """{
  "case_id": "case_0001",
  "canonical": {
    "summary": "Lease renewal request.",
    "doc_type": ["lease"],
    "entities_people": [],
    "entities_places": [],
    "entities_orgs": [],
    "entities_dates": [],
    "topics": ["lease renewal"],
    "keywords": ["renewal terms"],
    "key_facts": ["Tenant requested renewal."],
    "suggested_tags": ["lease"],
    "suggested_folder": "Housing/Leases",
    "importance": "0.8"
  },
  "alternates": {
    "suggested_tags": [],
    "suggested_folder": []
  }
}""",
        encoding="utf-8",
    )

    fake_client = FakeReplayClient(content="not json")

    run = run_benchmark(
        bench_dir=fixture_bench_dir,
        model="openai/gpt-4.1-mini",
        run_id="baseline",
        replay_client=fake_client,
    )

    assert run.summary["case_count"] == 1
    assert run.summary["parse_failures"] == 1
    assert run.per_case[0]["score"]["total_score"] == 0.0
    assert run.per_case[0]["score"]["reliability"]["parse_failed"] is True
