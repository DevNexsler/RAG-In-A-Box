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


def test_openrouter_json_schema_includes_importance(tmp_path):
    response_json = {
        "id": "chatcmpl-123",
        "choices": [{"message": {"content": '{"summary":"ok","importance":0.8}'}}],
        "usage": {"total_tokens": 42},
    }
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(200, json=response_json, request=request)

    with patch("providers.llm.openrouter_llm.httpx.post", return_value=response) as mock_post:
        generator = OpenRouterGenerator(
            model="openai/gpt-4.1-mini",
            api_key="secret-key",
            trace_capture={"enabled": True, "directory": str(tmp_path)},
        )
        generator.generate_with_metadata("hello", max_tokens=77)

    payload = mock_post.call_args.kwargs["json"]
    schema = payload["response_format"]["json_schema"]["schema"]

    assert "importance" in schema["properties"]
    assert "importance" in schema["required"]


def test_generate_with_metadata_uses_shorter_connect_timeout(tmp_path):
    response_json = {
        "id": "chatcmpl-123",
        "choices": [{"message": {"content": '{"summary":"ok","importance":0.8}'}}],
        "usage": {"total_tokens": 42},
    }
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(200, json=response_json, request=request)

    with patch("providers.llm.openrouter_llm.httpx.post", return_value=response) as mock_post:
        generator = OpenRouterGenerator(
            model="google/gemma-4-31b-it",
            api_key="secret-key",
            timeout=300.0,
            trace_capture={"enabled": True, "directory": str(tmp_path)},
        )
        generator.generate_with_metadata("hello", max_tokens=77)

    timeout = mock_post.call_args.kwargs["timeout"]

    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == 10.0
    assert timeout.read == 300.0


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


class SequencedReplayClient:
    def __init__(self, *responses: object) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def generate_with_metadata(self, user_prompt: str, max_tokens: int = 512):
        self.calls.append({"user_prompt": user_prompt, "max_tokens": max_tokens})
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _write_case_and_gold(bench_dir, *, case_id: str) -> None:
    case_dir = bench_dir / "cases"
    gold_dir = bench_dir / "gold"
    case_dir.mkdir(exist_ok=True)
    gold_dir.mkdir(exist_ok=True)
    (case_dir / f"{case_id}.json").write_text(
        f"""{{
  "case_id": "{case_id}",
  "prompt": "Prompt text for {case_id}",
  "baseline_response": "{{}}",
  "title": "Lease renewal",
  "source_type": "pdf",
  "category": "housing",
  "difficulty": "easy",
  "trace_file": "trace.jsonl",
  "trace_line": 1
}}""",
        encoding="utf-8",
    )
    (gold_dir / f"{case_id}.json").write_text(
        f"""{{
  "case_id": "{case_id}",
  "canonical": {{
    "summary": "Lease renewal request.",
    "doc_type": ["lease"],
    "entities_people": [],
    "entities_places": [],
    "entities_orgs": [],
    "entities_dates": ["2026-03-01"],
    "topics": ["lease renewal"],
    "keywords": ["renewal terms"],
    "key_facts": ["Tenant requested renewal."],
    "suggested_tags": ["lease"],
    "suggested_folder": "Housing/Leases",
    "importance": "0.8"
  }},
  "alternates": {{
    "suggested_tags": [],
    "suggested_folder": []
  }}
}}""",
        encoding="utf-8",
    )


def test_run_benchmark_persists_per_case_results_and_summary(tmp_path):
    fixture_bench_dir = tmp_path / "benchmarks"
    fixture_bench_dir.mkdir(parents=True)
    _write_case_and_gold(fixture_bench_dir, case_id="case_0001")

    fake_client = FakeReplayClient(
        content='{"summary":"Lease renewal request.","doc_type":["lease"],"entities_people":[],"entities_places":[],"entities_orgs":[],"entities_dates":["2026-03-01"],"topics":["lease renewal"],"keywords":["renewal terms"],"key_facts":["Tenant requested renewal."],"suggested_tags":["lease"],"suggested_folder":"Housing/Leases","importance":0.8}'
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
    assert run.summary["success_rate"] == 1.0
    assert run.summary["parse_failure_rate"] == 0.0
    assert run.summary["transport_failure_rate"] == 0.0
    assert run.summary["latency_p50"] == 12.5
    assert run.summary["latency_p95"] == 12.5
    assert run.summary["token_total"] == 42
    assert run.summary["token_average"] == 42.0
    assert fake_client.calls[0]["user_prompt"] == "Prompt text for case_0001"


def test_run_benchmark_records_parse_failures_per_case(tmp_path):
    fixture_bench_dir = tmp_path / "benchmarks"
    fixture_bench_dir.mkdir(parents=True)
    _write_case_and_gold(fixture_bench_dir, case_id="case_0001")

    fake_client = FakeReplayClient(content="not json")

    run = run_benchmark(
        bench_dir=fixture_bench_dir,
        model="openai/gpt-4.1-mini",
        run_id="baseline",
        replay_client=fake_client,
    )

    assert run.summary["case_count"] == 1
    assert run.summary["parse_failures"] == 1
    assert run.summary["parse_failure_rate"] == 1.0
    assert run.summary["success_rate"] == 0.0
    assert run.per_case[0]["score"]["total_score"] == 0.0
    assert run.per_case[0]["score"]["reliability"]["parse_failed"] is True


def test_run_benchmark_classifies_internal_value_error_separately(tmp_path, monkeypatch):
    fixture_bench_dir = tmp_path / "benchmarks"
    fixture_bench_dir.mkdir(parents=True)
    _write_case_and_gold(fixture_bench_dir, case_id="case_0001")

    fake_client = FakeReplayClient(
        content='{"summary":"Lease renewal request.","doc_type":["lease"],"entities_people":[],"entities_places":[],"entities_orgs":[],"entities_dates":["2026-03-01"],"topics":["lease renewal"],"keywords":["renewal terms"],"key_facts":["Tenant requested renewal."],"suggested_tags":["lease"],"suggested_folder":"Housing/Leases","importance":0.8}'
    )

    def explode(*args, **kwargs):
        raise ValueError("gold data mismatch")

    monkeypatch.setattr("core.benchmarking.runner.score_raw_case", explode)

    run = run_benchmark(
        bench_dir=fixture_bench_dir,
        model="openai/gpt-4.1-mini",
        run_id="baseline",
        replay_client=fake_client,
    )

    assert run.summary["parse_failure_rate"] == 0.0
    assert run.summary["transport_failure_rate"] == 0.0
    assert run.summary["success_rate"] == 0.0
    assert run.per_case[0]["status"] == "internal_error"
    assert run.per_case[0]["score"]["reliability"]["parse_failed"] is False
    assert run.per_case[0]["score"]["reliability"]["internal_failed"] is True


def test_run_benchmark_summary_includes_transport_latency_and_token_aggregates(tmp_path):
    fixture_bench_dir = tmp_path / "benchmarks"
    fixture_bench_dir.mkdir(parents=True)
    _write_case_and_gold(fixture_bench_dir, case_id="case_0001")
    _write_case_and_gold(fixture_bench_dir, case_id="case_0002")
    _write_case_and_gold(fixture_bench_dir, case_id="case_0003")

    fake_client = SequencedReplayClient(
        {
            "content": '{"summary":"Lease renewal request.","doc_type":["lease"],"entities_people":[],"entities_places":[],"entities_orgs":[],"entities_dates":["03/01/2026"],"topics":["lease renewal"],"keywords":["renewal terms"],"key_facts":["Tenant requested lease renewal."],"suggested_tags":["lease"],"suggested_folder":"Housing/Leases/","importance":0.8}',
            "request": {"payload": {"messages": [{"role": "user", "content": "one"}]}},
            "response": {"usage": {"total_tokens": 20}},
            "latency_ms": 50.0,
        },
        {
            "content": "not json",
            "request": {"payload": {"messages": [{"role": "user", "content": "two"}]}},
            "response": {"usage": {"total_tokens": 10}},
            "latency_ms": 100.0,
        },
        httpx.TimeoutException("timed out"),
    )

    run = run_benchmark(
        bench_dir=fixture_bench_dir,
        model="openai/gpt-4.1-mini",
        run_id="mixed",
        replay_client=fake_client,
    )

    assert run.summary["case_count"] == 3
    assert run.summary["success_rate"] == pytest.approx(1 / 3)
    assert run.summary["parse_failure_rate"] == pytest.approx(1 / 3)
    assert run.summary["transport_failure_rate"] == pytest.approx(1 / 3)
    assert run.summary["latency_p50"] == 75.0
    assert run.summary["latency_p95"] == 97.5
    assert run.summary["token_total"] == 30
    assert run.summary["token_average"] == 10.0
