from pathlib import Path
import json

from core.benchmarking.mining import (
    HUGE_PROMPT_CHARS,
    LONG_PROMPT_CHARS,
    load_trace_metadata,
    score_hard_flags,
    select_hard_cases,
)
from core.benchmarking.models import TraceMetadata

HARD_TRACES = Path("tests/fixtures/benchmarks/hard/hard_traces.jsonl")


def test_load_trace_metadata_extracts_safe_fields_only():
    rows = load_trace_metadata(HARD_TRACES)

    assert rows[0].trace_file == "hard_traces.jsonl"
    assert rows[0].trace_line == 1
    assert rows[0].title == "Payment cleared notice"
    assert rows[0].source_type == "pg_message"
    assert rows[0].prompt_hash
    assert rows[0].prompt_excerpt == ""


def test_load_trace_metadata_preserves_duplicate_prompt_lines():
    rows = load_trace_metadata(HARD_TRACES)

    assert len(rows) == 5
    assert rows[0].prompt_hash == rows[2].prompt_hash
    assert rows[2].trace_line == 3


def test_load_trace_metadata_skips_malformed_jsonl_lines(tmp_path):
    trace_path = tmp_path / "mixed.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                "{not-json",
                json.dumps(
                    {
                        "ts": "2026-05-07T00:00:00+00:00",
                        "provider": "openrouter",
                        "model": "openai/gpt-4.1-mini",
                        "success": True,
                        "request": {
                            "payload": {
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": "Document title: Valid row\nDocument type: pg_message",
                                    }
                                ]
                            }
                        },
                        "response": {"choices": [{"message": {"content": "{}"}}]},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    rows = load_trace_metadata(trace_path)

    assert len(rows) == 1
    assert rows[0].trace_line == 2
    assert rows[0].title == "Valid row"


def test_load_trace_metadata_tolerates_bad_nested_shapes(tmp_path):
    trace_path = tmp_path / "bad_shapes.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "ts": "2026-05-08T00:00:00+00:00",
                "provider": "openrouter",
                "model": "openai/gpt-4.1-mini",
                "success": False,
                "request": [],
                "response": [],
                "error": "not-an-error-object",
            }
        ),
        encoding="utf-8",
    )

    rows = load_trace_metadata(trace_path)

    assert len(rows) == 1
    assert rows[0].prompt_hash
    assert rows[0].failure_type is None


def test_score_hard_flags_detects_real_hard_patterns():
    row = load_trace_metadata(HARD_TRACES)[0]

    scored = score_hard_flags(row)

    assert "nearby_context" in scored.flags
    assert "taxonomy_bloat" in scored.flags
    assert "link_noise" in scored.flags
    assert "business_critical" in scored.flags
    assert "slow_success" in scored.flags
    assert scored.hard_score > 0


def test_score_hard_flags_detects_threshold_and_parse_patterns():
    rows = load_trace_metadata(HARD_TRACES)

    threshold_row = TraceMetadata(
        trace_file="synthetic.jsonl",
        trace_line=1,
        timestamp="2026-05-06T00:00:00+00:00",
        provider="openrouter",
        model="openai/gpt-4.1-mini",
        success=True,
        latency_ms=1,
        title="Large prompt",
        source_type="pg_message",
        prompt_length=HUGE_PROMPT_CHARS,
        text_length=HUGE_PROMPT_CHARS,
        prompt_hash="synthetic-huge",
        response_looks_parseable=True,
    )

    threshold_scored = score_hard_flags(threshold_row)
    parse_scored = score_hard_flags(rows[4])

    assert LONG_PROMPT_CHARS < HUGE_PROMPT_CHARS
    assert "long_prompt" in threshold_scored.flags
    assert "huge_prompt" in threshold_scored.flags
    assert "very_slow_success" in parse_scored.flags
    assert "parse_suspect" in parse_scored.flags


def test_select_hard_cases_keeps_provider_failures_separate():
    rows = load_trace_metadata(HARD_TRACES)

    selected = select_hard_cases(rows, limit=10)

    assert selected.hard_cases
    assert selected.provider_failure_cases
    assert selected.provider_failure_cases[0].failure_status_code == 402


def test_select_hard_cases_requires_provider_failure_type_and_status_code():
    rows = load_trace_metadata(HARD_TRACES)

    selected = select_hard_cases(rows, limit=10)

    assert [row.failure_type for row in selected.provider_failure_cases] == ["HTTPStatusError"]


def test_select_hard_cases_keeps_failure_when_duplicate_prompt_succeeds_first():
    success_row, _, failure_row = load_trace_metadata(HARD_TRACES)[:3]
    failure_after_success = TraceMetadata(
        trace_file=failure_row.trace_file,
        trace_line=99,
        timestamp=failure_row.timestamp,
        provider=failure_row.provider,
        model=failure_row.model,
        success=False,
        latency_ms=50,
        title=failure_row.title,
        source_type=failure_row.source_type,
        prompt_length=failure_row.prompt_length,
        text_length=failure_row.text_length,
        prompt_hash=success_row.prompt_hash,
        response_looks_parseable=False,
        failure_type="HTTPStatusError",
        failure_status_code=429,
    )

    selected = select_hard_cases([success_row, failure_after_success], limit=10)

    assert [row.failure_status_code for row in selected.provider_failure_cases] == [429]
