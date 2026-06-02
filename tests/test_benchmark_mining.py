from pathlib import Path

from core.benchmarking.mining import (
    HUGE_PROMPT_CHARS,
    LONG_PROMPT_CHARS,
    load_trace_metadata,
    score_hard_flags,
    select_hard_cases,
)
from core.benchmarking.models import TraceMetadata


def test_load_trace_metadata_extracts_safe_fields_only():
    rows = load_trace_metadata(Path("tests/fixtures/benchmarks/hard_traces.jsonl"))

    assert rows[0].trace_file == "hard_traces.jsonl"
    assert rows[0].trace_line == 1
    assert rows[0].title == "Payment cleared notice"
    assert rows[0].source_type == "pg_message"
    assert rows[0].prompt_hash
    assert rows[0].prompt_excerpt == ""


def test_load_trace_metadata_preserves_duplicate_prompt_lines():
    rows = load_trace_metadata(Path("tests/fixtures/benchmarks/hard_traces.jsonl"))

    assert len(rows) == 5
    assert rows[0].prompt_hash == rows[2].prompt_hash
    assert rows[2].trace_line == 3


def test_score_hard_flags_detects_real_hard_patterns():
    row = load_trace_metadata(Path("tests/fixtures/benchmarks/hard_traces.jsonl"))[0]

    scored = score_hard_flags(row)

    assert "nearby_context" in scored.flags
    assert "taxonomy_bloat" in scored.flags
    assert "link_noise" in scored.flags
    assert "business_critical" in scored.flags
    assert "slow_success" in scored.flags
    assert scored.hard_score > 0


def test_score_hard_flags_detects_threshold_and_parse_patterns():
    rows = load_trace_metadata(Path("tests/fixtures/benchmarks/hard_traces.jsonl"))

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
    rows = load_trace_metadata(Path("tests/fixtures/benchmarks/hard_traces.jsonl"))

    selected = select_hard_cases(rows, limit=10)

    assert selected.hard_cases
    assert selected.provider_failure_cases
    assert selected.provider_failure_cases[0].failure_status_code == 402


def test_select_hard_cases_requires_provider_failure_type_and_status_code():
    rows = load_trace_metadata(Path("tests/fixtures/benchmarks/hard_traces.jsonl"))

    selected = select_hard_cases(rows, limit=10)

    assert [row.failure_type for row in selected.provider_failure_cases] == ["HTTPStatusError"]
