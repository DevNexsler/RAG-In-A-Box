from pathlib import Path

from core.benchmarking.mining import load_trace_metadata, score_hard_flags, select_hard_cases


def test_load_trace_metadata_extracts_safe_fields_only():
    rows = load_trace_metadata(Path("tests/fixtures/benchmarks/hard_traces.jsonl"))

    assert rows[0].trace_file == "hard_traces.jsonl"
    assert rows[0].trace_line == 1
    assert rows[0].title == "Payment cleared notice"
    assert rows[0].source_type == "pg_message"
    assert rows[0].prompt_hash
    assert rows[0].prompt_excerpt == ""


def test_score_hard_flags_detects_real_hard_patterns():
    row = load_trace_metadata(Path("tests/fixtures/benchmarks/hard_traces.jsonl"))[0]

    scored = score_hard_flags(row)

    assert "nearby_context" in scored.flags
    assert "taxonomy_bloat" in scored.flags
    assert "link_noise" in scored.flags
    assert "business_critical" in scored.flags
    assert "slow_success" in scored.flags
    assert scored.hard_score > 0


def test_select_hard_cases_keeps_provider_failures_separate():
    rows = load_trace_metadata(Path("tests/fixtures/benchmarks/hard_traces.jsonl"))

    selected = select_hard_cases(rows, limit=10)

    assert selected.hard_cases
    assert selected.provider_failure_cases
    assert selected.provider_failure_cases[0].failure_status_code == 402
