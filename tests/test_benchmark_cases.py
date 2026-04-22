import json
from pathlib import Path

from core.benchmarking.cases import load_trace_rows, prepare_cases


def test_load_trace_rows_reads_saved_prompt_and_output():
    fixture_path = Path("tests/fixtures/benchmarks/sample_traces.jsonl")

    rows = load_trace_rows(fixture_path)

    assert rows[0].prompt.startswith("Extract metadata from this document")
    assert rows[0].baseline_response.startswith("{")


def test_prepare_cases_filters_smoke_rows_and_caps_to_limit(tmp_path):
    fixture_dir = Path("tests/fixtures/benchmarks")

    result = prepare_cases(trace_dir=fixture_dir, out_dir=tmp_path, limit=2, seed=7)

    assert result.selected_count == 2
    assert "smoke" not in {case["difficulty"] for case in result.cases}
    manifest = json.loads((tmp_path / "cases" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["selected_count"] == 2
