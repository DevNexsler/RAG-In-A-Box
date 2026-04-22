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
    assert "smoke" not in {case.difficulty for case in result.cases}
    assert {case.category for case in result.cases} == {"finance", "housing"}
    manifest = json.loads((tmp_path / "cases" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["selected_count"] == 2
    assert (tmp_path / "cases" / "case_0001.json").is_file()
    assert (tmp_path / "cases" / "case_0002.json").is_file()
    case_payload = json.loads((tmp_path / "cases" / "case_0001.json").read_text(encoding="utf-8"))
    assert case_payload["case_id"] == "case_0001"


def test_prepare_cases_selection_is_deterministic_for_seed(tmp_path):
    fixture_dir = Path("tests/fixtures/benchmarks")

    first = prepare_cases(trace_dir=fixture_dir, out_dir=tmp_path / "run-a", limit=2, seed=7)
    second = prepare_cases(trace_dir=fixture_dir, out_dir=tmp_path / "run-b", limit=2, seed=7)
    third = prepare_cases(trace_dir=fixture_dir, out_dir=tmp_path / "run-c", limit=2, seed=8)

    first_titles = [case.title for case in first.cases]
    second_titles = [case.title for case in second.cases]
    third_titles = [case.title for case in third.cases]

    assert first_titles == second_titles
    assert set(first_titles) != set(third_titles)
    assert "Vendor payment follow-up" in first_titles
    assert "Vendor payment follow-up" in third_titles
    assert {"Lease renewal", "Property inspection notes"} & set(first_titles)
    assert {"Lease renewal", "Property inspection notes"} & set(third_titles)


def test_prepare_cases_rerun_removes_stale_case_files(tmp_path):
    fixture_dir = Path("tests/fixtures/benchmarks")

    prepare_cases(trace_dir=fixture_dir, out_dir=tmp_path, limit=3, seed=7)
    rerun = prepare_cases(trace_dir=fixture_dir, out_dir=tmp_path, limit=1, seed=7)

    assert rerun.selected_count == 1
    case_files = sorted(path.name for path in (tmp_path / "cases").glob("case_*.json"))
    assert case_files == ["case_0001.json"]
