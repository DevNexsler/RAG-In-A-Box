import json
from pathlib import Path

from core.benchmarking.cases import load_trace_rows, prepare_audit_cases, prepare_cases


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


def test_prepare_audit_cases_seeds_manual_only_gold_stubs(tmp_path):
    source_dir = tmp_path / "benchmarks"
    prepare_cases(trace_dir=Path("tests/fixtures/benchmarks"), out_dir=source_dir, limit=3, seed=7)

    result = prepare_audit_cases(source_bench_dir=source_dir, out_dir=tmp_path / "audit", limit=2, seed=11)

    assert result.selected_count == 2
    assert (tmp_path / "audit" / "cases" / "manifest.json").is_file()
    assert (tmp_path / "audit" / "status").is_dir()
    assert (tmp_path / "audit" / "runs").is_dir()
    gold_path = tmp_path / "audit" / "gold" / "case_0001.json"
    assert gold_path.is_file()
    gold_payload = json.loads(gold_path.read_text(encoding="utf-8"))
    assert gold_payload["label_source"] == "manual_audit"
    assert gold_payload["alternates"] == {"suggested_tags": [], "suggested_folder": []}
    assert gold_payload["summary_rubric"] == {
        "coverage": [],
        "brevity": {"max_sentences": 0, "max_words": 0},
        "hallucination": [],
    }


def test_prepare_audit_cases_does_not_copy_existing_gold_labels(tmp_path):
    source_dir = tmp_path / "benchmarks"
    prepare_cases(trace_dir=Path("tests/fixtures/benchmarks"), out_dir=source_dir, limit=2, seed=7)
    source_gold_dir = source_dir / "gold"
    source_gold_dir.mkdir(parents=True)
    (source_gold_dir / "case_0001.json").write_text(
        json.dumps(
            {
                "case_id": "case_0001",
                "label_source": "baseline_assisted",
                "canonical": {
                    "summary": "preexisting",
                    "doc_type": ["invoice"],
                    "entities_people": [],
                    "entities_places": [],
                    "entities_orgs": [],
                    "entities_dates": [],
                    "topics": ["billing"],
                    "keywords": ["vendor"],
                    "key_facts": ["fact"],
                    "suggested_tags": ["finance"],
                    "suggested_folder": "Finance/Bills",
                    "importance": "0.6",
                },
                "alternates": {"suggested_tags": ["billing"], "suggested_folder": ["Finance"]},
                "summary_rubric": {
                    "coverage": ["old"],
                    "brevity": {"max_sentences": 2, "max_words": 40},
                    "hallucination": ["none"],
                },
            }
        ),
        encoding="utf-8",
    )

    prepare_audit_cases(source_bench_dir=source_dir, out_dir=tmp_path / "audit", limit=1, seed=11)

    audit_gold = json.loads((tmp_path / "audit" / "gold" / "case_0001.json").read_text(encoding="utf-8"))
    assert audit_gold["canonical"]["summary"] == ""
    assert audit_gold["canonical"]["doc_type"] == []
    assert audit_gold["alternates"] == {"suggested_tags": [], "suggested_folder": []}
    assert audit_gold["summary_rubric"]["coverage"] == []
