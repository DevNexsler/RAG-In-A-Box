import json
from pathlib import Path

from core.benchmarking.cases import build_labeling_status, write_gold_stub
from core.benchmarking.models import BenchmarkCase


def test_write_gold_stub_creates_all_required_fields(tmp_path):
    case = BenchmarkCase(
        case_id="case_0001",
        prompt="Prompt text",
        baseline_response='{"summary": "baseline", "suggested_tags": ["lease", "renewal"], "suggested_folder": "Housing/Leases"}',
        title="Lease renewal",
        source_type="pdf",
        category="housing",
        difficulty="easy",
        trace_file="trace.jsonl",
        trace_line=1,
    )

    path = write_gold_stub(case, bench_dir=tmp_path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["canonical"]["summary"] == ""
    assert data["alternates"]["suggested_tags"] == ["lease", "renewal"]
    assert data["alternates"]["suggested_folder"] == ["Housing/Leases"]


def test_write_gold_stub_rejects_missing_canonical_fields(tmp_path):
    case = BenchmarkCase(
        case_id="case_0001",
        prompt="Prompt text",
        baseline_response='{"suggested_tags": ["lease"]}',
        title="Lease renewal",
        source_type="pdf",
        category="housing",
        difficulty="easy",
        trace_file="trace.jsonl",
        trace_line=1,
    )
    gold_path = tmp_path / "gold" / "case_0001.json"
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    gold_path.write_text(
        json.dumps(
            {
                "case_id": "case_0001",
                "canonical": {"summary": ""},
                "alternates": {"suggested_tags": [], "suggested_folder": []},
            }
        ),
        encoding="utf-8",
    )

    try:
        write_gold_stub(case, bench_dir=tmp_path)
    except ValueError as exc:
        assert "missing fields" in str(exc)
    else:
        raise AssertionError("expected ValueError for incomplete canonical payload")


def test_labeling_status_counts_done_and_remaining():
    fixture_bench_dir = Path("tests/fixtures/benchmarks")

    status = build_labeling_status(bench_dir=fixture_bench_dir)

    assert status["labeled"] == 1
    assert status["remaining"] >= 0
    assert status["next_case_id"] is None
    status_path = fixture_bench_dir / "status" / "labeling_status.json"
    assert status_path.is_file()
    saved = json.loads(status_path.read_text(encoding="utf-8"))
    assert saved["next_case_id"] is None


def test_labeling_status_sets_next_case_id_for_unlabeled_case(tmp_path):
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir(parents=True)
    case_one = {
        "case_id": "case_0001",
        "prompt": "Prompt one",
        "baseline_response": "{}",
        "title": "Case one",
        "source_type": "pdf",
        "category": "housing",
        "difficulty": "easy",
        "trace_file": "trace.jsonl",
        "trace_line": 1,
    }
    case_two = {
        "case_id": "case_0002",
        "prompt": "Prompt two",
        "baseline_response": "{}",
        "title": "Case two",
        "source_type": "pdf",
        "category": "housing",
        "difficulty": "easy",
        "trace_file": "trace.jsonl",
        "trace_line": 2,
    }
    (cases_dir / "case_0001.json").write_text(json.dumps(case_one), encoding="utf-8")
    (cases_dir / "case_0002.json").write_text(json.dumps(case_two), encoding="utf-8")
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir(parents=True)
    (gold_dir / "case_0001.json").write_text(
        json.dumps(
            {
                "case_id": "case_0001",
                "canonical": {
                    "summary": "done",
                    "doc_type": [],
                    "entities_people": [],
                    "entities_places": [],
                    "entities_orgs": [],
                    "entities_dates": [],
                    "topics": [],
                    "keywords": [],
                    "key_facts": [],
                    "suggested_tags": [],
                    "suggested_folder": "",
                    "importance": "",
                },
                "alternates": {"suggested_tags": [], "suggested_folder": []},
            }
        ),
        encoding="utf-8",
    )

    status = build_labeling_status(bench_dir=tmp_path)

    assert status["labeled"] == 1
    assert status["remaining"] == 1
    assert status["next_case_id"] == "case_0002"
