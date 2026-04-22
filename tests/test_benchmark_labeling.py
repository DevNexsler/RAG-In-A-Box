import json
from pathlib import Path

from core.benchmarking.cases import build_labeling_status, write_gold_stub
from core.benchmarking.models import BenchmarkCase


def test_write_gold_stub_creates_all_required_fields(tmp_path):
    case = BenchmarkCase(
        case_id="case_0001",
        prompt="Prompt text",
        baseline_response='{"summary": "baseline"}',
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
    assert "suggested_folder" in data["alternates"]


def test_labeling_status_counts_done_and_remaining():
    fixture_bench_dir = Path("tests/fixtures/benchmarks")

    status = build_labeling_status(bench_dir=fixture_bench_dir)

    assert status["labeled"] == 1
    assert status["remaining"] >= 0
