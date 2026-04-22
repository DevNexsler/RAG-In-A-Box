import json
from pathlib import Path

from scripts.enrichment_benchmark import build_parser

from core.benchmarking.reporting import write_reports


def _write_run_artifacts(bench_dir: Path, run_id: str = "baseline") -> Path:
    run_dir = bench_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    per_case = [
        {
            "case_id": "case_0001",
            "model": "openai/gpt-4.1-mini",
            "status": "success",
            "latency_ms": 80.0,
            "request": {},
            "response": {"usage": {"total_tokens": 100}},
            "score": {
                "total_score": 0.85,
                "field_scores": {"summary": 1.0, "doc_type": 0.5},
                "weighted_scores": {"summary": 0.01, "doc_type": 0.09},
                "reliability": {
                    "parse_failed": False,
                    "transport_failed": False,
                    "internal_failed": False,
                },
            },
            "raw_output": '{"summary":"Lease renewal request."}',
        },
        {
            "case_id": "case_0002",
            "model": "openai/gpt-4.1-mini",
            "status": "parse_failed",
            "latency_ms": 120.0,
            "request": {},
            "response": {"usage": {"total_tokens": 40}},
            "score": {
                "total_score": 0.0,
                "field_scores": {"summary": 0.0, "doc_type": 0.0},
                "weighted_scores": {"summary": 0.0, "doc_type": 0.0},
                "reliability": {
                    "parse_failed": True,
                    "transport_failed": False,
                    "internal_failed": False,
                },
            },
            "raw_output": "not json",
        },
    ]
    summary = {
        "run_id": run_id,
        "model": "openai/gpt-4.1-mini",
        "case_count": 2,
        "average_total_score": 0.425,
        "parse_failures": 1,
        "request_failures": 0,
        "success_rate": 0.5,
        "parse_failure_rate": 0.5,
        "transport_failure_rate": 0.0,
        "latency_p50": 100.0,
        "latency_p95": 118.0,
        "token_total": 140,
        "token_average": 70.0,
        "field_scores": {"summary": 0.5, "doc_type": 0.25},
    }
    (run_dir / "per_case.jsonl").write_text(
        "\n".join(json.dumps(row) for row in per_case) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run_dir


def test_build_report_writes_leaderboard_and_field_breakdown(tmp_path):
    fixture_run_dir = _write_run_artifacts(tmp_path)

    paths = write_reports(run_dir=fixture_run_dir)

    assert paths["json"].exists()
    assert paths["csv"].exists()
    assert "overall_score" in paths["json"].read_text(encoding="utf-8")
    assert "| model | overall_score |" in paths["markdown"].read_text(encoding="utf-8")


def test_build_report_includes_worst_case_examples_and_field_rows(tmp_path):
    fixture_run_dir = _write_run_artifacts(tmp_path)

    paths = write_reports(run_dir=fixture_run_dir)

    csv_text = paths["csv"].read_text(encoding="utf-8")
    markdown_text = paths["markdown"].read_text(encoding="utf-8")

    assert "field,score" in csv_text
    assert "summary,0.5" in csv_text
    assert "case_0002" in markdown_text
    assert "parse_failed" in markdown_text


def test_build_parser_registers_report_command():
    parser = build_parser()

    args = parser.parse_args(["report", "--run-id", "baseline"])

    assert args.command == "report"
    assert args.bench_dir == ".evals/benchmarks"
    assert args.run_id == "baseline"
