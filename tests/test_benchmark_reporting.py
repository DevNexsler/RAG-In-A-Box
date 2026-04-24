import csv
import json
from pathlib import Path

import pytest

from scripts.enrichment_benchmark import build_parser

from core.benchmarking.reporting import write_reports


def _write_run_artifacts(
    bench_dir: Path,
    run_id: str = "baseline",
    *,
    include_costs: bool = False,
    score_mode: str = "standard",
    summary_overrides: dict[str, object] | None = None,
    remove_summary_fields: set[str] | None = None,
) -> Path:
    run_dir = bench_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    per_case = [
        {
            "case_id": "case_0001",
            "model": "openai/gpt-4.1-mini",
            "status": "success",
            "latency_ms": 80.0,
            "request": {},
            "response": {
                "usage": {
                    "total_tokens": 100,
                    **({"total_cost": 1.25} if include_costs else {}),
                }
            },
            "score": {
                "total_score": 0.85,
                "field_scores": {"summary": 1.0, "doc_type": 0.5},
                "weighted_scores": {"summary": 0.01, "doc_type": 0.09},
                "reliability": {
                    "parse_failed": False,
                    "transport_failed": False,
                    "internal_failed": False,
                },
                "subscores": (
                    {
                        "extraction_core": 0.9,
                        "filing_taxonomy": 0.7,
                        "summary_quality": 0.8,
                    }
                    if score_mode == "audit"
                    else None
                ),
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
                "subscores": (
                    {
                        "extraction_core": 0.0,
                        "filing_taxonomy": 0.0,
                        "summary_quality": 0.0,
                    }
                    if score_mode == "audit"
                    else None
                ),
            },
            "raw_output": "not json",
        },
    ]
    summary = {
        "run_id": run_id,
        "model": "openai/gpt-4.1-mini",
        "score_mode": score_mode,
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
    if score_mode == "audit":
        summary["subscore_averages"] = {
            "extraction_core": 0.45,
            "filing_taxonomy": 0.35,
            "summary_quality": 0.4,
        }
    if summary_overrides:
        for key, value in summary_overrides.items():
            summary[key] = value
    if remove_summary_fields:
        for key in remove_summary_fields:
            summary.pop(key, None)
    (run_dir / "per_case.jsonl").write_text(
        "\n".join(json.dumps(row) for row in per_case) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run_dir


def test_build_report_writes_leaderboard_and_field_breakdown(tmp_path):
    fixture_run_dir = _write_run_artifacts(tmp_path)

    paths = write_reports(run_dir=fixture_run_dir)
    report = json.loads(paths["json"].read_text(encoding="utf-8"))
    markdown = paths["markdown"].read_text(encoding="utf-8")

    assert paths["json"].exists()
    assert paths["csv"].exists()
    assert report["leaderboard"][0]["overall_score"] == 0.425
    assert report["field_scores"] == {"summary": 0.5, "doc_type": 0.25}
    assert "| model | overall_score | success_rate | parse_failure_rate | latency_p50 | latency_p95 | token_total | token_average |" in markdown


def test_build_report_includes_worst_case_examples_and_field_rows(tmp_path):
    fixture_run_dir = _write_run_artifacts(tmp_path)

    paths = write_reports(run_dir=fixture_run_dir)
    report = json.loads(paths["json"].read_text(encoding="utf-8"))

    with paths["csv"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {"field": "doc_type", "score": "0.25"},
        {"field": "summary", "score": "0.5"},
    ]
    assert [case["case_id"] for case in report["worst_cases"]] == ["case_0002", "case_0001"]
    assert report["worst_cases"][0]["error"] == "parse_failed"


def test_build_report_uses_cost_rows_only_for_cost_average_and_adds_optional_columns(tmp_path):
    fixture_run_dir = _write_run_artifacts(tmp_path, include_costs=True)

    paths = write_reports(run_dir=fixture_run_dir)
    report = json.loads(paths["json"].read_text(encoding="utf-8"))
    markdown = paths["markdown"].read_text(encoding="utf-8")

    assert report["leaderboard"][0]["cost_total"] == 1.25
    assert report["leaderboard"][0]["cost_average"] == 1.25
    assert "| model | overall_score | success_rate | parse_failure_rate | latency_p50 | latency_p95 | token_total | token_average | cost_total | cost_average |" in markdown


def test_build_report_fails_on_missing_required_summary_fields(tmp_path):
    fixture_run_dir = _write_run_artifacts(tmp_path, remove_summary_fields={"model"})

    with pytest.raises(ValueError, match="summary.json missing required field: model"):
        write_reports(run_dir=fixture_run_dir)


def test_build_report_allows_nullable_latency_summary_fields(tmp_path):
    fixture_run_dir = _write_run_artifacts(
        tmp_path,
        summary_overrides={"latency_p50": None, "latency_p95": None},
    )

    paths = write_reports(run_dir=fixture_run_dir)
    report = json.loads(paths["json"].read_text(encoding="utf-8"))

    assert report["leaderboard"][0]["latency_p50"] is None
    assert report["leaderboard"][0]["latency_p95"] is None


def test_build_parser_registers_report_command():
    parser = build_parser()

    args = parser.parse_args(["report", "--run-id", "baseline"])

    assert args.command == "report"
    assert args.bench_dir == ".evals/benchmarks"
    assert args.run_id == "baseline"


def test_build_report_includes_audit_subscores_in_json_csv_and_markdown(tmp_path):
    fixture_run_dir = _write_run_artifacts(tmp_path, run_id="audit-baseline", score_mode="audit")

    paths = write_reports(run_dir=fixture_run_dir)
    report = json.loads(paths["json"].read_text(encoding="utf-8"))
    markdown = paths["markdown"].read_text(encoding="utf-8")

    assert report["summary"]["score_mode"] == "audit"
    assert report["subscore_averages"] == {
        "extraction_core": 0.45,
        "filing_taxonomy": 0.35,
        "summary_quality": 0.4,
    }
    assert report["leaderboard"][0]["extraction_core"] == 0.45
    assert "| model | overall_score | extraction_core | filing_taxonomy | summary_quality | success_rate | parse_failure_rate | latency_p50 | latency_p95 | token_total | token_average |" in markdown
    assert paths["subscores_csv"].exists()

    with paths["subscores_csv"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {"subscore": "extraction_core", "score": "0.45"},
        {"subscore": "filing_taxonomy", "score": "0.35"},
        {"subscore": "summary_quality", "score": "0.4"},
    ]


def test_build_parser_registers_audit_commands():
    parser = build_parser()

    prepare_args = parser.parse_args(["prepare-audit"])
    show_args = parser.parse_args(["show-audit-case", "--case-id", "case_0001"])
    status_args = parser.parse_args(["audit-status"])
    run_args = parser.parse_args(["audit-run", "--model", "openai/gpt-4.1-mini", "--run-id", "audit"])
    report_args = parser.parse_args(["audit-report", "--run-id", "audit"])

    assert prepare_args.command == "prepare-audit"
    assert prepare_args.source_bench_dir == ".evals/benchmarks"
    assert prepare_args.bench_dir == ".evals/benchmarks/audit"
    assert show_args.command == "show-audit-case"
    assert show_args.bench_dir == ".evals/benchmarks/audit"
    assert status_args.command == "audit-status"
    assert status_args.bench_dir == ".evals/benchmarks/audit"
    assert run_args.command == "audit-run"
    assert run_args.bench_dir == ".evals/benchmarks/audit"
    assert report_args.command == "audit-report"
    assert report_args.bench_dir == ".evals/benchmarks/audit"
