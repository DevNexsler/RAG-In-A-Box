from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_reports(*, run_dir: str | Path) -> dict[str, Path]:
    run_path = Path(run_dir)
    summary = _load_json(run_path / "summary.json")
    per_case = _load_jsonl(run_path / "per_case.jsonl")
    report = _build_report(summary=summary, per_case=per_case)

    json_path = run_path / "report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_path = run_path / "field_scores.csv"
    _write_field_scores_csv(path=csv_path, field_scores=report["field_scores"])

    markdown_path = run_path / "leaderboard.md"
    markdown_path.write_text(_build_markdown(report=report), encoding="utf-8")

    return {
        "json": json_path,
        "csv": csv_path,
        "markdown": markdown_path,
    }


def _build_report(*, summary: dict[str, Any], per_case: list[dict[str, Any]]) -> dict[str, Any]:
    leaderboard_row = {
        "model": summary.get("model", ""),
        "run_id": summary.get("run_id", ""),
        "overall_score": summary.get("average_total_score", 0.0),
        "success_rate": summary.get("success_rate", 0.0),
        "parse_failure_rate": summary.get("parse_failure_rate", 0.0),
        "latency_p50": summary.get("latency_p50"),
        "latency_p95": summary.get("latency_p95"),
    }
    token_total = summary.get("token_total")
    token_average = summary.get("token_average")
    if token_total is not None:
        leaderboard_row["token_total"] = token_total
    if token_average is not None:
        leaderboard_row["token_average"] = token_average

    costs = _extract_costs(per_case)
    if costs["cost_total"] is not None:
        leaderboard_row["cost_total"] = costs["cost_total"]
    if costs["cost_average"] is not None:
        leaderboard_row["cost_average"] = costs["cost_average"]

    return {
        "run_id": summary.get("run_id", ""),
        "model": summary.get("model", ""),
        "overall_score": leaderboard_row["overall_score"],
        "summary": summary,
        "leaderboard": [leaderboard_row],
        "field_scores": summary.get("field_scores", {}),
        "worst_cases": _worst_cases(per_case),
    }


def _write_field_scores_csv(*, path: Path, field_scores: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["field", "score"])
        for field, score in sorted(field_scores.items()):
            writer.writerow([field, score])


def _build_markdown(*, report: dict[str, Any]) -> str:
    leaderboard = report["leaderboard"]
    headers = [
        "model",
        "overall_score",
        "success_rate",
        "parse_failure_rate",
        "latency_p50",
        "latency_p95",
    ]
    optional_headers = ["token_total", "token_average", "cost_total", "cost_average"]
    for header in optional_headers:
        if any(header in row for row in leaderboard):
            headers.append(header)

    lines = [
        "# Benchmark Report",
        "",
        f"Run ID: `{report['run_id']}`",
        "",
        _markdown_table(headers=headers, rows=leaderboard),
        "",
        "## Field Scores",
        "",
        "| field | score |",
        "| --- | --- |",
    ]
    for field, score in sorted(report["field_scores"].items()):
        lines.append(f"| {field} | {score} |")

    lines.extend(
        [
            "",
            "## Worst Cases",
            "",
            "| case_id | status | total_score | latency_ms | error |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for case in report["worst_cases"]:
        lines.append(
            "| {case_id} | {status} | {total_score} | {latency_ms} | {error} |".format(
                case_id=case["case_id"],
                status=case["status"],
                total_score=case["total_score"],
                latency_ms=case["latency_ms"],
                error=case["error"],
            )
        )

    return "\n".join(lines) + "\n"


def _markdown_table(*, headers: list[str], rows: list[dict[str, Any]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    divider_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = [
        "| " + " | ".join(_format_cell(row.get(header)) for header in headers) + " |"
        for row in rows
    ]
    return "\n".join([header_line, divider_line, *body_lines])


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _worst_cases(per_case: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    ranked = sorted(
        per_case,
        key=lambda item: (item.get("score", {}).get("total_score", 0.0), item.get("case_id", "")),
    )
    return [
        {
            "case_id": item.get("case_id", ""),
            "status": item.get("status", ""),
            "total_score": item.get("score", {}).get("total_score", 0.0),
            "latency_ms": item.get("latency_ms"),
            "error": _case_error(item),
        }
        for item in ranked[:limit]
    ]


def _case_error(item: dict[str, Any]) -> str:
    if item.get("error"):
        return str(item["error"])
    reliability = item.get("score", {}).get("reliability", {})
    for key in ("parse_failed", "transport_failed", "internal_failed"):
        if reliability.get(key):
            return key
    return ""


def _extract_costs(per_case: list[dict[str, Any]]) -> dict[str, float | None]:
    values: list[float] = []
    for item in per_case:
        usage = item.get("response", {}).get("usage", {})
        for key in ("total_cost", "cost", "cost_usd"):
            value = usage.get(key)
            if value is None:
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                pass
            break
    if not values:
        return {"cost_total": None, "cost_average": None}
    total = round(sum(values), 6)
    average = round(total / len(per_case), 6) if per_case else 0.0
    return {"cost_total": total, "cost_average": average}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
