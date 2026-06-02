from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from core.benchmarking.cases import load_case, resolve_bench_path
from core.benchmarking.scoring import score_failed_case
from core.benchmarking.tasks import BenchmarkTask, get_task
from providers.llm.openrouter_llm import OpenRouterGenerator


@dataclass(frozen=True)
class BenchmarkRunResult:
    run_dir: Path
    per_case: list[dict[str, Any]]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["run_dir"] = str(self.run_dir)
        return payload


def run_benchmark(
    *,
    bench_dir: str | Path,
    model: str,
    run_id: str,
    max_cases: int | None = None,
    replay_client: Any | None = None,
    score_mode: str | None = None,
    task: str = "enrichment",
    suite: str = "standard",
) -> BenchmarkRunResult:
    bench_task = get_task(task)
    bench_path = resolve_bench_path(bench_dir=bench_dir, task=task, suite=suite)
    run_dir = bench_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if score_mode is None:
        score_mode = bench_task.default_score_mode
    _validate_score_mode(score_mode, score_modes=bench_task.score_modes)
    client = replay_client or OpenRouterGenerator(model=model)
    case_ids = sorted(path.stem for path in (bench_path / "cases").glob("case_*.json"))
    if max_cases is not None:
        case_ids = case_ids[:max_cases]

    results: list[dict[str, Any]] = []
    for case_id in case_ids:
        case = load_case(bench_dir=bench_path, case_id=case_id)
        gold = _load_gold_record(bench_path=bench_path, case_id=case_id)
        result = _run_case(
            case=case,
            gold=gold,
            client=client,
            model=model,
            score_mode=score_mode,
            bench_task=bench_task,
        )
        results.append(result)

    per_case_path = run_dir / "per_case.jsonl"
    per_case_path.write_text(
        "\n".join(json.dumps(item) for item in results) + ("\n" if results else ""),
        encoding="utf-8",
    )

    summary = _build_summary(
        results=results,
        model=model,
        run_id=run_id,
        score_mode=score_mode,
        task=task,
        suite=suite,
        bench_path=bench_path,
    )
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return BenchmarkRunResult(run_dir=run_dir, per_case=results, summary=summary)


def _run_case(
    *,
    case: Any,
    gold: dict[str, Any],
    client: Any,
    model: str,
    score_mode: str,
    bench_task: BenchmarkTask,
) -> dict[str, Any]:
    replay: dict[str, Any] | None = None
    try:
        replay = client.generate_with_metadata(case.prompt, max_tokens=512)
        normalized_output, score = bench_task.score_raw_output(replay["content"], gold, score_mode)
        return {
            "case_id": case.case_id,
            "model": model,
            "status": "success",
            "raw_output": replay["content"],
            "normalized_output": normalized_output,
            "request": replay.get("request", {}),
            "response": replay.get("response", {}),
            "latency_ms": replay.get("latency_ms"),
            "score": score.to_dict(),
        }
    except json.JSONDecodeError as exc:
        score = score_failed_case(error="json_parse_error")
        return {
            "case_id": case.case_id,
            "model": model,
            "status": "parse_failed",
            "raw_output": replay.get("content", "") if replay else "",
            "normalized_output": None,
            "request": replay.get("request", {}) if replay else {},
            "response": replay.get("response", {}) if replay else {},
            "latency_ms": replay.get("latency_ms") if replay else None,
            "score": score.to_dict(),
            "error": str(exc),
        }
    except ValueError as exc:
        score = score_failed_case(error="internal_error")
        return {
            "case_id": case.case_id,
            "model": model,
            "status": "internal_error",
            "raw_output": replay.get("content", "") if replay else "",
            "normalized_output": None,
            "request": replay.get("request", {}) if replay else {},
            "response": replay.get("response", {}) if replay else {},
            "latency_ms": replay.get("latency_ms") if replay else None,
            "score": score.to_dict(),
            "error": str(exc),
        }
    except httpx.TimeoutException as exc:
        score = score_failed_case(error="timeout")
        return {
            "case_id": case.case_id,
            "model": model,
            "status": "transport_failed",
            "raw_output": "",
            "normalized_output": None,
            "request": {},
            "response": {},
            "latency_ms": None,
            "score": score.to_dict(),
            "error": str(exc),
        }
    except httpx.HTTPError as exc:
        score = score_failed_case(error="http_error")
        return {
            "case_id": case.case_id,
            "model": model,
            "status": "transport_failed",
            "raw_output": "",
            "normalized_output": None,
            "request": {},
            "response": {},
            "latency_ms": None,
            "score": score.to_dict(),
            "error": str(exc),
        }


def _load_gold_record(*, bench_path: Path, case_id: str) -> dict[str, Any]:
    path = bench_path / "gold" / f"{case_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _build_summary(
    *,
    results: list[dict[str, Any]],
    model: str,
    run_id: str,
    score_mode: str,
    task: str,
    suite: str,
    bench_path: Path,
) -> dict[str, Any]:
    case_count = len(results)
    parse_failures = sum(1 for item in results if item["score"]["reliability"].get("parse_failed"))
    request_failures = sum(1 for item in results if item.get("status") == "transport_failed")
    successes = sum(1 for item in results if item.get("status") == "success")
    mean_total = (
        round(sum(item["score"]["total_score"] for item in results) / case_count, 6)
        if case_count
        else 0.0
    )
    latencies = sorted(item["latency_ms"] for item in results if item.get("latency_ms") is not None)
    token_total = sum(_extract_total_tokens(item) for item in results)

    field_scores: dict[str, float] = {}
    if results:
        fields = list(results[0]["score"]["field_scores"])
        for field in fields:
            field_scores[field] = round(
                sum(item["score"]["field_scores"][field] for item in results) / case_count,
                6,
            )

    summary = {
        "run_id": run_id,
        "task": task,
        "suite": suite,
        "model": model,
        "score_mode": score_mode,
        "case_count": case_count,
        "average_total_score": mean_total,
        "parse_failures": parse_failures,
        "request_failures": request_failures,
        "success_rate": round(successes / case_count, 6) if case_count else 0.0,
        "parse_failure_rate": round(parse_failures / case_count, 6) if case_count else 0.0,
        "transport_failure_rate": round(request_failures / case_count, 6) if case_count else 0.0,
        "latency_p50": _percentile(latencies, 0.50),
        "latency_p95": _percentile(latencies, 0.95),
        "token_total": token_total,
        "token_average": round(token_total / case_count, 6) if case_count else 0.0,
        "field_scores": field_scores,
    }
    if score_mode == "audit":
        summary["subscore_averages"] = _average_subscores(results=results, case_count=case_count)
    hard_case_breakdown = _build_hard_case_breakdown(
        bench_path=bench_path,
        results=results,
        suite=suite,
    )
    if hard_case_breakdown is not None:
        summary["hard_case_breakdown"] = hard_case_breakdown
    provider_failure_breakdown = _build_provider_failure_breakdown(bench_path=bench_path)
    if provider_failure_breakdown is not None:
        summary["provider_failure_breakdown"] = provider_failure_breakdown
    return summary


def _build_hard_case_breakdown(
    *,
    bench_path: Path,
    results: list[dict[str, Any]],
    suite: str,
) -> dict[str, dict[str, float | int]] | None:
    if suite == "standard":
        return None

    manifest_path = bench_path / "cases" / "manifest.json"
    if not manifest_path.exists():
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest.get("cases", [])
    if not isinstance(cases, list):
        return {}

    results_by_case = {str(result.get("case_id")): result for result in results}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        if not isinstance(case, dict):
            continue
        result = results_by_case.get(str(case.get("case_id")))
        if result is None:
            continue
        flags = case.get("flags", [])
        if not isinstance(flags, list):
            continue
        for flag in flags:
            grouped.setdefault(str(flag), []).append(result)

    return {
        flag: _summarize_result_group(group_results)
        for flag, group_results in sorted(grouped.items())
    }


def _summarize_result_group(results: list[dict[str, Any]]) -> dict[str, float | int]:
    case_count = len(results)
    parse_failures = sum(
        1 for item in results if item.get("score", {}).get("reliability", {}).get("parse_failed")
    )
    transport_failures = sum(1 for item in results if item.get("status") == "transport_failed")
    successes = sum(1 for item in results if item.get("status") == "success")
    return {
        "case_count": case_count,
        "average_total_score": round(
            sum(item.get("score", {}).get("total_score", 0.0) for item in results) / case_count,
            6,
        )
        if case_count
        else 0.0,
        "success_rate": round(successes / case_count, 6) if case_count else 0.0,
        "parse_failure_rate": round(parse_failures / case_count, 6) if case_count else 0.0,
        "transport_failure_rate": round(transport_failures / case_count, 6) if case_count else 0.0,
    }


def _build_provider_failure_breakdown(*, bench_path: Path) -> dict[str, int] | None:
    failures_path = bench_path / "provider_failures.jsonl"
    if not failures_path.exists():
        return None

    counts: dict[str, int] = {}
    for line in failures_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            failure = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(failure, dict):
            continue
        key = failure.get("failure_status_code") or failure.get("failure_type") or "unknown"
        key = str(key)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _average_subscores(*, results: list[dict[str, Any]], case_count: int) -> dict[str, float]:
    if not results or case_count == 0:
        return {}
    keys = sorted(
        {
            key
            for item in results
            for key in (item.get("score", {}).get("subscores") or {})
        }
    )
    return {
        key: round(
            sum((item.get("score", {}).get("subscores") or {}).get(key, 0.0) for item in results)
            / case_count,
            6,
        )
        for key in keys
    }


def _validate_score_mode(score_mode: str, *, score_modes: frozenset[str]) -> None:
    if score_mode not in score_modes:
        raise ValueError(f"unsupported benchmark score_mode: {score_mode}")


def _extract_total_tokens(result: dict[str, Any]) -> int:
    usage = result.get("response", {}).get("usage", {})
    value = usage.get("total_tokens")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    fraction = index - lower
    return round(values[lower] + (values[upper] - values[lower]) * fraction, 6)
