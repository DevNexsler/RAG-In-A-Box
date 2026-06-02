from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from core.benchmarking.cases import (
    _infer_category,
    _infer_difficulty,
    _remove_stale_case_files,
    _remove_stale_gold_files,
    resolve_bench_path,
    write_gold_stub,
)
from core.benchmarking.models import (
    BenchmarkCase,
    HardCaseCandidate,
    HardCaseSelection,
    PreparedCasesResult,
    TraceMetadata,
)

LONG_PROMPT_CHARS = 25_000
HUGE_PROMPT_CHARS = 250_000
SLOW_SUCCESS_MS = 10_000
VERY_SLOW_SUCCESS_MS = 25_000

_TITLE_RE = re.compile(r"^Document title:\s*(?P<value>.+?)\s*$", re.MULTILINE)
_TYPE_RE = re.compile(r"^Document type:\s*(?P<value>.+?)\s*$", re.MULTILINE)
_TEXT_RE = re.compile(
    r"^Document text:\s*\n(?P<value>.*?)(?:\n[A-Z][A-Z0-9 _*-]+\n|\Z)",
    re.MULTILINE | re.DOTALL,
)
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_BUSINESS_CRITICAL_RE = re.compile(
    r"\b(payment|billing|rent|collection|cleared|failed|invoice|charge)\b",
    re.IGNORECASE,
)

_PROMPT_FEATURES_BY_HASH: dict[str, frozenset[str]] = {}


def load_trace_metadata(trace_path: str | Path) -> list[TraceMetadata]:
    path = Path(trace_path)
    rows: list[TraceMetadata] = []

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(raw, dict):
                continue

            prompt = _extract_user_prompt(raw)
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()

            _PROMPT_FEATURES_BY_HASH[prompt_hash] = frozenset(_extract_prompt_features(prompt))

            error = raw.get("error") if isinstance(raw.get("error"), dict) else {}
            success = bool(raw.get("success"))
            rows.append(
                TraceMetadata(
                    trace_file=path.name,
                    trace_line=line_number,
                    timestamp=str(raw.get("ts") or ""),
                    provider=str(raw.get("provider") or ""),
                    model=str(raw.get("model") or ""),
                    success=success,
                    latency_ms=_coerce_float(raw.get("latency_ms")),
                    title=_extract_regex(_TITLE_RE, prompt),
                    source_type=_extract_regex(_TYPE_RE, prompt),
                    prompt_length=len(prompt),
                    text_length=len(_extract_document_text(prompt)),
                    prompt_hash=prompt_hash,
                    response_looks_parseable=_response_looks_parseable(raw),
                    failure_type=str(error.get("type")) if not success and error.get("type") else None,
                    failure_status_code=_coerce_int(error.get("status_code")) if not success else None,
                )
            )

    return rows


def score_hard_flags(row: TraceMetadata) -> HardCaseCandidate:
    flags: list[str] = []
    features = _PROMPT_FEATURES_BY_HASH.get(row.prompt_hash, frozenset())

    if row.prompt_length >= LONG_PROMPT_CHARS:
        flags.append("long_prompt")
    if row.prompt_length >= HUGE_PROMPT_CHARS:
        flags.append("huge_prompt")
    if "nearby_context" in features:
        flags.append("nearby_context")
    if "taxonomy_bloat" in features:
        flags.append("taxonomy_bloat")
    if "link_noise" in features:
        flags.append("link_noise")
    if "business_critical" in features or _BUSINESS_CRITICAL_RE.search(row.title):
        flags.append("business_critical")
    if row.success and row.latency_ms is not None and row.latency_ms >= SLOW_SUCCESS_MS:
        flags.append("slow_success")
    if row.success and row.latency_ms is not None and row.latency_ms >= VERY_SLOW_SUCCESS_MS:
        flags.append("very_slow_success")
    if row.success and not row.response_looks_parseable:
        flags.append("parse_suspect")

    return HardCaseCandidate(trace=row, flags=tuple(flags), hard_score=len(flags))


def select_hard_cases(rows: list[TraceMetadata], *, limit: int = 50) -> HardCaseSelection:
    provider_failure_cases: list[TraceMetadata] = []
    candidates: list[HardCaseCandidate] = []
    seen_hashes: set[str] = set()
    seen_failure_keys: set[tuple[str, str, int]] = set()

    for row in rows:
        if not row.success and row.failure_type and row.failure_status_code is not None:
            failure_key = (row.prompt_hash, row.failure_type, row.failure_status_code)
            if failure_key not in seen_failure_keys:
                seen_failure_keys.add(failure_key)
                provider_failure_cases.append(row)
            continue

        if row.prompt_hash in seen_hashes:
            continue
        seen_hashes.add(row.prompt_hash)

        candidate = score_hard_flags(row)
        if candidate.hard_score > 0:
            candidates.append(candidate)

    candidates.sort(key=lambda item: (-item.hard_score, item.trace.trace_file, item.trace.trace_line))
    return HardCaseSelection(
        hard_cases=candidates[:limit],
        provider_failure_cases=provider_failure_cases[:limit],
    )


def materialize_hard_suite(
    *,
    trace_dir: str | Path,
    out_dir: str | Path,
    task: str = "enrichment",
    suite: str = "hard",
    limit: int = 50,
) -> PreparedCasesResult:
    trace_paths = _iter_jsonl_paths(Path(trace_dir))
    metadata_rows = _load_all_trace_metadata(trace_paths)
    selection = select_hard_cases(metadata_rows, limit=limit)
    full_rows = _load_full_trace_rows(trace_paths)

    suite_dir = resolve_bench_path(bench_dir=out_dir, task=task, suite=suite)
    cases_dir = suite_dir / "cases"
    gold_dir = suite_dir / "gold"
    cases_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_case_files(cases_dir)
    _remove_stale_gold_files(gold_dir)

    cases: list[BenchmarkCase] = []
    manifest_cases: list[dict[str, Any]] = []
    selection_flags: set[str] = set()
    case_index = 1
    for candidate in selection.hard_cases:
        if not candidate.trace.response_looks_parseable:
            continue
        full_row = full_rows.get((candidate.trace.trace_file, candidate.trace.trace_line))
        if full_row is None:
            continue

        case = BenchmarkCase(
            case_id=f"case_{case_index:04d}",
            prompt=full_row["prompt"],
            baseline_response=full_row["baseline_response"],
            title=candidate.trace.title,
            source_type=candidate.trace.source_type,
            category=_infer_category(
                title=candidate.trace.title,
                source_type=candidate.trace.source_type,
                prompt=full_row["prompt"],
            ),
            difficulty=_infer_difficulty(title=candidate.trace.title, prompt=full_row["prompt"]),
            trace_file=candidate.trace.trace_file,
            trace_line=candidate.trace.trace_line,
        )
        cases.append(case)
        selection_flags.update(candidate.flags)
        (cases_dir / f"{case.case_id}.json").write_text(
            json.dumps(asdict(case), indent=2),
            encoding="utf-8",
        )
        write_gold_stub(case, bench_dir=suite_dir)
        manifest_cases.append(_build_manifest_case(case, candidate))
        case_index += 1

    _write_provider_failures(suite_dir / "provider_failures.jsonl", selection.provider_failure_cases)

    manifest = {
        "task": task,
        "suite": suite,
        "selected_count": len(cases),
        "limit": limit,
        "selection_flags": sorted(selection_flags),
        "provider_failure_count": len(selection.provider_failure_cases),
        "cases": manifest_cases,
    }
    manifest_path = cases_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return PreparedCasesResult(
        selected_count=len(cases),
        cases=cases,
        manifest_path=manifest_path,
    )


def _load_all_trace_metadata(trace_paths: Iterable[Path]) -> list[TraceMetadata]:
    rows: list[TraceMetadata] = []
    for path in trace_paths:
        rows.extend(load_trace_metadata(path))
    return rows


def _load_full_trace_rows(trace_paths: Iterable[Path]) -> dict[tuple[str, int], dict[str, str]]:
    rows: dict[tuple[str, int], dict[str, str]] = {}
    for path in trace_paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(raw, dict):
                    continue
                prompt = _extract_user_prompt(raw)
                baseline_response = _extract_response_content(raw)
                if not prompt or not baseline_response:
                    continue
                rows[(path.name, line_number)] = {
                    "prompt": prompt,
                    "baseline_response": baseline_response,
                }
    return rows


def _iter_jsonl_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.jsonl"))


def _build_manifest_case(case: BenchmarkCase, candidate: HardCaseCandidate) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "trace_file": case.trace_file,
        "trace_line": case.trace_line,
        "title": case.title,
        "source_type": case.source_type,
        "category": case.category,
        "difficulty": case.difficulty,
        "flags": list(candidate.flags),
        "hard_score": candidate.hard_score,
    }


def _write_provider_failures(path: Path, rows: list[TraceMetadata]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = {
                "trace_file": row.trace_file,
                "trace_line": row.trace_line,
                "timestamp": row.timestamp,
                "provider": row.provider,
                "model": row.model,
                "title": row.title,
                "source_type": row.source_type,
                "prompt_hash": row.prompt_hash,
                "failure_type": row.failure_type,
                "failure_status_code": row.failure_status_code,
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _extract_user_prompt(raw: dict[str, Any]) -> str:
    request = raw.get("request")
    if not isinstance(request, dict):
        return ""
    payload = request.get("payload", {})
    if not isinstance(payload, dict):
        return ""
    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        return ""

    contents: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            contents.append(content)

    return "\n".join(contents)


def _extract_prompt_features(prompt: str) -> set[str]:
    features: set[str] = set()
    if "NEARBY SAME-CHANNEL CONTEXT CANDIDATES" in prompt or "BEFORE MESSAGES" in prompt:
        features.add("nearby_context")
    if "## Available Tags" in prompt or "## Available Folders" in prompt:
        features.add("taxonomy_bloat")
    if _URL_RE.search(prompt):
        features.add("link_noise")
    if _BUSINESS_CRITICAL_RE.search(prompt):
        features.add("business_critical")
    return features


def _extract_regex(pattern: re.Pattern[str], prompt: str) -> str:
    match = pattern.search(prompt)
    return match.group("value").strip() if match else ""


def _extract_document_text(prompt: str) -> str:
    match = _TEXT_RE.search(prompt)
    return match.group("value").strip() if match else ""


def _response_looks_parseable(raw: dict[str, Any]) -> bool:
    content = _extract_response_content(raw)
    if not content:
        return False
    try:
        json.loads(content)
    except json.JSONDecodeError:
        return False
    return True


def _extract_response_content(raw: dict[str, Any]) -> str:
    response = raw.get("response")
    if not isinstance(response, dict):
        return ""
    choices = response.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
