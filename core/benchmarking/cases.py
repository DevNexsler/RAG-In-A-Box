from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from core.benchmarking.models import (
    BenchmarkCase,
    GoldRecord,
    LabelingStatus,
    PreparedCasesResult,
    TraceRow,
)

_TITLE_RE = re.compile(r"^Document title:\s*(.+)$", re.MULTILINE)
_SOURCE_TYPE_RE = re.compile(r"^Document type:\s*(.+)$", re.MULTILINE)
_CANONICAL_FIELD_DEFAULTS: dict[str, Any] = {
    "summary": "",
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
}
_ALTERNATE_FIELDS = ("suggested_tags", "suggested_folder")


def load_trace_rows(trace_path: str | Path) -> list[TraceRow]:
    rows: list[TraceRow] = []
    for path in _iter_trace_paths(Path(trace_path)):
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            prompt = _extract_prompt(payload)
            baseline_response = _extract_baseline_response(payload)
            if not prompt or not baseline_response:
                continue
            title = _extract_prompt_field(_TITLE_RE, prompt)
            source_type = _extract_prompt_field(_SOURCE_TYPE_RE, prompt)
            category = _infer_category(title=title, source_type=source_type, prompt=prompt)
            difficulty = _infer_difficulty(title=title, prompt=prompt)
            rows.append(
                TraceRow(
                    prompt=prompt,
                    baseline_response=baseline_response,
                    title=title,
                    source_type=source_type,
                    category=category,
                    difficulty=difficulty,
                    trace_file=path.name,
                    trace_line=line_number,
                )
            )
    return rows


def prepare_cases(
    *,
    trace_dir: str | Path,
    out_dir: str | Path,
    limit: int = 100,
    seed: int = 42,
) -> PreparedCasesResult:
    trace_rows = load_trace_rows(trace_dir)
    selected_rows = _select_rows(trace_rows=trace_rows, limit=limit, seed=seed)

    out_path = Path(out_dir)
    cases_dir = out_path / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_case_files(cases_dir)

    cases: list[BenchmarkCase] = []
    for index, row in enumerate(selected_rows, start=1):
        case = BenchmarkCase(
            case_id=f"case_{index:04d}",
            prompt=row.prompt,
            baseline_response=row.baseline_response,
            title=row.title,
            source_type=row.source_type,
            category=row.category,
            difficulty=row.difficulty,
            trace_file=row.trace_file,
            trace_line=row.trace_line,
        )
        cases.append(case)
        (cases_dir / f"{case.case_id}.json").write_text(
            json.dumps(asdict(case), indent=2),
            encoding="utf-8",
        )

    manifest = {
        "selected_count": len(cases),
        "limit": limit,
        "seed": seed,
        "cases": [asdict(case) for case in cases],
    }
    manifest_path = cases_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return PreparedCasesResult(
        selected_count=len(cases),
        cases=cases,
        manifest_path=manifest_path,
    )


def load_case(*, bench_dir: str | Path, case_id: str) -> BenchmarkCase:
    payload = json.loads(_case_path(bench_dir=bench_dir, case_id=case_id).read_text(encoding="utf-8"))
    return BenchmarkCase(**payload)


def write_gold_stub(case: BenchmarkCase, *, bench_dir: str | Path) -> Path:
    path = _gold_path(bench_dir=bench_dir, case_id=case.case_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        _validate_canonical_fields(payload.get("canonical"))
        return path

    gold_record = GoldRecord(
        case_id=case.case_id,
        canonical=_build_canonical_stub(),
        alternates=_build_alternates_from_baseline(case.baseline_response),
    )
    path.write_text(json.dumps(asdict(gold_record), indent=2), encoding="utf-8")
    return path


def build_labeling_status(*, bench_dir: str | Path) -> dict[str, Any]:
    bench_path = Path(bench_dir)
    case_ids = sorted(path.stem for path in (bench_path / "cases").glob("case_*.json"))
    labeled_case_ids = [case_id for case_id in case_ids if _is_labeled_gold(bench_dir=bench_path, case_id=case_id)]

    next_case_id = next((case_id for case_id in case_ids if case_id not in set(labeled_case_ids)), None)
    status = LabelingStatus(
        total_cases=len(case_ids),
        labeled=len(labeled_case_ids),
        remaining=len(case_ids) - len(labeled_case_ids),
        next_case_id=next_case_id,
    )

    status_path = bench_path / "status" / "labeling_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(asdict(status), indent=2), encoding="utf-8")
    return asdict(status)


def _iter_trace_paths(path: Path) -> Iterable[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.jsonl"))


def _remove_stale_case_files(cases_dir: Path) -> None:
    for path in cases_dir.glob("case_*.json"):
        path.unlink()


def _extract_prompt(payload: dict[str, Any]) -> str:
    messages = payload.get("request", {}).get("payload", {}).get("messages", [])
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", "")).strip()
    return ""


def _extract_baseline_response(payload: dict[str, Any]) -> str:
    choices = payload.get("response", {}).get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    return str(message.get("content", "")).strip()


def _extract_prompt_field(pattern: re.Pattern[str], prompt: str) -> str:
    match = pattern.search(prompt)
    if match is None:
        return ""
    return match.group(1).strip()


def _infer_category(*, title: str, source_type: str, prompt: str) -> str:
    text = " ".join((title, source_type, prompt)).lower()
    if any(token in text for token in ("lease", "tenant", "property", "housing", "inspection")):
        return "housing"
    if any(token in text for token in ("invoice", "payment", "accounting", "vendor")):
        return "finance"
    if source_type.lower() in {"email", "note"}:
        return "communications"
    return "general"


def _infer_difficulty(*, title: str, prompt: str) -> str:
    text = " ".join((title, prompt)).lower()
    if any(token in text for token in ("smoke", "synthetic", "pipeline validation", "test record")):
        return "smoke"
    if len(prompt) > 1800:
        return "hard"
    if len(prompt) > 900:
        return "medium"
    return "easy"


def _select_rows(*, trace_rows: list[TraceRow], limit: int, seed: int) -> list[TraceRow]:
    eligible_rows = [row for row in trace_rows if row.difficulty != "smoke"]
    if limit <= 0 or not eligible_rows:
        return []

    rng = random.Random(seed)
    grouped: dict[str, list[TraceRow]] = defaultdict(list)
    for row in eligible_rows:
        grouped[row.category].append(row)

    for rows in grouped.values():
        rng.shuffle(rows)

    selected: list[TraceRow] = []
    categories = sorted(grouped)
    while len(selected) < limit:
        progressed = False
        for category in categories:
            bucket = grouped[category]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            progressed = True
            if len(selected) == limit:
                break
        if not progressed:
            break
    return selected


def _case_path(*, bench_dir: str | Path, case_id: str) -> Path:
    return Path(bench_dir) / "cases" / f"{case_id}.json"


def _gold_path(*, bench_dir: str | Path, case_id: str) -> Path:
    return Path(bench_dir) / "gold" / f"{case_id}.json"


def _build_canonical_stub() -> dict[str, Any]:
    stub: dict[str, Any] = {}
    for field, default in _CANONICAL_FIELD_DEFAULTS.items():
        stub[field] = list(default) if isinstance(default, list) else default
    return stub


def _build_alternates_from_baseline(baseline_response: str) -> dict[str, list[str]]:
    alternates = {field: [] for field in _ALTERNATE_FIELDS}
    try:
        payload = json.loads(baseline_response)
    except json.JSONDecodeError:
        return alternates
    if not isinstance(payload, dict):
        return alternates

    tags = payload.get("suggested_tags")
    if isinstance(tags, list):
        alternates["suggested_tags"] = [str(tag).strip() for tag in tags if str(tag).strip()]
    elif isinstance(tags, str) and tags.strip():
        alternates["suggested_tags"] = [tags.strip()]

    folder = payload.get("suggested_folder")
    if isinstance(folder, str) and folder.strip():
        alternates["suggested_folder"] = [folder.strip()]

    return alternates


def _load_gold_record(*, bench_dir: str | Path, case_id: str) -> dict[str, Any] | None:
    path = _gold_path(bench_dir=bench_dir, case_id=case_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_canonical_fields(canonical: Any) -> None:
    if not isinstance(canonical, dict):
        raise ValueError("gold record canonical payload must be an object")
    missing = [field for field in _CANONICAL_FIELD_DEFAULTS if field not in canonical]
    if missing:
        raise ValueError(f"gold record canonical payload missing fields: {', '.join(missing)}")


def _is_labeled_gold(*, bench_dir: str | Path, case_id: str) -> bool:
    payload = _load_gold_record(bench_dir=bench_dir, case_id=case_id)
    if payload is None:
        return False
    try:
        _validate_canonical_fields(payload.get("canonical"))
    except ValueError:
        return False

    canonical = payload["canonical"]
    if str(canonical.get("summary", "")).strip():
        return True
    return any(bool(canonical.get(field)) for field in _CANONICAL_FIELD_DEFAULTS if field != "summary")
