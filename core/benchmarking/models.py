from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceRow:
    prompt: str
    baseline_response: str
    title: str
    source_type: str
    category: str
    difficulty: str
    trace_file: str
    trace_line: int


@dataclass(frozen=True)
class TraceMetadata:
    trace_file: str
    trace_line: int
    timestamp: str
    provider: str
    model: str
    success: bool
    latency_ms: float | None
    title: str
    source_type: str
    prompt_length: int
    text_length: int
    prompt_hash: str
    response_looks_parseable: bool
    failure_type: str | None = None
    failure_status_code: int | None = None
    prompt_excerpt: str = ""


@dataclass(frozen=True)
class HardCaseCandidate:
    trace: TraceMetadata
    flags: tuple[str, ...]
    hard_score: int


@dataclass(frozen=True)
class HardCaseSelection:
    hard_cases: list[HardCaseCandidate]
    provider_failure_cases: list[TraceMetadata]


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    prompt: str
    baseline_response: str
    title: str
    source_type: str
    category: str
    difficulty: str
    trace_file: str
    trace_line: int


@dataclass(frozen=True)
class PreparedCasesResult:
    selected_count: int
    cases: list[BenchmarkCase]
    manifest_path: Path


@dataclass(frozen=True)
class GoldRecord:
    case_id: str
    canonical: dict[str, Any]
    alternates: dict[str, list[str]]
    label_source: str = "baseline_assisted"
    summary_rubric: dict[str, Any] | None = None


@dataclass(frozen=True)
class LabelingStatus:
    total_cases: int
    labeled: int
    remaining: int
    next_case_id: str | None
