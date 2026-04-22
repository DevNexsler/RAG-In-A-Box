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


@dataclass(frozen=True)
class LabelingStatus:
    total_cases: int
    labeled: int
    remaining: int
    next_case_id: str | None
