from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
