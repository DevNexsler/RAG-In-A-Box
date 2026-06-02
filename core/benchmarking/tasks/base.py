from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


ScoreFunction = Callable[[str, Mapping[str, Any]], tuple[dict[str, str], Any]]


@dataclass(frozen=True)
class BenchmarkTask:
    name: str
    default_score_mode: str
    score_modes: frozenset[str]
    score_raw_output: Callable[[str, Mapping[str, Any], str], tuple[dict[str, str], Any]]
