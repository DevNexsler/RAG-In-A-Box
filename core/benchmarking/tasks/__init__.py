from __future__ import annotations

from core.benchmarking.tasks.base import BenchmarkTask
from core.benchmarking.tasks.enrichment import ENRICHMENT_TASK

_TASKS = {ENRICHMENT_TASK.name: ENRICHMENT_TASK}


def get_task(name: str = "enrichment") -> BenchmarkTask:
    try:
        return _TASKS[name]
    except KeyError as exc:
        raise ValueError(f"unknown benchmark task: {name}") from exc


__all__ = ["BenchmarkTask", "get_task"]
