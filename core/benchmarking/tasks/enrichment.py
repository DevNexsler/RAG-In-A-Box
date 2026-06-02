from __future__ import annotations

from typing import Any, Mapping

from core.benchmarking.scoring import score_audit_raw_case, score_raw_case
from core.benchmarking.tasks.base import BenchmarkTask


def score_enrichment_output(
    raw_output: str,
    gold: Mapping[str, Any],
    score_mode: str,
) -> tuple[dict[str, str], Any]:
    if score_mode == "audit":
        return score_audit_raw_case(raw_output, gold)
    if score_mode == "standard":
        return score_raw_case(raw_output, gold)
    raise ValueError(f"unsupported enrichment score_mode: {score_mode}")


ENRICHMENT_TASK = BenchmarkTask(
    name="enrichment",
    default_score_mode="standard",
    score_modes=frozenset({"standard", "audit"}),
    score_raw_output=score_enrichment_output,
)
