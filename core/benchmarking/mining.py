from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from core.benchmarking.models import HardCaseCandidate, HardCaseSelection, TraceMetadata

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

            raw = json.loads(line)
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

    for row in rows:
        if row.prompt_hash in seen_hashes:
            continue
        seen_hashes.add(row.prompt_hash)

        if not row.success and row.failure_type and row.failure_status_code is not None:
            provider_failure_cases.append(row)
            continue

        candidate = score_hard_flags(row)
        if candidate.hard_score > 0:
            candidates.append(candidate)

    candidates.sort(key=lambda item: (-item.hard_score, item.trace.trace_file, item.trace.trace_line))
    return HardCaseSelection(
        hard_cases=candidates[:limit],
        provider_failure_cases=provider_failure_cases[:limit],
    )


def _extract_user_prompt(raw: dict[str, Any]) -> str:
    payload = raw.get("request", {}).get("payload", {})
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
    choices = raw.get("response", {}).get("choices", [])
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    content = first.get("message", {}).get("content")
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
