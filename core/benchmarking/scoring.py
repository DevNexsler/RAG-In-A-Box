from __future__ import annotations

import json
import math
import re
from datetime import datetime
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from doc_enrichment import parse_enrichment_response

DEFAULT_FIELD_WEIGHTS: dict[str, float] = {
    "doc_type": 0.18,
    "topics": 0.18,
    "keywords": 0.14,
    "key_facts": 0.16,
    "suggested_tags": 0.10,
    "suggested_folder": 0.10,
    "importance": 0.06,
    "entities_people": 0.03,
    "entities_places": 0.02,
    "entities_orgs": 0.01,
    "entities_dates": 0.01,
    "summary": 0.01,
}

_SET_FIELDS = {"doc_type", "topics", "keywords", "suggested_tags"}
_ENTITY_FIELDS = {"entities_people", "entities_places", "entities_orgs", "entities_dates"}
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_ORDINAL_RE = re.compile(r"(\d+)(st|nd|rd|th)\b", re.IGNORECASE)
_DATE_EXTRACT_PATTERNS = (
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b"),
    re.compile(r"\b\d{1,2}-\d{1,2}-\d{4}\b"),
    re.compile(
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+\d{4}\b",
        re.IGNORECASE,
    ),
)
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%B %d %Y",
    "%b %d %Y",
    "%d %B %Y",
    "%d %b %Y",
)


@dataclass(frozen=True)
class CaseScore:
    total_score: float
    field_scores: dict[str, float]
    weighted_scores: dict[str, float]
    reliability: dict[str, Any]
    subscores: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


AUDIT_COMPOSITE_WEIGHTS: dict[str, float] = {
    "extraction_core": 0.7,
    "filing_taxonomy": 0.2,
    "summary_quality": 0.1,
}
_AUDIT_EXTRACTION_FIELDS = (
    "doc_type",
    "topics",
    "keywords",
    "key_facts",
    "importance",
    "entities_people",
    "entities_places",
    "entities_orgs",
    "entities_dates",
)
_AUDIT_FILING_FIELDS = ("suggested_tags", "suggested_folder")


def normalize_prediction(raw_output: str) -> dict[str, str]:
    return parse_enrichment_response(raw_output)


def score_raw_case(
    raw_output: str,
    gold: Mapping[str, Any],
    *,
    field_weights: Mapping[str, float] | None = None,
) -> tuple[dict[str, str], CaseScore]:
    normalized = normalize_prediction(raw_output)
    return normalized, score_case(normalized, gold, field_weights=field_weights)


def score_audit_raw_case(
    raw_output: str,
    gold: Mapping[str, Any],
) -> tuple[dict[str, str], CaseScore]:
    normalized = normalize_prediction(raw_output)
    return normalized, score_audit_case(normalized, gold)


def score_case(
    prediction: Mapping[str, Any],
    gold: Mapping[str, Any],
    *,
    field_weights: Mapping[str, float] | None = None,
) -> CaseScore:
    weights = dict(field_weights or DEFAULT_FIELD_WEIGHTS)
    canonical = gold.get("canonical", {})
    alternates = gold.get("alternates", {})

    field_scores: dict[str, float] = {}
    weighted_scores: dict[str, float] = {}
    for field, weight in weights.items():
        if field not in canonical and field not in alternates:
            score = 0.0
        else:
            score = _score_field(
                field=field,
                predicted=prediction.get(f"enr_{field}", ""),
                canonical=canonical.get(field, "" if field not in _list_like_fields() else []),
                alternates=alternates.get(field, []),
            )
        field_scores[field] = score
        weighted_scores[field] = round(score * weight, 6)

    return CaseScore(
        total_score=sum(weighted_scores.values()),
        field_scores=field_scores,
        weighted_scores=weighted_scores,
        reliability={
            "parse_failed": False,
            "transport_failed": False,
            "internal_failed": False,
        },
    )


def score_audit_case(
    prediction: Mapping[str, Any],
    gold: Mapping[str, Any],
) -> CaseScore:
    summary_rubric = gold.get("summary_rubric")
    if not isinstance(summary_rubric, Mapping):
        raise ValueError("audit gold record missing summary_rubric")

    field_result = score_case(prediction, gold)
    summary_quality = _score_summary_rubric(
        predicted_summary=prediction.get("enr_summary", ""),
        summary_rubric=summary_rubric,
    )
    extraction_core = _weighted_subset_score(
        field_scores=field_result.field_scores,
        fields=_AUDIT_EXTRACTION_FIELDS,
    )
    canonical = gold.get("canonical", {})
    alternates = gold.get("alternates", {})
    filing_taxonomy = round(
        (
            _score_audit_suggested_tags(
                predicted=prediction.get("enr_suggested_tags", ""),
                canonical=canonical.get("suggested_tags", []),
                alternates=alternates.get("suggested_tags", []),
            )
            + _score_folder(
                prediction.get("enr_suggested_folder", ""),
                canonical.get("suggested_folder", ""),
                alternates.get("suggested_folder", []),
            )
        )
        / 2,
        6,
    )
    subscores = {
        "extraction_core": extraction_core,
        "filing_taxonomy": filing_taxonomy,
        "summary_quality": summary_quality,
    }
    total_score = round(
        sum(subscores[key] * AUDIT_COMPOSITE_WEIGHTS[key] for key in AUDIT_COMPOSITE_WEIGHTS),
        6,
    )
    return CaseScore(
        total_score=total_score,
        field_scores=field_result.field_scores,
        weighted_scores=field_result.weighted_scores,
        reliability=field_result.reliability,
        subscores=subscores,
    )


def score_failed_case(error: str) -> CaseScore:
    parse_failed = "parse" in error.lower() or "json" in error.lower()
    transport_failed = error.lower() in {"timeout", "http_error", "transport_error"}
    internal_failed = not parse_failed and not transport_failed
    field_scores = {field: 0.0 for field in DEFAULT_FIELD_WEIGHTS}
    weighted_scores = {field: 0.0 for field in DEFAULT_FIELD_WEIGHTS}
    return CaseScore(
        total_score=0.0,
        field_scores=field_scores,
        weighted_scores=weighted_scores,
        reliability={
            "parse_failed": parse_failed,
            "transport_failed": transport_failed,
            "internal_failed": internal_failed,
            "error": error,
        },
        subscores={
            "extraction_core": 0.0,
            "filing_taxonomy": 0.0,
            "summary_quality": 0.0,
        },
    )


def _weighted_subset_score(*, field_scores: Mapping[str, float], fields: tuple[str, ...]) -> float:
    total_weight = sum(DEFAULT_FIELD_WEIGHTS[field] for field in fields)
    if total_weight == 0:
        return 0.0
    score = sum(field_scores[field] * DEFAULT_FIELD_WEIGHTS[field] for field in fields) / total_weight
    return round(score, 6)


def _score_summary_rubric(*, predicted_summary: Any, summary_rubric: Mapping[str, Any]) -> float:
    coverage = summary_rubric.get("coverage")
    brevity = summary_rubric.get("brevity")
    hallucination = summary_rubric.get("hallucination")
    if not isinstance(coverage, list) or not isinstance(brevity, Mapping) or not isinstance(hallucination, list):
        raise ValueError("audit summary_rubric is invalid")

    coverage_score = _score_summary_coverage(str(predicted_summary), coverage)
    brevity_score = _score_summary_brevity(str(predicted_summary), brevity)
    hallucination_score = _score_summary_hallucination(str(predicted_summary), hallucination)
    return round((coverage_score * 0.5) + (brevity_score * 0.25) + (hallucination_score * 0.25), 6)


def _score_summary_coverage(predicted_summary: str, coverage_points: list[Any]) -> float:
    points = [str(point).strip() for point in coverage_points if str(point).strip()]
    if not points:
        return 1.0
    summary_tokens = _tokenize(predicted_summary)
    scores = []
    for point in points:
        point_tokens = _tokenize(point)
        if not point_tokens:
            scores.append(1.0)
            continue
        scores.append(len(summary_tokens & point_tokens) / len(point_tokens))
    return round(sum(scores) / len(scores), 6)


def _score_summary_brevity(predicted_summary: str, brevity: Mapping[str, Any]) -> float:
    max_sentences = brevity.get("max_sentences", 0)
    max_words = brevity.get("max_words", 0)
    if not isinstance(max_sentences, int) or not isinstance(max_words, int):
        raise ValueError("audit summary_rubric brevity values must be integers")

    if max_sentences <= 0 and max_words <= 0:
        return 1.0

    sentence_count = max(1, len([part for part in re.split(r"[.!?]+", predicted_summary) if part.strip()])) if predicted_summary.strip() else 0
    word_count = len(_tokenize(predicted_summary))
    sentence_score = _length_limit_score(actual=sentence_count, maximum=max_sentences)
    word_score = _length_limit_score(actual=word_count, maximum=max_words)
    return round((sentence_score + word_score) / 2, 6)


def _score_summary_hallucination(predicted_summary: str, forbidden_points: list[Any]) -> float:
    points = [str(point).strip() for point in forbidden_points if str(point).strip()]
    if not points:
        return 1.0
    penalty = max(_fact_similarity(predicted_summary, point) for point in points)
    return round(max(0.0, 1.0 - penalty), 6)


def _length_limit_score(*, actual: int, maximum: int) -> float:
    if maximum <= 0:
        return 1.0
    if actual <= maximum:
        return 1.0
    overflow_ratio = (actual - maximum) / maximum
    return round(max(0.0, 1.0 - overflow_ratio), 6)


def _score_audit_suggested_tags(*, predicted: Any, canonical: Any, alternates: Any) -> float:
    predicted_tags = _to_string_set(predicted)
    canonical_tags = _to_string_set(canonical)
    alternate_tags = _to_string_set(alternates)
    allowed_tags = canonical_tags | alternate_tags

    if not predicted_tags and not canonical_tags:
        return 1.0
    if not predicted_tags:
        return 0.0

    precision = len(predicted_tags & allowed_tags) / len(predicted_tags) if predicted_tags else 0.0
    recall = len(predicted_tags & canonical_tags) / len(canonical_tags) if canonical_tags else 1.0
    return round((precision + recall) / 2, 6)


def _score_field(*, field: str, predicted: Any, canonical: Any, alternates: Any) -> float:
    if field == "suggested_tags":
        return _score_suggested_tags(predicted, canonical, alternates)
    if field in _SET_FIELDS:
        accepted = _canonical_plus_alternates(canonical, alternates)
        return _score_semantic_overlap(
            _to_string_set(predicted),
            [_to_string_set(value) for value in accepted],
        )
    if field == "key_facts":
        return _score_key_facts(predicted, canonical)
    if field == "suggested_folder":
        return _score_folder(predicted, canonical, alternates)
    if field == "importance":
        return _score_importance(predicted, canonical)
    if field == "entities_dates":
        return _score_best_overlap(_to_string_set(predicted, date_values=True), [_to_string_set(canonical, date_values=True)])
    if field in _ENTITY_FIELDS:
        return _score_best_overlap(_to_string_set(predicted), [_to_string_set(canonical)])
    if field == "summary":
        return _score_summary(predicted, canonical)
    return 0.0


def _canonical_plus_alternates(canonical: Any, alternates: Any) -> list[Any]:
    values = [canonical]
    if isinstance(alternates, list):
        values.extend(alternates)
    return values


def _score_best_overlap(predicted: set[str], accepted_sets: list[set[str]]) -> float:
    if not accepted_sets:
        return 1.0 if not predicted else 0.0
    best = 0.0
    for expected in accepted_sets:
        score = _jaccard(predicted, expected)
        if score > best:
            best = score
    return round(best, 6)


def _score_semantic_overlap(predicted: set[str], accepted_sets: list[set[str]]) -> float:
    if not accepted_sets:
        return 1.0 if not predicted else 0.0
    best = 0.0
    for expected in accepted_sets:
        score = _score_phrase_collection(sorted(predicted), sorted(expected))
        if score > best:
            best = score
    return round(best, 6)


def _score_phrase_collection(predicted: list[str], expected: list[str]) -> float:
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0

    remaining = set(range(len(expected)))
    matched_score = 0.0
    for predicted_item in predicted:
        best_idx = None
        best_score = 0.0
        for idx in remaining:
            score = _fact_similarity(predicted_item, expected[idx])
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is not None and best_score > 0.0:
            remaining.remove(best_idx)
            matched_score += best_score

    return matched_score / max(len(predicted), len(expected))


def _score_folder(predicted: Any, canonical: Any, alternates: Any) -> float:
    predicted_norm = _normalize_folder(predicted)
    if not predicted_norm:
        return 1.0 if not _normalize_scalar(canonical) else 0.0

    accepted = [_normalize_folder(canonical)]
    if isinstance(alternates, list):
        accepted.extend(_normalize_folder(value) for value in alternates)
    accepted = [value for value in accepted if value]

    if predicted_norm in accepted:
        return 1.0

    predicted_parts = _split_folder(predicted_norm)
    for expected in accepted:
        expected_parts = _split_folder(expected)
        if predicted_parts == expected_parts[: len(predicted_parts)]:
            return 0.5
    return 0.0


def _score_suggested_tags(predicted: Any, canonical: Any, alternates: Any) -> float:
    allowed = _to_string_set(canonical)
    allowed.update(_to_string_set(alternates))
    return _score_best_overlap(_to_string_set(predicted), [allowed])


def _score_importance(predicted: Any, canonical: Any) -> float:
    predicted_value = _parse_importance(predicted)
    canonical_value = _parse_importance(canonical)
    if predicted_value is None or canonical_value is None:
        return 1.0 if predicted_value is None and canonical_value is None else 0.0
    return round(max(0.0, 1.0 - abs(predicted_value - canonical_value)), 6)


def _score_summary(predicted: Any, canonical: Any) -> float:
    predicted_tokens = _tokenize(_normalize_scalar(predicted))
    canonical_tokens = _tokenize(_normalize_scalar(canonical))
    return round(_jaccard(predicted_tokens, canonical_tokens), 6)


def _score_key_facts(predicted: Any, canonical: Any) -> float:
    predicted_facts = _to_string_list(predicted, key_facts=True)
    canonical_facts = _to_string_list(canonical, key_facts=True)
    if not predicted_facts and not canonical_facts:
        return 1.0
    if not predicted_facts or not canonical_facts:
        return 0.0

    remaining = set(range(len(canonical_facts)))
    matched_score = 0.0
    for predicted_fact in predicted_facts:
        best_idx = None
        best_score = 0.0
        for idx in remaining:
            score = _fact_similarity(predicted_fact, canonical_facts[idx])
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is not None and best_score > 0.0:
            remaining.remove(best_idx)
            matched_score += best_score

    return round(matched_score / max(len(predicted_facts), len(canonical_facts)), 6)


def _fact_similarity(left: str, right: str) -> float:
    left_tokens = _tokenize(left)
    right_tokens = _tokenize(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    if overlap == 0:
        return 0.0
    return (2 * overlap) / (len(left_tokens) + len(right_tokens))


def _to_string_set(value: Any, *, key_facts: bool = False, date_values: bool = False) -> set[str]:
    return set(_to_string_list(value, key_facts=key_facts, date_values=date_values))


def _to_string_list(value: Any, *, key_facts: bool = False, date_values: bool = False) -> list[str]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        stripped = value.strip()
        if key_facts and stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                items = [part for part in stripped.split("\n") if part.strip()]
            else:
                items = parsed if isinstance(parsed, list) else [stripped]
        elif date_values:
            items = _extract_date_items(stripped)
        elif "," in stripped:
            items = [part.strip() for part in stripped.split(",")]
        elif stripped:
            items = [stripped]
        else:
            items = []
    elif value:
        items = [value]
    else:
        items = []

    normalized_items: list[str] = []
    for item in items:
        normalized = _normalize_date_value(item) if date_values else _normalize_scalar(item)
        if normalized:
            normalized_items.append(normalized)
    return normalized_items


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _normalize_scalar(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def _normalize_date_value(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    cleaned = _ORDINAL_RE.sub(r"\1", text)
    cleaned = cleaned.replace(",", " ")
    cleaned = " ".join(cleaned.split())
    for date_format in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned.title(), date_format).date().isoformat()
        except ValueError:
            continue
    return _normalize_scalar(text)


def _extract_date_items(value: str) -> list[str]:
    if _normalize_date_value(value) != _normalize_scalar(value):
        return [value]

    for separator in ("\n", ";", "|"):
        if separator in value:
            return [part.strip() for part in value.split(separator) if part.strip()]

    matches: list[str] = []
    for pattern in _DATE_EXTRACT_PATTERNS:
        matches.extend(match.group(0) for match in pattern.finditer(value))
    if matches:
        return matches

    return [value]


def _normalize_folder(value: Any) -> str:
    normalized = _normalize_scalar(value)
    return normalized.strip("/")


def _split_folder(value: str) -> list[str]:
    return [part for part in value.split("/") if part]


def _parse_importance(value: Any) -> float | None:
    normalized = _normalize_scalar(value)
    if not normalized:
        return None
    try:
        parsed = float(normalized)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return max(0.0, min(1.0, parsed))


def _tokenize(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(_normalize_scalar(value)))


def _list_like_fields() -> set[str]:
    return _SET_FIELDS | _ENTITY_FIELDS | {"key_facts"}
