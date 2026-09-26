"""LLM-based document enrichment for the indexing pipeline.

Sends a representative sample of each document's text to a generative LLM
(via configurable LLM provider) and parses its structured JSON response into
consistent metadata fields stored in LanceDB.

For documents longer than max_input_chars, uses head+tail sampling (first
half + last half) so that conclusions, summaries, and late-document facts
are captured alongside the opening context.

All fields are returned as strings (comma-separated for lists, JSON array
for key_facts) for consistent querying and filtering.
"""

from __future__ import annotations

import copy
import itertools
import json
import logging
import math
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from core.enrichment_postprocess import (
    canonicalize_doc_type,
    ground_card_suffixes,
    repair_enrichment,
)
from core.doc_type_vocabulary import (
    constrain_doc_type,
    enrichment_input_hash,
    reconcile_doc_type,
    vocabulary_alias_map,
)
from core.resilience import is_transient
from core.tracing import get_tracer

if TYPE_CHECKING:
    from providers.llm import LLMGenerator
    from taxonomy_store import TaxonomyStore

logger = logging.getLogger(__name__)

# Lazy tracer (resolves provider per call); spans are no-ops when tracing is off.
_tracer = get_tracer("pipeline")

_MODALITY_INSTRUCTIONS = """\
Preserve the primary item's speech act and uncertainty in summary, doc_type, and key_facts:
- Questions remain questions, not proposals or confirmations. Describe what the
  sender asks; do not assert the answer or infer that a rule exists from a question.
  A question about whether a rule applies does not request its adoption. For
  example, "Is that the rule from now on?" asks what applies; it does not mean
  "I propose making that the rule." Do not substitute "should" for "is", or
  describe the question as seeking to establish, change, or confirm a policy.
  Every key fact about the question must retain this distinction too: report
  "The sender asks whether ...", not a proposed or established rule.
- Proposals remain proposals, not adopted rules. Require explicit evidence of
  adoption before describing a proposal as an established policy.
- Preserve temporal limits: temporary instructions must not become standing policy.
  Asking whether an instruction applies in future does not extend its duration.
- Explicitly adopted standing rules remain statements of policy; do not weaken
  them into questions or proposals.
- Nearby context must not change the primary item's intent. Attribute contextual
  instructions or decisions to their source and keep their stated time scope.
Apply these distinctions to all other metadata too, including context_key_facts.
"""

_PROMPT_TEMPLATE = """\
Extract metadata from this document. Respond with ONLY valid JSON, no other text.
{modality_instructions}

{{
  "summary": "2-3 sentence summary of the document's purpose and key content",
  "doc_type": ["type1", "type2"],
  "entities_people": ["full names of people mentioned"],
  "entities_places": ["addresses, cities, locations"],
  "entities_orgs": ["company and organization names"],
  "entities_dates": ["YYYY-MM-DD format dates mentioned"],
  "topics": ["5-10 high-level topics"],
  "keywords": ["10-20 specific terms and phrases"],
  "key_facts": ["most important facts, conclusions, or action items"],
  "suggested_tags": ["classification tags for this document"],
  "suggested_folder": "best folder path for filing this document",
  "importance": 0.5,
  "atomic_entities_people": ["people mentioned in the primary item only"],
  "atomic_entities_places": ["places mentioned in the primary item only"],
  "atomic_entities_orgs": ["organizations mentioned in the primary item only"],
  "atomic_entities_dates": ["dates mentioned in the primary item only"],
  "atomic_topics": ["topics from the primary item only"],
  "context_entities_people": ["people inferred from relevant nearby context"],
  "context_entities_places": ["places inferred from relevant nearby context"],
  "context_entities_orgs": ["organizations inferred from relevant nearby context"],
  "context_entities_dates": ["dates inferred from relevant nearby context"],
  "context_topics": ["topics inferred from relevant nearby context"],
  "context_key_facts": ["facts inferred from relevant nearby context"],
  "context_relationship": "why the nearby context is relevant",
  "context_confidence": "high|medium|low|ambiguous",
  "context_source_message_ids": ["nearby message ids used"],
  "context_warning": "ambiguity or unrelated nearby context warning"
}}

For "importance": rate the document's overall importance/usefulness on a 0.0-1.0 scale:
- 1.0 = critical reference, frequently needed, high-value knowledge
- 0.7-0.9 = important, actionable, or broadly useful
- 0.4-0.6 = average utility, general notes or routine content
- 0.1-0.3 = low importance, ephemeral, or narrowly relevant
- 0.0 = trivial, outdated, or noise
Use atomic_* fields for facts visible in the primary item itself.
If no nearby context section is provided, leave all context_* fields empty.
Never copy placeholder/example/schema description text into values. If a field
has no evidence, use [] for arrays and "" for strings.
{taxonomy_block}
Document title: {title}
Document type: {source_type}

Document text:
{text}"""

_CONTEXT_PROMPT_TEMPLATE = """\
Extract metadata from this document. Respond with ONLY valid JSON, no other text.
The context_* fields are required output keys; never omit them.
{modality_instructions}

{{
  "context_entities_people": ["people inferred from relevant nearby context"],
  "context_entities_places": ["places inferred from relevant nearby context"],
  "context_entities_orgs": ["organizations inferred from relevant nearby context"],
  "context_entities_dates": ["dates inferred from relevant nearby context"],
  "context_topics": ["topics inferred from relevant nearby context"],
  "context_key_facts": ["facts inferred from relevant nearby context"],
  "context_relationship": "why the nearby context is relevant, or empty string",
  "context_confidence": "high|medium|low|ambiguous, or empty string",
  "context_source_message_ids": ["nearby message ids used"],
  "context_warning": "ambiguity or unrelated nearby context warning",
  "atomic_entities_people": ["people mentioned in the PRIMARY ITEM only"],
  "atomic_entities_places": ["places mentioned in the PRIMARY ITEM only"],
  "atomic_entities_orgs": ["organizations mentioned in the PRIMARY ITEM only"],
  "atomic_entities_dates": ["dates mentioned in the PRIMARY ITEM only"],
  "atomic_topics": ["topics from the PRIMARY ITEM only"],
  "summary": "2-3 sentence summary of the primary item's purpose and key content",
  "doc_type": ["type1", "type2"],
  "entities_people": ["full names of people mentioned"],
  "entities_places": ["addresses, cities, locations"],
  "entities_orgs": ["company and organization names"],
  "entities_dates": ["YYYY-MM-DD format dates mentioned"],
  "topics": ["5-10 high-level topics"],
  "keywords": ["10-20 specific terms and phrases"],
  "key_facts": ["most important facts, conclusions, or action items"],
  "suggested_tags": ["classification tags for this document"],
  "suggested_folder": "best folder path for filing this document",
  "importance": 0.5
}}

For "importance": rate the primary item's overall importance/usefulness on a 0.0-1.0 scale:
- 1.0 = critical reference, frequently needed, high-value knowledge
- 0.7-0.9 = important, actionable, or broadly useful
- 0.4-0.6 = average utility, general notes or routine content
- 0.1-0.3 = low importance, ephemeral, or narrowly relevant
- 0.0 = trivial, outdated, or noise

Nearby same-channel context candidates may or may not describe the primary item.
Treat nearby messages as candidates only. Judge relevance before using them.
Avoid adding unrelated nearby conversation to any field.
key_facts and keywords describe the PRIMARY ITEM only. Put facts and terms taken
from nearby context in context_key_facts only, never in key_facts or keywords.
summary describes what the PRIMARY ITEM itself says. It may name what the item
replies to, but must not present nearby-context details as the item's content.
If you use nearby context in entities, topics, tags, folder, or importance, you
MUST also fill the matching context_* fields.
Fill context_* fields only when nearby context is relevant to the PRIMARY ITEM.
When using nearby context, set context_confidence, context_relationship, and
context_source_message_ids. Use context_warning for ambiguity or rejected context.
Never copy placeholder/example/schema description text into values. If a field
has no evidence, use [] for arrays and "" for strings.
If candidates conflict, set context_confidence to ambiguous and explain the
conflict in context_warning.
{taxonomy_block}
PRIMARY ITEM
Document title: {title}
Document type: {source_type}

Document text:
{text}

NEARBY SAME-CHANNEL CONTEXT CANDIDATES
{context_text}"""

_TAXONOMY_INSTRUCTION = """
For "doc_type": select ONLY from Available Document Types below. Use the exact
names. Prefer one primary type; add a second only when both clearly apply. If
none fit, use ["unclassified"]. Do not invent new type labels.
For "suggested_tags" and "suggested_folder": use the taxonomy below.
Pick the most relevant tags from Available Tags (you may also add new ones).
Pick the single best matching folder path from Available Folders (use the exact path).
"""

# Written beside enrichment fields so a later pass can skip the LLM when the
# enrichment inputs (not the broader change_hash) are unchanged.
ENRICHMENT_INPUT_HASH_FIELD = "enr_input_hash"

# Raw keys the LLM prompt asks for (unprefixed)
_ENRICHMENT_KEYS_RAW = (
    "summary",
    "doc_type",
    "entities_people",
    "entities_places",
    "entities_orgs",
    "entities_dates",
    "topics",
    "keywords",
    "key_facts",
    "suggested_tags",
    "suggested_folder",
    "importance",
)

_CONTEXT_KEYS_RAW = (
    "atomic_entities_people",
    "atomic_entities_places",
    "atomic_entities_orgs",
    "atomic_entities_dates",
    "atomic_topics",
    "context_entities_people",
    "context_entities_places",
    "context_entities_orgs",
    "context_entities_dates",
    "context_topics",
    "context_key_facts",
    "context_relationship",
    "context_confidence",
    "context_source_message_ids",
    "context_warning",
)

# Raw keys without which an enriched row carries no usable metadata. An LLM
# response that omits them is rejected rather than stored (see
# missing_required_fields / structured_response_is_usable).
REQUIRED_ENRICHMENT_FIELDS = ("summary", "doc_type")

# Prefixed field names stored in LanceDB metadata (prevent collision with frontmatter)
CORE_ENRICHMENT_FIELDS = tuple(f"enr_{k}" for k in _ENRICHMENT_KEYS_RAW)
ENRICHMENT_FIELDS = tuple(f"enr_{k}" for k in (*_ENRICHMENT_KEYS_RAW, *_CONTEXT_KEYS_RAW))

_SCHEMA_STRING_KEYS = {
    "summary",
    "suggested_folder",
    "context_relationship",
    "context_confidence",
    "context_warning",
}

_SCHEMA_FIELD_DESCRIPTIONS = {
    "summary": "2-3 sentence summary of the primary item's purpose and key content",
    "doc_type": "Document type classifications",
    "entities_people": "Full names of people mentioned",
    "entities_places": "Addresses, cities, locations",
    "entities_orgs": "Company and organization names",
    "entities_dates": "Dates mentioned in YYYY-MM-DD format",
    "topics": "5-10 high-level topics",
    "keywords": "10-20 specific terms and phrases",
    "key_facts": "Most important facts, conclusions, or action items",
    "suggested_tags": "Classification tags for this document",
    "suggested_folder": "Best folder path for filing this document, or empty string",
    "importance": "Importance score from 0.0 to 1.0",
    "atomic_entities_people": "People mentioned in the primary item only",
    "atomic_entities_places": "Places mentioned in the primary item only",
    "atomic_entities_orgs": "Organizations mentioned in the primary item only",
    "atomic_entities_dates": "Dates mentioned in the primary item only",
    "atomic_topics": "Topics from the primary item only",
    "context_entities_people": "People inferred from relevant nearby context",
    "context_entities_places": "Places inferred from relevant nearby context",
    "context_entities_orgs": "Organizations inferred from relevant nearby context",
    "context_entities_dates": "Dates inferred from relevant nearby context",
    "context_topics": "Topics inferred from relevant nearby context",
    "context_key_facts": "Facts inferred from relevant nearby context",
    "context_relationship": "Why the nearby context is relevant, or empty string",
    "context_confidence": "high, medium, low, ambiguous, or empty string",
    "context_source_message_ids": "Nearby message ids used",
    "context_warning": "Ambiguity or unrelated nearby context warning",
}

_SCHEMA_PLACEHOLDER_VALUES = {
    description.lower() for description in _SCHEMA_FIELD_DESCRIPTIONS.values()
}
_SCHEMA_PLACEHOLDER_VALUES.update(
    {
        "type1",
        "type2",
        *(_ENRICHMENT_KEYS_RAW + _CONTEXT_KEYS_RAW),
        *(f"enr_{key}" for key in (_ENRICHMENT_KEYS_RAW + _CONTEXT_KEYS_RAW)),
    }
)

_SCHEMA_ARRAY_KEYS = set(_ENRICHMENT_KEYS_RAW + _CONTEXT_KEYS_RAW) - {
    *_SCHEMA_STRING_KEYS,
    "importance",
}
_SEMANTIC_ARRAY_KEYS = _SCHEMA_ARRAY_KEYS - {"context_source_message_ids"}

_CONTEXT_AMBIGUITY_TERMS = (
    "ambiguous",
    "ambiguity",
    "conflict",
    "conflicting",
    "correction",
    "might",
    "not sure",
    "possibly",
    "uncertain",
)

# Numbers are how a fact shows which text it came from: amounts, card and
# account tails, order ids and phone groups are copied, not paraphrased.
# A thousands separator only counts between digit groups, so "101, 102" stays
# two numbers.
_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
# Dates, times and years prove nothing about the source: the model rewrites
# them (the prompt asks for YYYY-MM-DD), and every nearby-context line starts
# with a timestamp.
_DATE_TIME_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?"
    r"|\d{1,2}/\d{1,2}/\d{2,4}"
    r"|\d{1,2}:\d{2}(?::\d{2})?"
    r"|\b(?:19|20)\d{2}\b"
)
# Shorter numbers (quantities, sizes, "4 burners") occur in almost any text.
_MIN_EVIDENCE_DIGITS = 3
# A dollar total the model added up from the primary item's own dollar amounts
# is supported by the primary item, even when only a nearby message prints it.
_AMOUNT_RE = re.compile(r"\$\s?(" + _NUMBER_RE.pattern + ")")
_MAX_SUMMED_AMOUNTS = 3
_MAX_AMOUNTS_TO_SUM = 20
# Text-only phrases need enough characters to avoid moving common short tokens.
_MIN_TEXT_EVIDENCE_LEN = 8


def _schema_property_for(raw_key: str) -> dict[str, Any]:
    description = _SCHEMA_FIELD_DESCRIPTIONS.get(raw_key, raw_key.replace("_", " "))
    if raw_key == "importance":
        return {"type": "number", "description": description}
    if raw_key in _SCHEMA_STRING_KEYS:
        return {"type": "string", "description": description}
    return {
        "type": "array",
        "items": {"type": "string"},
        "description": description,
    }


_ENRICHMENT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        raw_key: _schema_property_for(raw_key)
        for raw_key in (*_ENRICHMENT_KEYS_RAW, *_CONTEXT_KEYS_RAW)
    },
    "required": list((*_ENRICHMENT_KEYS_RAW, *_CONTEXT_KEYS_RAW)),
    "additionalProperties": False,
}


def enrichment_response_schema() -> dict[str, Any]:
    """Return the provider JSON schema for the same fields requested in prompts."""
    return copy.deepcopy(_ENRICHMENT_RESPONSE_SCHEMA)


def empty_enrichment() -> dict[str, str]:
    """Return a dict with all enrichment fields set to empty strings."""
    return {f: "" for f in ENRICHMENT_FIELDS}


def failed_enrichment(reason: str, transient: bool = False) -> dict:
    """Return an enrichment dict that signals failure with a reason.

    ``transient`` marks provider-level failures (connection refused, timeout,
    5xx — see core.resilience.is_transient) so the degraded ledger does not
    charge its attempts cap for an LLM-provider outage (#0251).

    The caller should check for ``_enrichment_failed`` and remove it (and
    ``_enrichment_transient``) before storing in LanceDB.
    """
    result = empty_enrichment()
    result["_enrichment_failed"] = reason
    result["_enrichment_transient"] = transient
    return result


def _extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from LLM output, stripping markdown fences.

    Handles common LLM quirks:
      - Markdown ```json fences
      - <think>...</think> tags (Qwen3 reasoning)
      - Trailing text after valid JSON ("Extra data" errors)
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    # Handle Qwen3 thinking tags — strip <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

    # Use JSONDecoder to parse only the first JSON object, ignoring trailing text
    try:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(cleaned)
        return obj
    except json.JSONDecodeError:
        pass

    # Attempt to salvage truncated JSON (e.g. token limit cut off mid-value).
    # Progressively strip trailing incomplete tokens and try to close the object.
    salvaged = _salvage_truncated_json(cleaned)
    if salvaged is not None:
        return salvaged

    # Nothing worked — raise a clear error
    return json.loads(cleaned)


def _salvage_truncated_json(text: str) -> dict[str, Any] | None:
    """Try to recover a partial JSON object truncated by token limits.

    Strips trailing incomplete values and attempts to close open arrays/objects.
    Returns the parsed dict or None if recovery fails.
    """
    s = text.rstrip()
    # Try closing open structures, stripping up to 200 chars from the tail
    for trim in range(0, min(200, len(s))):
        candidate = s if trim == 0 else s[:-trim]
        # Remove trailing comma
        candidate = candidate.rstrip().rstrip(",").rstrip()
        # Count open/close brackets to figure out what needs closing
        closers = ""
        for ch in candidate:
            if ch in ('{', '['):
                closers = ('}' if ch == '{' else ']') + closers
            elif ch in ('}', ']') and closers and closers[0] == ch:
                closers = closers[1:]  # doesn't match LIFO but close enough
        # Close any remaining open brackets
        # Actually, rebuild closers by scanning properly
        stack = []
        in_string = False
        escape = False
        for ch in candidate:
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in ('{', '['):
                stack.append('}' if ch == '{' else ']')
            elif ch in ('}', ']') and stack:
                stack.pop()
        # If we're inside a string, close it first
        if in_string:
            candidate += '"'
        closers = "".join(reversed(stack))
        attempt = candidate + closers
        try:
            obj = json.loads(attempt)
            if isinstance(obj, dict):
                logger.info("Salvaged truncated JSON (%d chars trimmed)", trim)
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _normalize_list(value: Any) -> str:
    """Convert a list (or string) to a comma-separated string."""
    if isinstance(value, list):
        return ", ".join(str(v).strip() for v in value if str(v).strip())
    if isinstance(value, str):
        return value.strip()
    return str(value).strip() if value else ""


def _normalize_metadata_list(raw_key: str, value: Any) -> str:
    """Normalize list-like metadata and drop prompt/schema placeholder values."""
    if isinstance(value, list):
        values = [str(v).strip() for v in value if str(v).strip()]
    elif isinstance(value, str):
        values = [value.strip()] if value.strip() else []
    elif value:
        values = [str(value).strip()]
    else:
        values = []

    values = [item for item in values if not _is_placeholder_value(raw_key, item)]
    return ", ".join(values)


def _is_placeholder_value(raw_key: str, value: str) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return True
    if normalized == _SCHEMA_FIELD_DESCRIPTIONS.get(raw_key, "").lower():
        return True
    return normalized in _SCHEMA_PLACEHOLDER_VALUES


def _normalize_enrichment(raw: dict[str, Any]) -> dict[str, str]:
    """Normalize raw LLM JSON into consistent prefixed string fields."""
    result: dict[str, str] = {}
    for raw_key in (*_ENRICHMENT_KEYS_RAW, *_CONTEXT_KEYS_RAW):
        enr_key = f"enr_{raw_key}"
        value = raw.get(raw_key)
        if value is None:
            result[enr_key] = ""
        elif raw_key == "importance":
            # Normalize to a clamped float string
            try:
                imp = max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                imp = 0.5
            result[enr_key] = str(imp)
        elif raw_key in (
            "summary",
            "suggested_folder",
            "context_relationship",
            "context_confidence",
            "context_warning",
        ):
            result[enr_key] = str(value).strip()
        elif raw_key == "doc_type":
            # Canonical spelling keeps the published enr_doc_type filter whole:
            # LIKE matching cannot bridge separator variants of one concept (#1251).
            result[enr_key] = canonicalize_doc_type(
                _normalize_metadata_list(raw_key, value)
            )
        elif raw_key in ("key_facts", "context_key_facts"):
            if isinstance(value, list):
                result[enr_key] = json.dumps(
                    [
                        str(v).strip()
                        for v in value
                        if str(v).strip()
                        and not _is_placeholder_value(raw_key, str(v).strip())
                    ]
                )
            elif isinstance(value, str):
                result[enr_key] = (
                    "" if _is_placeholder_value(raw_key, value) else value.strip()
                )
            else:
                result[enr_key] = ""
        else:
            result[enr_key] = _normalize_metadata_list(raw_key, value)
    return result


def parse_enrichment_response(raw_response: str) -> dict[str, str]:
    """Parse raw LLM output and normalize it into enrichment fields."""
    parsed = _extract_json(raw_response)
    return _normalize_enrichment(parsed)


def enrichment_contract_errors(raw_response: str) -> list[str]:
    """Return violations of the complete provider enrichment contract."""
    try:
        parsed = _extract_json(raw_response)
    except (ValueError, TypeError):
        return ["invalid_json"]
    if not isinstance(parsed, dict):
        return ["root:not_object"]

    errors = [
        f"{key}:missing"
        for key in _ENRICHMENT_KEYS_RAW
        if key not in parsed
    ]
    for key in _SCHEMA_STRING_KEYS:
        if key in parsed and not isinstance(parsed[key], str):
            errors.append(f"{key}:not_string")
    for key in _SCHEMA_ARRAY_KEYS:
        if key not in parsed:
            continue
        value = parsed[key]
        if not isinstance(value, list):
            errors.append(f"{key}:not_array")
            continue
        if any(not isinstance(item, str) for item in value):
            errors.append(f"{key}:non_string_item")
            continue
        if key in _SEMANTIC_ARRAY_KEYS and any(
            _is_placeholder_value(key, item) for item in value
        ):
            errors.append(f"{key}:schema_placeholder")

    importance = parsed.get("importance")
    if "importance" in parsed and (
        isinstance(importance, bool)
        or not isinstance(importance, (int, float))
        or not math.isfinite(importance)
        or not 0.0 <= importance <= 1.0
    ):
        errors.append("importance:invalid_number")
    if isinstance(parsed.get("summary"), str) and not parsed["summary"].strip():
        errors.append("summary:empty")
    doc_types = parsed.get("doc_type")
    if isinstance(doc_types, list) and not any(
        isinstance(item, str)
        and item.strip()
        and not _is_placeholder_value("doc_type", item)
        for item in doc_types
    ):
        errors.append("doc_type:empty")
    return errors


def missing_required_fields(enrichment: dict[str, str]) -> list[str]:
    """Required enrichment fields this normalized enrichment does not carry.

    A row without these is not searchable metadata, so it is written degraded
    and re-processed on a later run.
    """
    return [
        field
        for field in REQUIRED_ENRICHMENT_FIELDS
        if not enrichment.get(f"enr_{field}")
    ]


def structured_response_is_usable(raw_response: str) -> bool:
    """Whether a raw structured response yields an enrichment we can store.

    This is the authoritative first-pass validity test for an enrichment call:
    it asks the payload, not the provider's token accounting. Providers bill
    reasoning into ``usage.completion_tokens`` and stop honoring ``max_tokens``
    as a ceiling on it, so token counters answer a different question than
    "did the model deliver the metadata we asked for" (#1097).
    """
    return not enrichment_contract_errors(raw_response)


def _repair_context_omissions(
    enrichment: dict[str, str],
    primary_text: str,
    context_text: str,
) -> dict[str, str]:
    """Preserve provenance when a model uses context but omits context_* fields.

    Key facts and keywords that only the nearby context supports are moved to
    context_key_facts, so the row does not claim another message's facts.
    """
    context_text = (context_text or "").strip()
    if not context_text:
        return enrichment

    repaired = dict(enrichment)
    copied_values: list[str] = []
    had_context_fields = _has_context_fields(repaired)
    primary_lower = primary_text.lower()
    context_lower = context_text.lower()

    for suffix in ("people", "places", "orgs", "dates"):
        source_key = f"enr_entities_{suffix}"
        context_key = f"enr_context_entities_{suffix}"
        if repaired.get(context_key):
            continue
        values = [
            value
            for value in _metadata_values(repaired.get(source_key, ""))
            if value.lower() in context_lower and value.lower() not in primary_lower
        ]
        if values:
            repaired[context_key] = ", ".join(values)
            copied_values.extend(values)

    moved_numbers = _move_context_only_facts(repaired, primary_text, context_text)

    context_used = bool(copied_values or moved_numbers) or _mentions_nearby_context(repaired)
    if not context_used:
        return _normalize_context_consistency(repaired)

    if had_context_fields and not copied_values:
        return _normalize_context_consistency(repaired)

    if not repaired.get("enr_context_confidence"):
        repaired["enr_context_confidence"] = "medium"
    if not repaired.get("enr_context_relationship"):
        repaired["enr_context_relationship"] = "llm_used_nearby_context"
    if not repaired.get("enr_context_source_message_ids"):
        ids = _context_source_ids_for_values(context_text, copied_values + moved_numbers)
        repaired["enr_context_source_message_ids"] = ", ".join(ids)
    if not repaired.get("enr_context_warning"):
        repaired["enr_context_warning"] = (
            "LLM used nearby context in non-context fields but omitted "
            "structured context fields; provenance inferred from prompt context."
        )
    return _normalize_context_consistency(repaired)


def _move_context_only_facts(
    enrichment: dict[str, str],
    primary_text: str,
    context_text: str,
) -> list[str]:
    """Move key facts and keywords only the nearby context supports, in place.

    A value is context-only when it names a number or distinctive phrase that
    the context contains and the primary item does not. Moved key facts are
    appended to context_key_facts; a moved keyword is appended too unless a
    context fact already carries its evidence. Returns the evidence that
    justified each move, for provenance.
    """
    primary_numbers = _number_haystack(primary_text)
    primary_totals = _amount_totals(primary_text)
    context_numbers = _number_haystack(context_text)
    primary_text_haystack = _text_haystack(primary_text)
    context_text_haystack = _text_haystack(context_text)

    def context_only(value: str) -> list[str]:
        amounts = set(_amounts(value))
        numbers = [
            number
            for number in _evidence_numbers(value)
            if number not in primary_numbers
            and number in context_numbers
            and not (number in amounts and number in primary_totals)
        ]
        if numbers:
            return numbers
        if _numbers(_DATE_TIME_RE.sub(" ", value)):
            return []

        normalized = _normalize_text_evidence(value)
        if (
            len(normalized) >= _MIN_TEXT_EVIDENCE_LEN
            and normalized in context_text_haystack
            and normalized not in primary_text_haystack
        ):
            return [normalized]
        return []

    evidence: list[str] = []
    moved_facts: list[str] = []
    facts = _fact_list(enrichment.get("enr_key_facts", "")) or []
    kept_facts = []
    for fact in facts:
        fact_evidence = context_only(fact)
        if fact_evidence:
            moved_facts.append(fact)
            evidence.extend(fact_evidence)
        else:
            kept_facts.append(fact)

    # Split on the separator _normalize_metadata_list joins with, so the
    # keywords that stay are rejoined exactly as stored.
    moved_keywords: list[tuple[str, list[str]]] = []
    kept_keywords = []
    for keyword in (enrichment.get("enr_keywords") or "").split(", "):
        keyword_evidence = context_only(keyword)
        if keyword_evidence:
            moved_keywords.append((keyword, keyword_evidence))
            evidence.extend(keyword_evidence)
        else:
            kept_keywords.append(keyword)

    if not evidence:
        return evidence

    existing = (enrichment.get("enr_context_key_facts") or "").strip()
    context_facts = _fact_list(existing)
    if context_facts is None:
        context_facts = [existing] if existing else []
    context_facts.extend(fact for fact in moved_facts if fact not in context_facts)
    carried_text = _text_haystack(" ".join(context_facts))
    carried_numbers = _number_haystack(" ".join(context_facts))
    for keyword, keyword_evidence in moved_keywords:
        if all(
            evidence_item in carried_numbers or evidence_item in carried_text
            for evidence_item in keyword_evidence
        ):
            continue
        context_facts.append(keyword)

    if moved_facts:
        enrichment["enr_key_facts"] = json.dumps(kept_facts)
    if moved_keywords:
        enrichment["enr_keywords"] = ", ".join(kept_keywords)
    enrichment["enr_context_key_facts"] = json.dumps(context_facts)
    logger.info(
        "Moved %d key fact(s) and %d keyword(s) that only nearby context supports "
        "to context_key_facts (evidence: %s)",
        len(moved_facts),
        len(moved_keywords),
        ", ".join(dict.fromkeys(evidence)),
    )
    return evidence


def _fact_list(value: str) -> list[str] | None:
    """The facts in a stored JSON-array field, or None if it is not one."""
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, list):
        return None
    return [str(item) for item in parsed]


def _normalize_number(number: str) -> str:
    """Spell a number one way: no thousands separator, no trailing zero cents."""
    number = number.replace(",", "")
    if "." in number:
        number = number.rstrip("0").rstrip(".")
    return number


def _numbers(text: str) -> list[str]:
    return [_normalize_number(match.group(0)) for match in _NUMBER_RE.finditer(text)]


def _amounts(text: str) -> list[str]:
    """Dollar amounts in ``text``, normalized like ``_numbers``."""
    return [_normalize_number(match.group(1)) for match in _AMOUNT_RE.finditer(text)]


def _amount_totals(text: str) -> set[str]:
    """Totals of two or three of the first dollar amounts in ``text``.

    A repeated amount is kept: two $10 fees are part of the total.
    """
    cents = [round(float(amount) * 100) for amount in _amounts(text)[:_MAX_AMOUNTS_TO_SUM]]
    return {
        _normalize_number(f"{total // 100}.{total % 100:02d}")
        for size in range(2, _MAX_SUMMED_AMOUNTS + 1)
        for total in map(sum, itertools.combinations(cents, size))
    }


def _number_haystack(text: str) -> str:
    """Numbers in ``text`` joined by spaces; ``number in haystack`` also finds a
    number printed inside a longer one, such as a card tail in a full number."""
    return " ".join(_numbers(text or ""))


def _evidence_numbers(value: str) -> list[str]:
    """Numbers in ``value`` specific enough to show which text it came from."""
    return [
        number
        for number in _numbers(_DATE_TIME_RE.sub(" ", value))
        if sum(char.isdigit() for char in number) >= _MIN_EVIDENCE_DIGITS
    ]


def _normalize_text_evidence(value: str) -> str:
    """Lowercase phrase with collapsed whitespace for substring matching."""
    return re.sub(r"\s+", " ", (value or "").lower()).strip()


def _text_haystack(text: str) -> str:
    """Searchable lowercase text with dates/times stripped."""
    return _normalize_text_evidence(_DATE_TIME_RE.sub(" ", text or ""))


def _has_context_fields(enrichment: dict[str, str]) -> bool:
    return any(
        enrichment.get(f"enr_{raw_key}")
        for raw_key in _CONTEXT_KEYS_RAW
        if raw_key.startswith("context_")
    )


def _normalize_context_consistency(enrichment: dict[str, str]) -> dict[str, str]:
    context_text = " ".join(
        enrichment.get(key, "")
        for key in (
            "enr_context_relationship",
            "enr_context_warning",
        )
    ).lower()
    if enrichment.get("enr_context_confidence", "").lower() == "high" and any(
        term in context_text for term in _CONTEXT_AMBIGUITY_TERMS
    ):
        enrichment = dict(enrichment)
        enrichment["enr_context_confidence"] = "ambiguous"
        if not enrichment.get("enr_context_warning"):
            enrichment["enr_context_warning"] = (
                "Nearby context is ambiguous or conflicting; verify before relying on it."
            )
    return enrichment


def _metadata_values(value: str) -> list[str]:
    value = (value or "").strip()
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in value.split(",") if item.strip()]


def _mentions_nearby_context(enrichment: dict[str, str]) -> bool:
    text = " ".join(
        enrichment.get(key, "")
        for key in (
            "enr_summary",
            "enr_key_facts",
            "enr_entities_people",
            "enr_entities_places",
            "enr_entities_orgs",
            "enr_entities_dates",
            "enr_topics",
            "enr_keywords",
            "enr_suggested_tags",
            "enr_suggested_folder",
        )
    ).lower()
    return any(
        phrase in text
        for phrase in (
            "nearby context",
            "provided context",
            "nearby message",
            "nearby messages",
            "context identifies",
            "as indicated by context",
            "as indicated by nearby",
            "based on context",
            "from context",
        )
    )


def _context_source_ids_for_values(context_text: str, values: list[str]) -> list[str]:
    matching_ids: list[str] = []
    lowered_values = [value.lower() for value in values if value]
    for line in context_text.splitlines():
        if lowered_values and not any(value in line.lower() for value in lowered_values):
            continue
        for message_id in _context_message_ids(line):
            if message_id not in matching_ids:
                matching_ids.append(message_id)
    if matching_ids:
        return matching_ids

    ids: list[str] = []
    for message_id in _context_message_ids(context_text):
        if message_id not in ids:
            ids.append(message_id)
    return ids


def _context_message_ids(text: str) -> list[str]:
    return [
        match.group(1).strip()
        for match in re.finditer(r"\b(?:source_message_id|message_id)=([^\]\s]+)", text)
        if match.group(1).strip()
    ]


def apply_doc_type_vocabulary(
    enrichment: dict[str, str],
    taxonomy_store: "TaxonomyStore | None" = None,
    *,
    existing_doc_type: str = "",
) -> tuple[dict[str, str], dict[str, int]]:
    """Constrain ``enr_doc_type`` to the taxonomy vocabulary and keep sticky labels.

    Returns the updated enrichment plus counters:
    ``unknown`` (labels mapped away) and ``disagreement`` (kept existing over new).
    """
    counters = {"unknown": 0, "disagreement": 0}
    alias_map = vocabulary_alias_map(taxonomy_store)
    constrained = constrain_doc_type(enrichment.get("enr_doc_type", ""), alias_map)
    enrichment = dict(enrichment)
    enrichment["enr_doc_type"] = constrained.value
    counters["unknown"] = constrained.unknown_count
    if constrained.rejected:
        logger.info(
            "doc_type vocabulary rejected %d label(s): %s",
            len(constrained.rejected),
            ", ".join(constrained.rejected[:8]),
        )

    final, disagreed = reconcile_doc_type(
        existing=existing_doc_type,
        proposed=enrichment["enr_doc_type"],
    )
    enrichment["enr_doc_type"] = final
    if disagreed:
        counters["disagreement"] = 1
        logger.info(
            "doc_type disagreement kept existing=%r rejected_new=%r",
            final,
            constrained.value,
        )
    return enrichment, counters


def enrich_document(
    text: str,
    title: str,
    source_type: str,
    generator: "LLMGenerator",
    max_input_chars: int = 4000,
    max_output_tokens: int = 512,
    taxonomy_store: "TaxonomyStore | None" = None,
    context_text: str = "",
    record_taxonomy_usage: bool = True,
    postprocess_enrichment: bool = False,
    postprocess_rules: Iterable[str] | None = None,
    existing_doc_type: str = "",
) -> dict[str, str]:
    """Extract structured metadata from document text using an LLM.

    Returns a dict with all ENRICHMENT_FIELDS populated (or empty strings
    on failure).  Never raises — logs warnings on parse errors.

    ``existing_doc_type`` is the previously stored facet value: when a fresh
    enrichment disagrees, the existing label is kept and a disagreement
    counter is recorded on the result (``_doc_type_disagreement``).
    """
    with _tracer.start_as_current_span("enrich"):
        if not text or not text.strip():
            logger.debug("Skipping enrichment for empty document: %s", title)
            return empty_enrichment()

        input_hash = enrichment_input_hash(
            text=text,
            title=title,
            source_type=source_type,
            context_text=context_text,
            max_input_chars=max_input_chars,
        )

        if len(text) <= max_input_chars:
            truncated = text
        else:
            # Head + tail sampling: capture both opening context and late-document
            # conclusions/facts that a simple head truncation would miss.
            half = max_input_chars // 2
            truncated = text[:half] + "\n\n[...]\n\n" + text[-half:]

        # Build taxonomy context block for the prompt
        taxonomy_block = ""
        if taxonomy_store is not None:
            try:
                raw_block = taxonomy_store.format_for_prompt(
                    query=f"{title}\n{truncated}",
                    max_chars=96000,
                )
                if raw_block:
                    taxonomy_block = f"\n{_TAXONOMY_INSTRUCTION}\n{raw_block}\n"
            except Exception as exc:
                logger.warning("Failed to load taxonomy for prompt: %s", exc)

        normalized_context_text = (context_text or "").strip()
        template = _CONTEXT_PROMPT_TEMPLATE if normalized_context_text else _PROMPT_TEMPLATE
        prompt = template.format(
            modality_instructions=_MODALITY_INSTRUCTIONS,
            title=title,
            source_type=source_type,
            text=truncated,
            taxonomy_block=taxonomy_block,
            context_text=normalized_context_text,
        )

        try:
            raw_response = generator.generate(prompt, max_tokens=max_output_tokens)
            logger.debug(
                "LLM enrichment response received for '%s' (%d chars)",
                title,
                len(raw_response),
            )

            contract_errors = enrichment_contract_errors(raw_response)
            if contract_errors and contract_errors != ["invalid_json"]:
                logger.warning(
                    "LLM structured output for '%s' violates enrichment contract: %s",
                    title,
                    ", ".join(contract_errors),
                )
                return failed_enrichment(
                    "structured_output_contract_violation: "
                    + ", ".join(contract_errors)
                )

            enrichment = parse_enrichment_response(raw_response)
            enrichment = _repair_context_omissions(enrichment, truncated, context_text)
            enrichment, card_corrections = ground_card_suffixes(
                enrichment,
                source_text=f"{truncated}\n{normalized_context_text}",
            )
            for correction in card_corrections:
                logger.warning(
                    "Ungrounded card suffix in enrichment for '%s': %s %s -> %s",
                    title,
                    correction.field,
                    correction.claimed,
                    correction.corrected or "(dropped)",
                )
            enrichment = repair_enrichment(
                enrichment,
                text=truncated,
                title=title,
                source_type=source_type,
                enabled=postprocess_enrichment,
                enabled_rules=postprocess_rules,
            )
            enrichment, vocab_counters = apply_doc_type_vocabulary(
                enrichment,
                taxonomy_store,
                existing_doc_type=existing_doc_type,
            )
            if vocab_counters["unknown"]:
                enrichment["_doc_type_unknown"] = str(vocab_counters["unknown"])
            if vocab_counters["disagreement"]:
                enrichment["_doc_type_disagreement"] = "1"

            missing_required = missing_required_fields(enrichment)
            if missing_required:
                logger.warning(
                    "LLM structured output for '%s' is missing required fields: %s",
                    title,
                    ", ".join(missing_required),
                )
                return failed_enrichment(
                    "structured_output_missing_required_fields: "
                    + ", ".join(missing_required)
                )

            # Increment usage_count for matched taxonomy entries
            if taxonomy_store is not None and record_taxonomy_usage:
                try:
                    for tag in (enrichment.get("enr_suggested_tags") or "").split(","):
                        tag = tag.strip()
                        if tag:
                            taxonomy_store.increment_usage(f"tag:{tag}")
                    folder = (enrichment.get("enr_suggested_folder") or "").strip()
                    if folder:
                        taxonomy_store.increment_usage(f"folder:{folder}")
                    for doc_type in (enrichment.get("enr_doc_type") or "").split(","):
                        doc_type = doc_type.strip()
                        if doc_type:
                            taxonomy_store.increment_usage(f"doc_type:{doc_type}")
                except Exception as exc:
                    logger.warning("Failed to increment taxonomy usage: %s", exc)

            enrichment[ENRICHMENT_INPUT_HASH_FIELD] = input_hash
            logger.info(
                "Enriched '%s': doc_type=%s, topics=%s",
                title,
                enrichment.get("enr_doc_type", ""),
                enrichment.get("enr_topics", "")[:80],
            )
            return enrichment

        except json.JSONDecodeError as exc:
            logger.warning(
                "Failed to parse LLM JSON for '%s': %s",
                title,
                exc,
            )
            return failed_enrichment(f"json_parse_error: {exc}")
        except Exception as exc:
            logger.error(
                "LLM enrichment failed for '%s': %s: %s", title, type(exc).__name__, exc,
            )
            return failed_enrichment(
                f"{type(exc).__name__}: {exc}", transient=is_transient(exc)
            )
