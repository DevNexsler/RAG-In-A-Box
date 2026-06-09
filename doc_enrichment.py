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
import json
import logging
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from core.enrichment_postprocess import repair_enrichment

if TYPE_CHECKING:
    from providers.llm import LLMGenerator
    from taxonomy_store import TaxonomyStore

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
Extract metadata from this document. Respond with ONLY valid JSON, no other text.

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
If you use nearby context in summary, entities, topics, keywords, key_facts, tags, folder, or importance, you MUST also fill the matching context_* fields.
Do not place context-derived facts only in non-context fields.
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
For "suggested_tags" and "suggested_folder": use the taxonomy below.
Pick the most relevant tags from Available Tags (you may also add new ones).
Pick the single best matching folder path from Available Folders (use the exact path).
"""

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
_SCHEMA_PLACEHOLDER_VALUES.update({"type1", "type2"})

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


def failed_enrichment(reason: str) -> dict[str, str]:
    """Return an enrichment dict that signals failure with a reason.

    The caller should check for ``_enrichment_failed`` and remove it
    before storing in LanceDB.
    """
    result = empty_enrichment()
    result["_enrichment_failed"] = reason
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


def _repair_context_omissions(
    enrichment: dict[str, str],
    primary_text: str,
    context_text: str,
) -> dict[str, str]:
    """Preserve provenance when a model uses context but omits context_* fields."""
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

    context_used = bool(copied_values) or _mentions_nearby_context(repaired)
    if not context_used:
        return _normalize_context_consistency(repaired)

    if had_context_fields and not copied_values:
        return _normalize_context_consistency(repaired)

    if not repaired.get("enr_context_confidence"):
        repaired["enr_context_confidence"] = "medium"
    if not repaired.get("enr_context_relationship"):
        repaired["enr_context_relationship"] = "llm_used_nearby_context"
    if not repaired.get("enr_context_source_message_ids"):
        ids = _context_source_ids_for_values(context_text, copied_values)
        repaired["enr_context_source_message_ids"] = ", ".join(ids)
    if not repaired.get("enr_context_warning"):
        repaired["enr_context_warning"] = (
            "LLM used nearby context in non-context fields but omitted "
            "structured context fields; provenance inferred from prompt context."
        )
    return _normalize_context_consistency(repaired)


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
) -> dict[str, str]:
    """Extract structured metadata from document text using an LLM.

    Returns a dict with all ENRICHMENT_FIELDS populated (or empty strings
    on failure).  Never raises — logs warnings on parse errors.
    """
    if not text or not text.strip():
        logger.debug("Skipping enrichment for empty document: %s", title)
        return empty_enrichment()

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
            raw_block = taxonomy_store.format_for_prompt()
            if raw_block:
                taxonomy_block = f"\n{_TAXONOMY_INSTRUCTION}\n{raw_block}\n"
        except Exception as exc:
            logger.warning("Failed to load taxonomy for prompt: %s", exc)

    normalized_context_text = (context_text or "").strip()
    template = _CONTEXT_PROMPT_TEMPLATE if normalized_context_text else _PROMPT_TEMPLATE
    prompt = template.format(
        title=title,
        source_type=source_type,
        text=truncated,
        taxonomy_block=taxonomy_block,
        context_text=normalized_context_text,
    )

    try:
        raw_response = generator.generate(prompt, max_tokens=max_output_tokens)
        logger.debug("LLM enrichment raw response for '%s': %s", title, raw_response[:200])

        enrichment = parse_enrichment_response(raw_response)
        enrichment = _repair_context_omissions(enrichment, truncated, context_text)
        enrichment = repair_enrichment(
            enrichment,
            text=truncated,
            title=title,
            source_type=source_type,
            enabled=postprocess_enrichment,
            enabled_rules=postprocess_rules,
        )

        if not enrichment.get("enr_summary"):
            logger.warning(
                "LLM returned empty summary for '%s'. Raw response: %s",
                title, raw_response[:300],
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
            except Exception as exc:
                logger.warning("Failed to increment taxonomy usage: %s", exc)

        logger.info(
            "Enriched '%s': doc_type=%s, topics=%s",
            title,
            enrichment.get("enr_doc_type", ""),
            enrichment.get("enr_topics", "")[:80],
        )
        return enrichment

    except json.JSONDecodeError as exc:
        logger.warning(
            "Failed to parse LLM JSON for '%s': %s. Response: %s",
            title, exc, raw_response[:300] if "raw_response" in dir() else "N/A",
        )
        return failed_enrichment(f"json_parse_error: {exc}")
    except Exception as exc:
        logger.error(
            "LLM enrichment failed for '%s': %s: %s", title, type(exc).__name__, exc,
        )
        return failed_enrichment(f"{type(exc).__name__}: {exc}")
