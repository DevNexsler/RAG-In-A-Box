from __future__ import annotations

import json
import re
from collections.abc import Iterable

_TOKEN_RE = re.compile(r"[a-z0-9$,.#/-]+", re.IGNORECASE)
_LABEL_SEPARATOR_RE = re.compile(r"[-_ ]+")
_DATE_RE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b")
_MONEY_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d{2})?")

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "was",
    "with",
}

_GENERIC_FACT_PHRASES = (
    "document contains important information",
    "contains important information",
    "important information",
    "document is about",
    "document discusses",
    "document provides",
    "no specific",
)

_PAYMENT_TERMS = (
    "balance",
    "bill",
    "due",
    "failed payment",
    "fee",
    "invoice",
    "ledger",
    "overdue",
    "payment",
    "rent",
    "tenantcloud",
    "unpaid",
)
_LEASE_TERMS = ("agreement", "lease", "renewal", "sign", "signature")
_RENTAL_INQUIRY_TERMS = (
    "application question",
    "rental",
    "renter",
    "showing",
    "tour",
    "zillow rental manager",
)
_LISTING_TERMS = ("coming soon", "listing", "open house", "price cut", "saved search")
_LEGAL_TERMS = ("court", "eviction", "legal", "notice to quit", "summons")
_DELIVERY_TERMS = ("delivery", "package", "pickup", "shipping", "tracking", "ups", "usps")
_ESTIMATE_TERMS = ("estimate", "proposal", "quote", "scope of work")
_ACTION_TERMS = (
    "action required",
    "appointment",
    "call",
    "deadline",
    "due",
    "failed",
    "follow up",
    "must",
    "overdue",
    "request",
    "requested",
    "requires",
    "schedule",
    "sign",
    "tour",
)
_GENERIC_DOC_TYPES = {
    "document",
    "email",
    "image",
    "message",
    "note",
    "notification",
    "text",
}
_DEFAULT_RULES = {"importance", "doc_type", "key_facts"}

# Compounds the enrichment model spells both as one word and as two. #1251's
# separator fold cannot reconcile these — they differ in word count, not in
# punctuation — so `followup` and `follow_up` stay two buckets no single LIKE
# filter value reaches (#1330). The canonical form is the multi-word spelling:
# it is what the term reads as in English, it keeps the boundaries a tokenizer
# and a published facet need, and deleting separators instead would leave
# unreadable values like `rentalinquiry`. Add a compound here, not a whole label
# — the fold applies wherever the compound appears, so `leasing_followup` and
# the next prefix nobody has seen yet are both covered by one entry.
_DOC_TYPE_COMPOUNDS = (
    "follow_up",
    "health_check",
    "pay_stub",
    "section_8",
    "w_9",
)
_COMPOUND_SEGMENTS = {
    compound.replace("_", ""): tuple(compound.split("_")) for compound in _DOC_TYPE_COMPOUNDS
}
_COMPOUND_MAX_WORDS = max(len(words) for words in _COMPOUND_SEGMENTS.values())


def canonicalize_doc_type(value: str) -> str:
    """Fold an ``enr_doc_type`` field to its canonical spelling.

    ``enr_doc_type`` is a published filter key (``file_search``/``file_facets``)
    matched with LIKE, so a separator or case difference silently splits one
    concept into several unreachable buckets (#1251, recurrence of #0233).
    Each comma-separated label is lowercased and every run of ``-``, ``_`` or
    space becomes a single ``_``; duplicates created by the fold collapse and
    the model's ordering is preserved. Words that spell one of
    ``_DOC_TYPE_COMPOUNDS`` are then re-segmented to that compound's canonical
    form, which closes the word-segmentation variants the separator fold leaves
    behind (``followup`` -> ``follow_up``, #1330).
    """
    canonical: list[str] = []
    seen: set[str] = set()
    for label in _csv_values(value):
        label = _canonical_label(label)
        if label and label not in seen:
            canonical.append(label)
            seen.add(label)
    return ", ".join(canonical)


def _canonical_label(label: str) -> str:
    label = _LABEL_SEPARATOR_RE.sub("_", label.strip().lower()).strip("_")
    if not label:
        return label
    return "_".join(_resegment_words(label.split("_")))


def _resegment_words(words: list[str]) -> list[str]:
    """Rewrite each run of words spelling a known compound to its canonical form.

    Longest run first, so a compound is matched whether the model wrote it joined
    (``followup``) or already split (``follow``, ``up``); words outside the
    vocabulary are left exactly as they are.
    """
    segmented: list[str] = []
    start = 0
    while start < len(words):
        for end in range(min(start + _COMPOUND_MAX_WORDS, len(words)), start, -1):
            canonical = _COMPOUND_SEGMENTS.get("".join(words[start:end]))
            if canonical is not None:
                segmented.extend(canonical)
                start = end
                break
        else:
            segmented.append(words[start])
            start += 1
    return segmented


def repair_enrichment(
    enrichment: dict[str, str],
    *,
    text: str,
    title: str,
    source_type: str,
    enabled: bool = False,
    enabled_rules: Iterable[str] | None = None,
) -> dict[str, str]:
    repaired = dict(enrichment)
    if not enabled:
        return repaired
    rules = _rule_set(enabled_rules)

    source_text = _document_text(text)
    corpus = "\n".join(
        part for part in (title, source_type, source_text, _metadata_corpus(repaired)) if part
    )
    corpus_lower = corpus.lower()

    if "doc_type" in rules:
        repaired["enr_doc_type"] = _repair_doc_type(
            current=repaired.get("enr_doc_type", ""),
            corpus_lower=corpus_lower,
        )
    if "importance" in rules:
        repaired["enr_importance"] = _repair_importance(
            current=repaired.get("enr_importance", ""),
            corpus_lower=corpus_lower,
        )
    if "key_facts" in rules:
        repaired["enr_key_facts"] = _repair_key_facts(
            current=repaired.get("enr_key_facts", ""),
            corpus_lower=corpus_lower,
        )
    return repaired


def _document_text(text: str) -> str:
    marker = "Document text:"
    if marker in text:
        text = text.split(marker, 1)[1]
    context_marker = "NEARBY SAME-CHANNEL CONTEXT CANDIDATES"
    if context_marker in text:
        text = text.split(context_marker, 1)[0]
    return text.strip()


def _rule_set(enabled_rules: Iterable[str] | None) -> set[str]:
    if enabled_rules is None:
        return set(_DEFAULT_RULES)
    if isinstance(enabled_rules, str):
        return {rule.strip() for rule in enabled_rules.split(",") if rule.strip()}
    return {str(rule).strip() for rule in enabled_rules if str(rule).strip()}


def _metadata_corpus(enrichment: dict[str, str]) -> str:
    keys = (
        "enr_summary",
        "enr_doc_type",
        "enr_topics",
        "enr_keywords",
        "enr_suggested_tags",
        "enr_suggested_folder",
    )
    return " ".join(enrichment.get(key, "") for key in keys)


def _repair_doc_type(*, current: str, corpus_lower: str) -> str:
    values = _csv_values(canonicalize_doc_type(current))
    existing = set(values)
    if values and any(value not in _GENERIC_DOC_TYPES for value in values):
        return ", ".join(values[:5])

    inferred: list[str] = []
    inferred_seen: set[str] = set()

    def add(value: str) -> None:
        value = _canonical_label(value)
        if value not in existing and value not in inferred_seen:
            inferred.append(value)
            inferred_seen.add(value)

    if _has_any(corpus_lower, _LEGAL_TERMS):
        add("legal notice")
    if _has_any(corpus_lower, _PAYMENT_TERMS) and _has_any(
        corpus_lower, ("failed", "due", "overdue", "balance", "payment")
    ):
        add("payment notice")
    if _has_any(corpus_lower, _LEASE_TERMS):
        add("lease administration")
    if "zillow" in corpus_lower and _has_any(corpus_lower, _LISTING_TERMS):
        add("listing report")
    if _has_any(corpus_lower, _RENTAL_INQUIRY_TERMS):
        add("rental inquiry")
    if _has_any(corpus_lower, _DELIVERY_TERMS):
        add("delivery notification")
    if _has_any(corpus_lower, _ESTIMATE_TERMS):
        add("estimate")

    return ", ".join((values + inferred[:1])[:5])


def _repair_importance(*, current: str, corpus_lower: str) -> str:
    try:
        current_value = max(0.0, min(1.0, float(current)))
    except (TypeError, ValueError):
        current_value = 0.5

    heuristic = 0.5
    if _has_any(corpus_lower, _LEGAL_TERMS):
        heuristic = 0.9
    elif _has_any(corpus_lower, ("failed", "overdue", "unpaid")) and _has_any(
        corpus_lower, _PAYMENT_TERMS
    ):
        heuristic = 0.85
    elif _has_any(corpus_lower, _LEASE_TERMS) and _has_any(
        corpus_lower, ("due", "must", "sign", "renewal")
    ):
        heuristic = 0.8
    elif _has_any(corpus_lower, _PAYMENT_TERMS) and (
        _MONEY_RE.search(corpus_lower) or "due" in corpus_lower
    ):
        heuristic = 0.75
    elif _has_any(corpus_lower, _ACTION_TERMS):
        heuristic = 0.7
    elif _has_any(corpus_lower, _ESTIMATE_TERMS + _DELIVERY_TERMS):
        heuristic = 0.6

    return str(round(max(current_value, heuristic), 2))


def _repair_key_facts(*, current: str, corpus_lower: str) -> str:
    facts: list[str] = []
    for fact in _fact_values(current):
        if _supported_fact(fact=fact, corpus_lower=corpus_lower):
            facts.append(_trim_fact(fact))

    return json.dumps(facts[:6])


def _fact_values(value: str) -> list[str]:
    value = (value or "").strip()
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [value]


def _supported_fact(*, fact: str, corpus_lower: str) -> bool:
    fact_lower = fact.lower().strip()
    if not fact_lower or any(phrase in fact_lower for phrase in _GENERIC_FACT_PHRASES):
        return False
    if _MONEY_RE.search(fact) or _DATE_RE.search(fact):
        return True
    tokens = [token for token in _tokens(fact_lower) if token not in _STOPWORDS]
    overlap = sum(1 for token in tokens if token in corpus_lower)
    return overlap >= 2


def _trim_fact(value: str) -> str:
    value = " ".join(value.split()).strip()
    if len(value) <= 180:
        return value
    return value[:177].rstrip() + "..."


def _csv_values(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _tokens(value: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(value)]


def _has_any(value: str, terms: tuple[str, ...]) -> bool:
    return any(term in value for term in terms)
