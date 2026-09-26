"""Policy for skip-ledger outcomes requiring source-owner action."""

CORRUPT_MANGLED_BINARY = "corrupt_mangled_binary"

ACTIONABLE_PERMANENT_SKIP_REASONS = frozenset({CORRUPT_MANGLED_BINARY})


def actionable_skip_docs(ledger: dict) -> dict[str, list[str]]:
    """Return permanent actionable skip IDs grouped by reason."""
    grouped: dict[str, list[str]] = {}
    docs = ledger.get("docs", {}) if isinstance(ledger, dict) else {}
    if not isinstance(docs, dict):
        return grouped
    for doc_id, entry in docs.items():
        if not isinstance(entry, dict):
            continue
        reasons = entry.get("reasons", [])
        if not isinstance(reasons, list):
            continue
        for reason in reasons:
            if reason in ACTIONABLE_PERMANENT_SKIP_REASONS:
                grouped.setdefault(reason, []).append(str(doc_id))
    return {reason: sorted(set(doc_ids)) for reason, doc_ids in sorted(grouped.items())}


def is_permanent_skip_entry(entry: dict) -> bool:
    """Whether unchanged source bytes must remain parked without timed retry."""
    reasons = entry.get("reasons", []) if isinstance(entry, dict) else []
    return isinstance(reasons, list) and bool(
        ACTIONABLE_PERMANENT_SKIP_REASONS.intersection(reasons)
    )


#: Prefix of the skip reason the dedupe gate records for a copy whose content
#: is carried by another registry row.
DEDUPE_SKIP_REASON_PREFIX = "duplicate_of:"


def content_terminal_skip_reasons(entry: dict) -> list[str]:
    """Skip reasons that are a verdict about the document's own bytes.

    A skip recorded because the document deduped against a canonical says
    nothing about whether these bytes can be indexed — the content is in the
    canonical's rows — so reading it as evidence about the content is circular.
    Every other skip reason (no text extracted, unreadable or encrypted PDF,
    retrieval stub, quarantine, corruption, terminal processing error) means
    the same thing: while these bytes are unchanged this document will not
    produce an index row.

    Unknown future reasons are therefore treated as verdicts, which is the safe
    direction — it keeps a document that cannot land content out of the dedupe
    canonical election (#2097) rather than letting the next new reason
    reintroduce a canonical no duplicate can ever resolve to.
    """
    reasons = entry.get("reasons", []) if isinstance(entry, dict) else []
    if not isinstance(reasons, list):
        return []
    return sorted(
        {
            reason
            for reason in reasons
            if isinstance(reason, str)
            and reason
            and not reason.startswith(DEDUPE_SKIP_REASON_PREFIX)
        }
    )
