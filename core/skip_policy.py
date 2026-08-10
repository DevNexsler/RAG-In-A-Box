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
