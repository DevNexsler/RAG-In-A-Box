"""Policy for degraded-ledger entries: retrying vs terminal.

The degraded ledger holds two populations that look identical in the file and
mean opposite things to an operator:

  * **retrying** — the doc is still in the self-heal lane; a later run can fix
    it with no human involved.
  * **terminal** — the doc exhausted its budget (doc-specific attempts, or an
    unchanged upstream provider-error artifact). No indexer run can change it;
    the source or a human must.

Both the indexing flow and the health probe have to make that distinction, so
it lives here rather than being re-derived at each seam — a terminal state
reported in transient language ("retry_pending") tells an operator to wait for
a retry that is no longer scheduled (#2101).
"""

# Doc-specific failures charge `attempts`; a doc is abandoned once it reaches
# this cap. Raised from 5 to 12 alongside exponential backoff (dpark 2026-07-28):
# with retries now spaced out, more of them span a long time without hammering,
# so a genuinely-flaky doc gets more chances before we give up.
MAX_ATTEMPTS = 12

# Stored provider-error artifacts cannot heal through another indexer pass: the
# upstream producer must replace their bytes. Give that producer a short grace
# window, then park an unchanged artifact instead of retrying it forever.
MAX_BLOCKED_ATTEMPTS = 3

BLOCKED_UPSTREAM_REASON_SUFFIX = ":blocked_on_upstream"


def _counter(entry: dict, field: str) -> int:
    try:
        return int(entry.get(field, 0))
    except (TypeError, ValueError):
        return 0


def is_terminal_entry(entry: dict) -> bool:
    """Whether a degraded entry is parked for good (manual action required)."""
    if not isinstance(entry, dict):
        return False
    return (
        _counter(entry, "attempts") >= MAX_ATTEMPTS
        or _counter(entry, "blocked_attempts") >= MAX_BLOCKED_ATTEMPTS
    )


def partition_degraded_docs(ledger: dict) -> tuple[set[str], set[str]]:
    """Split a degraded ledger's ids into (still retrying, terminal)."""
    retrying: set[str] = set()
    terminal: set[str] = set()
    docs = ledger.get("docs", {}) if isinstance(ledger, dict) else {}
    if not isinstance(docs, dict):
        return retrying, terminal
    for doc_id, entry in docs.items():
        target = terminal if is_terminal_entry(entry) else retrying
        target.add(str(doc_id))
    return retrying, terminal


def terminal_degraded_docs(ledger: dict) -> dict[str, list[str]]:
    """Return terminal degraded doc ids grouped by reason.

    Mirrors ``core.skip_policy.actionable_skip_docs`` — both describe the same
    class of standing operator action, so they are reported the same way.
    """
    grouped: dict[str, list[str]] = {}
    docs = ledger.get("docs", {}) if isinstance(ledger, dict) else {}
    if not isinstance(docs, dict):
        return grouped
    for doc_id, entry in docs.items():
        if not is_terminal_entry(entry):
            continue
        reasons = entry.get("reasons", [])
        if not isinstance(reasons, list) or not reasons:
            reasons = ["unknown"]
        for reason in reasons:
            grouped.setdefault(str(reason), []).append(str(doc_id))
    return {reason: sorted(set(doc_ids)) for reason, doc_ids in sorted(grouped.items())}
