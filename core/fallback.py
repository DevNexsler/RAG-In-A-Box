"""Modality-agnostic transient-vs-blank disambiguation for enrichment providers.

One rule reused by every fallback wrapper (OCR describe/extract, media video/audio):
consult an independent fallback ONLY when the primary is reachable-but-empty, so a
provider outage never fans out into paid calls and never caps a doc.
"""
from __future__ import annotations

from typing import Callable

from core.resilience import TransientError


def resolve_with_fallback(
    primary_call: Callable[[], str],
    fallback_call: Callable[[], str] | None,
) -> str:
    """Return enrichment text, or "" for a fallback-CONFIRMED blank.

    Contract for both callables: RAISE a transient error (satisfying
    core.resilience.is_transient) when the provider is unreachable; RETURN "" when it
    is reachable but produced nothing.

    - primary raises           -> propagates (unreachable; fallback NOT called)
    - primary returns text      -> returned as-is
    - primary reachable-empty:
        - fallback is None       -> raise TransientError (unconfirmed empty -> retry)
        - fallback returns text  -> returned (RECOVERED)
        - fallback returns ""    -> return "" (CONFIRMED BLANK -> caller treats as clean)
        - fallback raises        -> propagates (both down -> retry)
    """
    text = primary_call()
    if text.strip():
        return text
    if fallback_call is None:
        raise TransientError(
            "enrichment empty and no fallback configured (unconfirmed blank)"
        )
    recovered = fallback_call()
    return recovered if recovered.strip() else ""
