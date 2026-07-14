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
    *,
    empty_is_clean: bool = False,
) -> str:
    """Return enrichment text, or "" for a fallback-CONFIRMED blank.

    Contract for both callables: RAISE a transient error (satisfying
    core.resilience.is_transient) when the provider is unreachable; RETURN "" when it
    is reachable but produced nothing.

    - primary raises           -> propagates (unreachable; fallback NOT called)
    - primary returns text      -> returned as-is
    - primary reachable-empty:
        - fallback is None:
            - empty_is_clean=False -> raise TransientError (ambiguous empty -> retry).
              Use for describe/media, where an empty result may be a starved provider.
            - empty_is_clean=True  -> return "" (presumptively clean). Use for OCR text
              extraction, where an empty page is normally just a textless page, not an
              outage — an unreachable extract still RAISES (handled above), so only the
              reachable-empty case is treated as clean, and it must not perpetually
              re-process a blank page.
        - fallback returns text  -> returned (RECOVERED)
        - fallback returns ""    -> return "" (CONFIRMED BLANK -> caller treats as clean)
        - fallback raises        -> propagates (both down -> retry)
    """
    text = primary_call()
    if text.strip():
        return text
    if fallback_call is None:
        if empty_is_clean:
            return ""
        raise TransientError(
            "enrichment empty and no fallback configured (unconfirmed blank)"
        )
    recovered = fallback_call()
    return recovered if recovered.strip() else ""
