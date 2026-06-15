"""Guards for the shadow-rebuild decision — an expensive-to-get-wrong path.

A shadow rebuild reprocesses the ENTIRE corpus (re-OCR, re-enrich, re-embed
every doc) into a fresh table. It must NEVER be auto-inferred from diff size:

- A large incremental change (bulk add, mass edit) must upsert in place — only
  the changed docs are processed and search stays live. Reprocessing untouched
  docs is pure waste (the 2026-06-14 incident: a routine refresh + a large
  degraded ledger reprocessed all 27k docs).
- A config/embedding-model change that genuinely needs a full reindex does NOT
  change file mtimes, so it could never be diff-detected anyway.

So shadow fires ONLY on an explicit request (safety.force_full_rebuild) against
a populated table. No diff shape, scheduled refresh, partial scan, or self-heal
can escalate to a full-corpus reprocess.
"""

from flow_index_vault import _should_use_shadow_rebuild


# --- the load-bearing guarantee: no auto-escalation, ever ---

def test_not_forced_never_rebuilds_regardless_of_diff():
    # No diff size, corpus size, or change count can trigger it.
    for stored in (0, 100, 1000, 23624, 100000):
        assert _should_use_shadow_rebuild(force_full_rebuild=False, stored_doc_count=stored) is False


def test_incident_numbers_never_rebuild_without_explicit_flag():
    # 2026-06-14: 23,624 stored, scheduled unscoped refresh, large degraded
    # ledger. Without the explicit flag, no rebuild — full stop.
    assert _should_use_shadow_rebuild(force_full_rebuild=False, stored_doc_count=23624) is False


# --- explicit, deliberate rebuild ---

def test_forced_rebuild_on_populated_table():
    assert _should_use_shadow_rebuild(force_full_rebuild=True, stored_doc_count=23624) is True


def test_forced_but_empty_store_builds_in_place_not_shadow():
    # Empty table → nothing to preserve with a shadow → build directly.
    assert _should_use_shadow_rebuild(force_full_rebuild=True, stored_doc_count=0) is False


def test_forced_single_doc_table_rebuilds():
    assert _should_use_shadow_rebuild(force_full_rebuild=True, stored_doc_count=1) is True
