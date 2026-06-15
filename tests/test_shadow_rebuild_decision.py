"""Guards for the shadow-rebuild decision — an expensive-to-get-wrong path.

A shadow rebuild reprocesses the ENTIRE corpus (re-OCR, re-enrich, re-embed
every doc) into a fresh table. It is the right move for a genuine bulk change
or disaster recovery, and catastrophically wasteful when triggered by routine
work. Two real incidents motivated this file:

1. A degraded-docs self-heal re-queued ~3,900 transient OCR failures; counting
   them as "changed" tripped a full 27k-doc shadow rebuild to fix a few
   thousand docs (2026-06-14).
2. Source-scoped runs must never shadow-rebuild — their shadow only contains
   one source and promoting it would drop every other source.

These tests pin the decision boundaries directly (cheap) and the flow-level
behavior lives in test_scan.py (integration).
"""

from flow_index_vault import _should_use_shadow_rebuild


# --- empty / trivial cases never rebuild ---

def test_no_stored_docs_never_rebuilds():
    # First-ever index (empty store) builds directly, no shadow needed.
    assert _should_use_shadow_rebuild(scanned_count=50000, stored_doc_count=0, changed_doc_count=50000) is False


def test_zero_changes_never_rebuilds():
    assert _should_use_shadow_rebuild(scanned_count=30000, stored_doc_count=30000, changed_doc_count=0) is False


def test_tiny_incremental_change_never_rebuilds():
    # The common case: a handful of new/edited docs in a large stable corpus.
    assert _should_use_shadow_rebuild(scanned_count=30000, stored_doc_count=30000, changed_doc_count=5) is False


# --- absolute threshold (>= 1000 changed) ---

# --- the load-bearing guard: a POPULATED index never shadow-rebuilds ---
# Shadow reprocesses every scanned doc into a fresh table. Against a populated
# index (store >= half of scanned) every change is incremental and upserts in
# place, so shadow must NOT fire no matter how large the diff. This is what
# makes "an unscoped run can never auto-trigger a full-corpus reprocess of a
# healthy index" actually true.

def test_populated_index_never_rebuilds_even_above_absolute_threshold():
    assert _should_use_shadow_rebuild(scanned_count=30000, stored_doc_count=30000, changed_doc_count=1000) is False


def test_populated_index_never_rebuilds_even_with_huge_diff():
    assert _should_use_shadow_rebuild(scanned_count=30000, stored_doc_count=30000, changed_doc_count=25000) is False


def test_half_full_index_does_not_rebuild():
    # store == half of scanned → still incremental, no shadow.
    assert _should_use_shadow_rebuild(scanned_count=600, stored_doc_count=300, changed_doc_count=300) is False


# --- shadow DOES fire for a genuine from-near-empty rebuild ---

def test_near_empty_store_rebuilds_on_absolute_threshold():
    assert _should_use_shadow_rebuild(scanned_count=30000, stored_doc_count=200, changed_doc_count=1000) is True


def test_near_empty_store_rebuilds_on_proportional_threshold():
    assert _should_use_shadow_rebuild(scanned_count=600, stored_doc_count=100, changed_doc_count=300) is True


def test_small_genuine_diff_near_empty_no_rebuild():
    assert _should_use_shadow_rebuild(scanned_count=600, stored_doc_count=100, changed_doc_count=50) is False


# --- the expensive incident, now double-protected ---

def test_incident_numbers_never_rebuild():
    """The 2026-06-14 incident: 27,512 scanned, 23,624 stored. Whether you pass
    the genuine diff (8) OR the degraded-inflated count (3,938), a populated
    index of this size must NEVER shadow-rebuild — both the genuine-count fix
    and the from-near-empty guard independently prevent it.
    """
    for changed in (8, 3938):
        assert _should_use_shadow_rebuild(
            scanned_count=27512, stored_doc_count=23624, changed_doc_count=changed
        ) is False
