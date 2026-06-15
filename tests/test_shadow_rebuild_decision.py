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

def test_just_below_absolute_threshold_no_rebuild():
    assert _should_use_shadow_rebuild(scanned_count=30000, stored_doc_count=30000, changed_doc_count=999) is False


def test_at_absolute_threshold_rebuilds():
    assert _should_use_shadow_rebuild(scanned_count=30000, stored_doc_count=30000, changed_doc_count=1000) is True


# --- proportional threshold (changed*2 >= baseline) ---

def test_half_the_corpus_changed_rebuilds():
    assert _should_use_shadow_rebuild(scanned_count=600, stored_doc_count=600, changed_doc_count=300) is True


def test_small_fraction_changed_no_rebuild_on_small_corpus():
    assert _should_use_shadow_rebuild(scanned_count=600, stored_doc_count=600, changed_doc_count=100) is False


# --- the expensive incident: large self-heal, small genuine diff ---

def test_genuine_diff_must_be_used_not_inflated_count():
    """The fix is to pass the GENUINE changed count, excluding degraded re-queue.

    Simulate the incident's numbers: 27,512 scanned, 23,624 stored, but only a
    handful genuinely changed (the ~3,900 difference was degraded self-heal).
    With the genuine count the decision must be 'no shadow rebuild'.
    """
    genuine_changed = 8  # real new/edited docs
    assert _should_use_shadow_rebuild(
        scanned_count=27512, stored_doc_count=23624, changed_doc_count=genuine_changed
    ) is False

    # And the cautionary inverse: had we (wrongly) passed the inflated count,
    # it WOULD have rebuilt — proving why excluding the re-queue matters.
    inflated_changed = 3930 + 8
    assert _should_use_shadow_rebuild(
        scanned_count=27512, stored_doc_count=23624, changed_doc_count=inflated_changed
    ) is True
