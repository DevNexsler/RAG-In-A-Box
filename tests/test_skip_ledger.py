"""Skip ledger — stop reprocessing docs intentionally not indexed.

Duplicates, oversized media, and corrupt files never land in the table, so a
plain diff sees them as "new" every run and re-processes them forever. The
skip ledger records each with the file's change key; the diff excludes it
while unchanged, and re-evaluates it if the file changes.
"""

from flow_index_vault import (
    _change_key,
    _exclude_skipped_docs,
    _load_skip_ledger,
    _merge_skip_ledger,
    _save_skip_ledger,
)


def _rec(doc_id, mtime=1.0, change_hash=""):
    return {"doc_id": doc_id, "mtime": mtime, "change_hash": change_hash}


# --- exclusion ---

def test_unchanged_skip_doc_is_excluded():
    ledger = {"docs": {"d::1": {"reasons": ["duplicate_of:d::9"], "change_key": "h1"}}}
    kept, n = _exclude_skipped_docs([_rec("d::1", change_hash="h1")], ledger)
    assert kept == [] and n == 1


def test_changed_skip_doc_is_reevaluated():
    # file changed (hash differs) -> keep it so dedup/extraction re-runs
    ledger = {"docs": {"d::1": {"reasons": ["duplicate_of:d::9"], "change_key": "h1"}}}
    kept, n = _exclude_skipped_docs([_rec("d::1", change_hash="h2")], ledger)
    assert [r["doc_id"] for r in kept] == ["d::1"] and n == 0


def test_non_skip_docs_pass_through():
    ledger = {"docs": {"d::1": {"reasons": ["x"], "change_key": "h1"}}}
    kept, n = _exclude_skipped_docs([_rec("d::2", change_hash="h2")], ledger)
    assert [r["doc_id"] for r in kept] == ["d::2"] and n == 0


def test_mtime_fallback_change_key():
    # no change_hash -> key derived from mtime
    ledger = {"docs": {"d::1": {"reasons": ["x"], "change_key": "mtime:5.0"}}}
    kept, n = _exclude_skipped_docs([_rec("d::1", mtime=5.0)], ledger)
    assert kept == [] and n == 1
    kept, n = _exclude_skipped_docs([_rec("d::1", mtime=6.0)], ledger)  # mtime moved
    assert [r["doc_id"] for r in kept] == ["d::1"]


# --- merge ---

def test_merge_adds_new_skips():
    out = _merge_skip_ledger({"docs": {}}, {"d::1": {"reasons": ["duplicate"], "change_key": "h1"}}, set())
    assert out["docs"]["d::1"]["change_key"] == "h1"


def test_merge_drops_clean_docs():
    # a doc that indexed cleanly (e.g. its duplicate canonical was removed)
    ledger = {"docs": {"d::1": {"reasons": ["duplicate"], "change_key": "h1"}}}
    out = _merge_skip_ledger(ledger, {}, {"d::1"})
    assert out["docs"] == {}


def test_merge_clean_takes_precedence_is_not_needed_but_consistent():
    ledger = {"docs": {"d::1": {"reasons": ["x"], "change_key": "old"}}}
    # same doc skipped again with new key -> updated
    out = _merge_skip_ledger(ledger, {"d::1": {"reasons": ["dup"], "change_key": "new"}}, set())
    assert out["docs"]["d::1"]["change_key"] == "new"


# --- persistence ---

def test_skip_ledger_roundtrip(tmp_path):
    ledger = {"docs": {"d::1": {"reasons": ["duplicate"], "change_key": "h1"}}}
    _save_skip_ledger(tmp_path, ledger)
    assert _load_skip_ledger(tmp_path) == ledger


def test_load_missing_returns_empty(tmp_path):
    assert _load_skip_ledger(tmp_path) == {"docs": {}}


# --- change key ---

def test_change_key_prefers_hash():
    assert _change_key({"change_hash": "abc", "mtime": 1.0}) == "abc"


def test_change_key_falls_back_to_mtime():
    assert _change_key({"change_hash": "", "mtime": 3.5}) == "mtime:3.5"
