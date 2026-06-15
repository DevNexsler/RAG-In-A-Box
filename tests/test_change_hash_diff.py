"""Content-hash change detection — churn-proof re-index avoidance.

A doc must be re-indexed only when its CONTENT changes, never when an upstream
timestamp field (updated_at, mtime) moves without a content change. The diff
compares a content change_hash when both sides have one, and falls back to
mtime otherwise.
"""

from flow_index_vault import diff_index_task


def _rec(doc_id, mtime, change_hash=""):
    return {"doc_id": doc_id, "mtime": mtime, "change_hash": change_hash}


# --- hash-first behavior ---

def test_same_hash_different_mtime_is_not_reindexed():
    # The churn case: updated_at moved (mtime differs) but content identical.
    scanned = [_rec("comm::1", mtime=2000.0, change_hash="abc")]
    stored_mtimes = {"comm::1": 1000.0}
    stored_hashes = {"comm::1": "abc"}
    add, delete = diff_index_task(scanned, stored_mtimes, stored_hashes)
    assert add == []  # not re-indexed despite mtime change
    assert delete == []


def test_different_hash_is_reindexed():
    scanned = [_rec("comm::1", mtime=1000.0, change_hash="def")]
    stored_mtimes = {"comm::1": 1000.0}
    stored_hashes = {"comm::1": "abc"}
    add, delete = diff_index_task(scanned, stored_mtimes, stored_hashes)
    assert [r["doc_id"] for r in add] == ["comm::1"]  # content changed


def test_new_doc_is_added():
    scanned = [_rec("comm::2", mtime=1.0, change_hash="xyz")]
    add, delete = diff_index_task(scanned, {}, {})
    assert [r["doc_id"] for r in add] == ["comm::2"]


# --- mtime fallback when a hash is missing ---

def test_falls_back_to_mtime_when_no_stored_hash():
    # Old doc indexed before the hash scheme: stored hash absent -> use mtime.
    scanned = [_rec("doc::1", mtime=2000.0, change_hash="abc")]
    stored_mtimes = {"doc::1": 1000.0}
    add, delete = diff_index_task(scanned, stored_mtimes, {})
    assert [r["doc_id"] for r in add] == ["doc::1"]  # mtime differs -> reindex


def test_falls_back_to_mtime_when_no_scanned_hash():
    # Filesystem source provides no change_hash -> mtime comparison.
    scanned = [_rec("doc::1", mtime=1000.0, change_hash="")]
    stored_mtimes = {"doc::1": 1000.0}
    stored_hashes = {"doc::1": "abc"}
    add, delete = diff_index_task(scanned, stored_mtimes, stored_hashes)
    assert add == []  # mtime equal -> skip


def test_mtime_fallback_unchanged_is_skipped():
    scanned = [_rec("doc::1", mtime=1000.0)]
    add, delete = diff_index_task(scanned, {"doc::1": 1000.0}, {})
    assert add == []


# --- deletion unaffected ---

def test_deletion_still_detected():
    scanned = [_rec("doc::1", mtime=1.0, change_hash="a")]
    stored_mtimes = {"doc::1": 1.0, "doc::gone": 1.0}
    add, delete = diff_index_task(scanned, stored_mtimes, {"doc::1": "a"})
    assert delete == ["doc::gone"]


# --- backward compatibility: old 2-arg signature still works ---

def test_legacy_two_arg_call_uses_mtime():
    scanned = [_rec("doc::1", mtime=2.0), _rec("doc::2", mtime=5.0)]
    add, delete = diff_index_task(scanned, {"doc::1": 2.0})
    assert [r["doc_id"] for r in add] == ["doc::2"]  # new doc only
