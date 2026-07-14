"""Skip ledger — stop reprocessing docs intentionally not indexed.

Duplicates, oversized media, corrupt files, and genuinely contentless docs
never land in the table, so a plain diff sees them as "new" every run and
re-processes them forever. The skip ledger records each with the file's
change key and a skipped_at stamp; the diff excludes it while unchanged and
younger than _SKIP_RETRY_SECONDS, then re-attempts it once (bounded retry —
never permanently abandoned). A changed file is re-evaluated immediately.
"""

from types import SimpleNamespace

import flow_index_vault as fiv
from extractors import (
    Degradation,
    begin_degradation_capture,
    collect_degradations,
    collect_skips,
    note_degradation,
)
from flow_index_vault import (
    _SKIP_RETRY_SECONDS,
    _change_key,
    _exclude_skipped_docs,
    _load_skip_ledger,
    _merge_skip_ledger,
    _save_skip_ledger,
)


def _rec(doc_id, mtime=1.0, change_hash=""):
    return {"doc_id": doc_id, "mtime": mtime, "change_hash": change_hash}


def _entry(change_key, skipped_at=None, reasons=("x",)):
    entry = {"reasons": sorted(reasons), "change_key": change_key}
    if skipped_at is not None:
        entry["skipped_at"] = skipped_at
    return entry


# --- exclusion ---

def test_unchanged_skip_doc_is_excluded():
    ledger = {"docs": {"d::1": _entry("h1", skipped_at=1000.0, reasons=["duplicate_of:d::9"])}}
    kept, n = _exclude_skipped_docs([_rec("d::1", change_hash="h1")], ledger, now=1000.0 + 900)
    assert kept == [] and n == 1


def test_changed_skip_doc_is_reevaluated():
    # file changed (hash differs) -> keep it so dedup/extraction re-runs,
    # even though the entry is fresh
    ledger = {"docs": {"d::1": _entry("h1", skipped_at=1000.0, reasons=["duplicate_of:d::9"])}}
    kept, n = _exclude_skipped_docs([_rec("d::1", change_hash="h2")], ledger, now=1000.0 + 900)
    assert [r["doc_id"] for r in kept] == ["d::1"] and n == 0


def test_non_skip_docs_pass_through():
    ledger = {"docs": {"d::1": _entry("h1", skipped_at=1000.0)}}
    kept, n = _exclude_skipped_docs([_rec("d::2", change_hash="h2")], ledger, now=1000.0)
    assert [r["doc_id"] for r in kept] == ["d::2"] and n == 0


def test_mtime_fallback_change_key():
    # no change_hash -> key derived from mtime
    ledger = {"docs": {"d::1": _entry("mtime:5.0", skipped_at=1000.0)}}
    kept, n = _exclude_skipped_docs([_rec("d::1", mtime=5.0)], ledger, now=1000.0)
    assert kept == [] and n == 1
    kept, n = _exclude_skipped_docs([_rec("d::1", mtime=6.0)], ledger, now=1000.0)  # mtime moved
    assert [r["doc_id"] for r in kept] == ["d::1"]


# --- bounded retry ---

def test_entry_past_retry_window_is_due():
    # unchanged doc, but the entry aged past _SKIP_RETRY_SECONDS -> re-attempt
    ledger = {"docs": {"d::1": _entry("h1", skipped_at=1000.0)}}
    kept, n = _exclude_skipped_docs(
        [_rec("d::1", change_hash="h1")], ledger, now=1000.0 + _SKIP_RETRY_SECONDS + 1
    )
    assert [r["doc_id"] for r in kept] == ["d::1"] and n == 0


def test_legacy_entry_without_stamp_is_due_immediately():
    # pre-bounded-retry ledger entries carry no skipped_at: one-time
    # re-evaluation, after which the merge stamps them
    ledger = {"docs": {"d::1": _entry("h1")}}
    kept, n = _exclude_skipped_docs([_rec("d::1", change_hash="h1")], ledger)
    assert [r["doc_id"] for r in kept] == ["d::1"] and n == 0


def test_no_text_doc_not_reprocessed_on_second_run(tmp_path):
    # The #0107 loop: a contentless doc must be excluded on the next run and
    # re-attempted once after the retry window — chained over a persisted ledger.
    rec = _rec("d::1", change_hash="h1")
    t0 = 1000.0

    # run 1: doc not in ledger yet -> processed, yields no text -> skip merged
    kept, n = _exclude_skipped_docs([rec], _load_skip_ledger(tmp_path), now=t0)
    assert kept == [rec]
    merged = _merge_skip_ledger(
        _load_skip_ledger(tmp_path),
        {"d::1": {"reasons": ["no_text_extracted"], "change_key": "h1"}},
        set(),
        now=t0,
    )
    _save_skip_ledger(tmp_path, merged)

    # run 2 (~15 min later): unchanged -> excluded, no reprocessing
    kept, n = _exclude_skipped_docs([rec], _load_skip_ledger(tmp_path), now=t0 + 900)
    assert kept == [] and n == 1

    # next day: due for one bounded retry
    kept, n = _exclude_skipped_docs(
        [rec], _load_skip_ledger(tmp_path), now=t0 + _SKIP_RETRY_SECONDS + 1
    )
    assert [r["doc_id"] for r in kept] == ["d::1"] and n == 0


# --- merge ---

def test_merge_adds_new_skips():
    out = _merge_skip_ledger({"docs": {}}, {"d::1": {"reasons": ["duplicate"], "change_key": "h1"}}, set())
    assert out["docs"]["d::1"]["change_key"] == "h1"


def test_merge_stamps_skipped_at():
    out = _merge_skip_ledger(
        {"docs": {}}, {"d::1": {"reasons": ["duplicate"], "change_key": "h1"}}, set(), now=1234.0
    )
    assert out["docs"]["d::1"]["skipped_at"] == 1234.0


def test_merge_restamps_reskipped_doc():
    # a due entry that yields no text again gets a fresh stamp -> next window
    ledger = {"docs": {"d::1": _entry("h1", skipped_at=1000.0, reasons=["no_text_extracted"])}}
    out = _merge_skip_ledger(
        ledger,
        {"d::1": {"reasons": ["no_text_extracted"], "change_key": "h1"}},
        set(),
        now=1000.0 + _SKIP_RETRY_SECONDS + 5,
    )
    assert out["docs"]["d::1"]["skipped_at"] == 1000.0 + _SKIP_RETRY_SECONDS + 5


def test_merge_drops_clean_docs():
    # a doc that indexed cleanly (e.g. its duplicate canonical was removed)
    ledger = {"docs": {"d::1": _entry("h1", skipped_at=1000.0, reasons=["duplicate"])}}
    out = _merge_skip_ledger(ledger, {}, {"d::1"})
    assert out["docs"] == {}


def test_merge_clean_takes_precedence_is_not_needed_but_consistent():
    ledger = {"docs": {"d::1": _entry("old", skipped_at=1.0)}}
    # same doc skipped again with new key -> updated
    out = _merge_skip_ledger(ledger, {"d::1": {"reasons": ["dup"], "change_key": "new"}}, set())
    assert out["docs"]["d::1"]["change_key"] == "new"


# --- persistence ---

def test_skip_ledger_roundtrip(tmp_path):
    ledger = {"docs": {"d::1": _entry("h1", skipped_at=1000.0, reasons=["duplicate"])}}
    _save_skip_ledger(tmp_path, ledger)
    assert _load_skip_ledger(tmp_path) == ledger


def test_load_missing_returns_empty(tmp_path):
    assert _load_skip_ledger(tmp_path) == {"docs": {}}


# --- change key ---

def test_change_key_prefers_hash():
    assert _change_key({"change_hash": "abc", "mtime": 1.0}) == "abc"


def test_change_key_falls_back_to_mtime():
    assert _change_key({"change_hash": "", "mtime": 3.5}) == "mtime:3.5"


# --- process_doc_task: contentless docs must enter the ledger ---

class _EmptyResult:
    full_text = ""
    frontmatter = {}


def _no_text_runtime(source, doc_id, record=None):
    return {
        "store": object(),
        "embed_provider": object(),
        "splitter": object(),
        "config": {},
        "sources_by_name": {"comm": source},
        "source_records_by_ns_doc_id": {doc_id: record or SimpleNamespace(metadata={})},
    }


def _no_text_doc(doc_id):
    return {
        "doc_id": doc_id,
        "rel_path": doc_id.split("::", 1)[-1],
        "mtime": 1.0,
        "size": 0,
        "ext": "txt",
        "source_name": "comm",
    }


def test_no_text_doc_notes_skip(monkeypatch):
    # A genuinely contentless doc (empty transcript row, blank mail body) must
    # note a skip so it lands in the ledger — not silently return and be
    # re-diffed as "new" every run.
    class _Source:
        def extract(self, record):
            return _EmptyResult()

    doc_id = "comm::pg_transcript/1"
    monkeypatch.setattr(fiv, "_RUNTIME", _no_text_runtime(_Source(), doc_id))
    begin_degradation_capture()
    fiv.process_doc_task.fn(_no_text_doc(doc_id))
    assert collect_skips() == ["no_text_extracted"]


def test_degraded_no_text_doc_stays_in_degraded_lane(monkeypatch):
    # Emptiness caused by a transient extraction failure (OCR/vision timeout)
    # is NOT a permanent skip — the degraded lane retries it with capped
    # attempts, which is faster than the skip ledger's daily window.
    class _Source:
        def extract(self, record):
            note_degradation("ocr_timeout")
            return _EmptyResult()

    doc_id = "comm::pg_transcript/2"
    monkeypatch.setattr(fiv, "_RUNTIME", _no_text_runtime(_Source(), doc_id))
    begin_degradation_capture()
    fiv.process_doc_task.fn(_no_text_doc(doc_id))
    assert collect_skips() == []
    assert collect_degradations() == [Degradation("ocr_timeout")]
