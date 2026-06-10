"""Tests for the degraded-docs self-heal ledger.

Docs that index successfully but with transient degradations (OCR/vision
timeouts, enrichment failures) must be re-queued on later runs even though
their mtime is unchanged — otherwise they stay silently degraded until a
full rebuild (901 description-less images sat that way in production).
"""

import json
import threading

from extractors import begin_degradation_capture, collect_degradations, note_degradation
from flow_index_vault import (
    _DEGRADED_MAX_ATTEMPTS,
    _include_degraded_docs,
    _load_degraded_ledger,
    _merge_degraded_ledger,
    _save_degraded_ledger,
)


# --- thread-local collector ---

def test_collector_captures_within_capture_window():
    begin_degradation_capture()
    note_degradation("ocr_describe_failed")
    note_degradation("enrichment_failed")
    assert collect_degradations() == ["ocr_describe_failed", "enrichment_failed"]


def test_collector_is_noop_without_begin():
    # A fresh thread that never called begin: notes are dropped, not crashed.
    results = {}

    def worker():
        note_degradation("ignored")
        results["items"] = collect_degradations()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert results["items"] == []


def test_collector_is_thread_isolated():
    results = {}

    def worker(name):
        begin_degradation_capture()
        note_degradation(name)
        results[name] = collect_degradations()

    threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for i in range(3):
        assert results[f"t{i}"] == [f"t{i}"]


# --- re-include logic ---

def _scanned(*doc_ids):
    return [{"doc_id": d, "mtime": 1.0} for d in doc_ids]


def test_include_degraded_requeues_ledger_docs():
    ledger = {"docs": {"documents::img1": {"reasons": ["ocr_describe_failed"], "attempts": 1}}}
    out = _include_degraded_docs(_scanned("documents::img1", "documents::other"), [], ledger)
    assert [r["doc_id"] for r in out] == ["documents::img1"]


def test_include_degraded_skips_exhausted_attempts():
    ledger = {"docs": {"documents::dead": {"reasons": ["ocr_page_failed:0"], "attempts": _DEGRADED_MAX_ATTEMPTS}}}
    out = _include_degraded_docs(_scanned("documents::dead"), [], ledger)
    assert out == []


def test_include_degraded_no_duplicates_when_already_queued():
    ledger = {"docs": {"documents::img1": {"reasons": ["x"], "attempts": 1}}}
    queued = [{"doc_id": "documents::img1", "mtime": 2.0}]
    out = _include_degraded_docs(_scanned("documents::img1"), queued, ledger)
    assert len(out) == 1


def test_include_degraded_ignores_docs_no_longer_scanned():
    ledger = {"docs": {"documents::gone": {"reasons": ["x"], "attempts": 1}}}
    out = _include_degraded_docs(_scanned("documents::other"), [], ledger)
    assert out == []


# --- merge logic ---

def test_merge_clean_docs_drop_out():
    ledger = {"docs": {"a": {"reasons": ["x"], "attempts": 2}}}
    merged = _merge_degraded_ledger(ledger, {}, {"a"})
    assert merged["docs"] == {}


def test_merge_degraded_docs_accumulate_attempts():
    ledger = {"docs": {"a": {"reasons": ["ocr_describe_failed"], "attempts": 1}}}
    merged = _merge_degraded_ledger(ledger, {"a": ["ocr_describe_failed"]}, set())
    assert merged["docs"]["a"]["attempts"] == 2


def test_merge_new_degraded_doc_starts_at_one():
    merged = _merge_degraded_ledger({"docs": {}}, {"b": ["enrichment_failed", "enrichment_failed"]}, set())
    assert merged["docs"]["b"] == {"reasons": ["enrichment_failed"], "attempts": 1}


# --- persistence ---

def test_ledger_roundtrip(tmp_path):
    ledger = {"docs": {"a": {"reasons": ["x"], "attempts": 1}}}
    _save_degraded_ledger(tmp_path, ledger)
    assert _load_degraded_ledger(tmp_path) == ledger


def test_load_missing_or_corrupt_ledger_returns_empty(tmp_path):
    assert _load_degraded_ledger(tmp_path) == {"docs": {}}
    (tmp_path / "degraded_docs.json").write_text("{broken")
    assert _load_degraded_ledger(tmp_path) == {"docs": {}}
