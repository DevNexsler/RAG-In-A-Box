"""Tests for the degraded-docs self-heal ledger.

Docs that index successfully but with transient degradations (OCR/vision
timeouts, enrichment failures) must be re-queued on later runs even though
their mtime is unchanged — otherwise they stay silently degraded until a
full rebuild (901 description-less images sat that way in production).
"""

import json
import socket
import threading

from extractors import (
    begin_degradation_capture,
    collect_degradations,
    extract_image,
    note_degradation,
)
from flow_index_vault import (
    _DEGRADED_MAX_ATTEMPTS,
    _include_degraded_docs,
    _load_degraded_ledger,
    _merge_degraded_ledger,
    _save_degraded_ledger,
)


# --- thread-local collector ---

def test_collector_captures_within_capture_window():
    from extractors import Degradation

    begin_degradation_capture()
    note_degradation("ocr_describe_failed")
    note_degradation("enrichment_failed")
    assert collect_degradations() == [
        Degradation("ocr_describe_failed"),
        Degradation("enrichment_failed"),
    ]


def test_collector_captures_transient_classification():
    # Provider-level failures (connection refused, timeout) are marked
    # transient so the ledger merge can avoid charging the attempts cap.
    from extractors import Degradation

    begin_degradation_capture()
    note_degradation("ocr_describe_failed", transient=True)
    note_degradation("enrichment_failed")
    assert collect_degradations() == [
        Degradation("ocr_describe_failed", transient=True),
        Degradation("enrichment_failed", transient=False),
    ]


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
    from extractors import Degradation

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
        assert results[f"t{i}"] == [Degradation(f"t{i}")]


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


def test_merge_all_transient_run_does_not_charge_attempts():
    # A run degraded ONLY by provider-level failures (provider down — nothing
    # wrong with the doc) must not consume the attempts cap; it counts in the
    # observability-only transient_attempts instead.
    from extractors import Degradation

    ledger = {"docs": {"a": {"reasons": ["ocr_describe_failed"], "attempts": 2}}}
    merged = _merge_degraded_ledger(
        ledger, {"a": [Degradation("ocr_describe_failed", transient=True)]}, set()
    )
    assert merged["docs"]["a"]["attempts"] == 2
    assert merged["docs"]["a"]["transient_attempts"] == 1


def test_merge_doc_specific_failure_still_charges_attempts():
    # A doc-specific failure alongside a transient one still charges the cap —
    # the doc genuinely failed on its own merits this run.
    from extractors import Degradation

    merged = _merge_degraded_ledger(
        {"docs": {}},
        {"a": [
            Degradation("ocr_describe_failed", transient=True),
            Degradation("enrichment_failed", transient=False),
        ]},
        set(),
    )
    assert merged["docs"]["a"]["attempts"] == 1


# --- provider-outage repro (ticket #0251) ---

def _dead_port() -> int:
    """A localhost port with nothing listening — connect gets ECONNREFUSED."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_provider_outage_never_caps_doc(tmp_path):
    """Ticket #0251 repro: a vision-provider outage (real OllamaVisionOCR,
    connection refused on every describe) must never burn a doc to the
    attempts cap — the doc keeps re-queueing and self-heals on recovery."""
    from PIL import Image
    from providers.ocr.ollama_vision import OllamaVisionOCR

    img_path = tmp_path / "photo.png"
    Image.new("RGB", (8, 8), "white").save(img_path)
    provider = OllamaVisionOCR(
        base_url=f"http://127.0.0.1:{_dead_port()}", timeout=5.0
    )

    doc_id = "documents::outage"
    ledger = {"docs": {}}
    for _ in range(_DEGRADED_MAX_ATTEMPTS):
        begin_degradation_capture()
        extract_image(img_path, ocr_provider=provider)
        degradations = collect_degradations()
        assert degradations, "describe against a dead port must degrade the doc"
        ledger = _merge_degraded_ledger(ledger, {doc_id: degradations}, set())

    attempts = ledger["docs"][doc_id]["attempts"]
    assert attempts == 0, (
        "provider-down failures must not consume degraded-ledger attempts — "
        f"assert {attempts} == 0"
    )
    requeued = _include_degraded_docs(_scanned(doc_id), [], ledger)
    assert [r["doc_id"] for r in requeued] == [doc_id]


# --- persistence ---

def test_ledger_roundtrip(tmp_path):
    ledger = {"version": 2, "docs": {"a": {"reasons": ["x"], "attempts": 1}}}
    _save_degraded_ledger(tmp_path, ledger)
    assert _load_degraded_ledger(tmp_path) == ledger


def test_load_missing_or_corrupt_ledger_returns_empty(tmp_path):
    assert _load_degraded_ledger(tmp_path) == {"version": 2, "docs": {}}
    (tmp_path / "degraded_docs.json").write_text("{broken")
    assert _load_degraded_ledger(tmp_path) == {"version": 2, "docs": {}}


# --- v1 -> v2 migration (absorbs scripts/reopen_capped_ocr_docs.py) ---

def test_migration_reopens_v1_capped_ocr_only_docs(tmp_path):
    # Under v1 every degraded run charged attempts, so OCR/vision-capped
    # entries are ambiguous (likely outage-burned) — reopen them. Other
    # failure classes are untouched.
    v1 = {"docs": {
        "documents::ocr": {"reasons": ["ocr_describe_failed"], "attempts": 5},
        "documents::page": {"reasons": ["ocr_page_failed:0"], "attempts": 5},
        "documents::backfill": {"reasons": ["vision_describe_backfill"], "attempts": 5},
        "documents::enrich": {"reasons": ["enrichment_failed"], "attempts": 5},
        "documents::mixed": {"reasons": ["enrichment_failed", "ocr_describe_failed"], "attempts": 5},
        "documents::fresh": {"reasons": ["ocr_describe_failed"], "attempts": 2},
    }}
    (tmp_path / "degraded_docs.json").write_text(json.dumps(v1))

    ledger = _load_degraded_ledger(tmp_path)

    docs = ledger["docs"]
    assert docs["documents::ocr"]["attempts"] == 0
    assert docs["documents::page"]["attempts"] == 0
    assert docs["documents::backfill"]["attempts"] == 0
    assert docs["documents::enrich"]["attempts"] == 5
    assert docs["documents::mixed"]["attempts"] == 5
    assert docs["documents::fresh"]["attempts"] == 2
    assert ledger["version"] == 2


def test_migration_stamps_version_on_disk_once(tmp_path):
    (tmp_path / "degraded_docs.json").write_text(json.dumps(
        {"docs": {"documents::ocr": {"reasons": ["ocr_describe_failed"], "attempts": 5}}}
    ))
    _load_degraded_ledger(tmp_path)
    on_disk = json.loads((tmp_path / "degraded_docs.json").read_text())
    assert on_disk["version"] == 2
    assert on_disk["docs"]["documents::ocr"]["attempts"] == 0


def test_migration_does_not_reopen_v2_capped_docs(tmp_path):
    # v2 attempts only count doc-specific failures — a v2 cap is genuine.
    v2 = {"version": 2, "docs": {
        "documents::ocr": {"reasons": ["ocr_describe_failed"], "attempts": 5},
    }}
    (tmp_path / "degraded_docs.json").write_text(json.dumps(v2))
    ledger = _load_degraded_ledger(tmp_path)
    assert ledger["docs"]["documents::ocr"]["attempts"] == 5
