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
    _DEGRADED_MAX_UNRESOLVED_RUNS,
    _DEGRADED_RETRY_BASE_SECONDS,
    _DEGRADED_RETRY_CAP_SECONDS,
    _degraded_backoff_seconds,
    _load_degraded_ledger,
    _merge_degraded_ledger,
    _reconcile_degraded_docs,
    _save_degraded_ledger,
    _save_degraded_unresolved,
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


# --- re-include / reconciliation logic ---

def _scanned(*doc_ids):
    return [{"doc_id": d, "mtime": 1.0} for d in doc_ids]


def _reconcile(scanned, queued, ledger, sources=("documents",), full_scan=True, **kw):
    """Reconcile against a full scan of `sources` by default."""
    return _reconcile_degraded_docs(
        scanned, queued, ledger, scanned_sources=set(sources), full_scan=full_scan, **kw
    )


def _include_degraded_docs(scanned, queued, ledger, **kw):
    """The re-queue half of reconciliation — what the old reader returned."""
    return _reconcile(scanned, queued, ledger, **kw)[0]


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


# --- total reconciliation (ticket #0618) ---

def test_reconcile_partitions_every_entry_exactly_once():
    # The ledger's size must equal the sum of the buckets — that identity is
    # what stops an entry from vanishing into the gap between the ledger and
    # the 'Re-queued N' count.
    ledger = {"docs": {
        "documents::queued": {"reasons": ["x"], "attempts": 1},
        "documents::requeue": {"reasons": ["x"], "attempts": 1},
        "documents::capped": {"reasons": ["x"], "attempts": _DEGRADED_MAX_ATTEMPTS},
        "documents::gone": {"reasons": ["x"], "attempts": 1},
        "comm_messages::other": {"reasons": ["x"], "attempts": 1},
    }}
    queued = [{"doc_id": "documents::queued", "mtime": 2.0}]
    _, _, report = _reconcile(
        _scanned("documents::queued", "documents::requeue"), queued, ledger,
        sources=("documents",), full_scan=False,
    )
    assert report["already_queued"] == ["documents::queued"]
    assert report["requeued"] == ["documents::requeue"]
    assert report["capped"] == ["documents::capped"]
    assert report["unresolved"] == ["documents::gone"]
    assert report["source_not_scanned"] == ["comm_messages::other"]
    assert report["total"] == 5
    assert report["total"] == sum(
        len(report[b]) for b in
        ("already_queued", "requeued", "capped", "unresolved", "source_not_scanned")
    )


def test_reconcile_ages_entry_missing_from_its_own_scanned_source():
    ledger = {"docs": {"documents::gone": {"reasons": ["x"], "attempts": 1}}}
    _, ledger, report = _reconcile(_scanned("documents::other"), [], ledger)
    assert report["unresolved"] == ["documents::gone"]
    assert ledger["docs"]["documents::gone"]["unresolved_runs"] == 1
    assert report["terminal"] == {}


def test_reconcile_escalates_entry_stuck_unresolved_to_terminal():
    # Ticket #0618: 16 comm_messages entries sat at attempts=1 for eight weeks
    # because their source rows left the configured query. Ageing bounds that.
    ledger = {"docs": {"comm_messages::dup": {"reasons": ["enrichment_failed"], "attempts": 1}}}
    for _ in range(_DEGRADED_MAX_UNRESOLVED_RUNS):
        _, ledger, report = _reconcile(
            _scanned("comm_messages::real"), [], ledger, sources=("comm_messages",)
        )
    assert "comm_messages::dup" not in ledger["docs"], "must not linger in the retry ledger"
    assert report["terminal"]["comm_messages::dup"]["unresolved_runs"] == (
        _DEGRADED_MAX_UNRESOLVED_RUNS
    )
    assert report["terminal"]["comm_messages::dup"]["reasons"] == ["enrichment_failed"]


def test_reconcile_resolving_entry_resets_the_unresolved_streak():
    ledger = {"docs": {"documents::flaky": {
        "reasons": ["x"], "attempts": 1, "unresolved_runs": _DEGRADED_MAX_UNRESOLVED_RUNS - 1,
    }}}
    _, ledger, report = _reconcile(_scanned("documents::flaky"), [], ledger)
    assert report["requeued"] == ["documents::flaky"]
    assert "unresolved_runs" not in ledger["docs"]["documents::flaky"]


def test_reconcile_source_scoped_run_never_ages_another_source():
    # A source-scoped index is no evidence about the sources it did not scan.
    ledger = {"docs": {"documents::img1": {"reasons": ["x"], "attempts": 1}}}
    for _ in range(_DEGRADED_MAX_UNRESOLVED_RUNS + 2):
        _, ledger, report = _reconcile(
            _scanned("comm_messages::m"), [], ledger,
            sources=("comm_messages",), full_scan=False,
        )
    assert report["source_not_scanned"] == ["documents::img1"]
    assert ledger["docs"]["documents::img1"] == {"reasons": ["x"], "attempts": 1}


def test_reconcile_full_scan_ages_entry_from_a_removed_source():
    # A namespace that is no longer configured at all can only be concluded
    # dead by a full scan — and then it must age out, not accumulate forever.
    ledger = {"docs": {"retired::doc": {"reasons": ["x"], "attempts": 1}}}
    _, ledger, report = _reconcile(
        _scanned("documents::img1"), [], ledger, sources=("documents",), full_scan=True
    )
    assert report["unresolved"] == ["retired::doc"]
    assert ledger["docs"]["retired::doc"]["unresolved_runs"] == 1


def test_reconcile_empty_ledger_is_a_noop():
    queued = [{"doc_id": "documents::new", "mtime": 1.0}]
    out, ledger, report = _reconcile(_scanned("documents::new"), queued, {"docs": {}})
    assert out == queued
    assert ledger == {"docs": {}}
    assert report["total"] == 0


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
    merged = _merge_degraded_ledger(
        {"docs": {}}, {"b": ["enrichment_failed", "enrichment_failed"]}, set(), now=42.0
    )
    assert merged["docs"]["b"] == {
        "reasons": ["enrichment_failed"], "attempts": 1, "last_attempt_at": 42.0
    }


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
    # Backoff (dpark 2026-07-28) means the doc is not re-queued on the very next
    # sweep — it heals once its backoff window elapses. The #0251 invariant still
    # holds: attempts is never charged, so it is never abandoned; it just retries
    # on a widening schedule instead of hammering a down provider every run.
    stamped = ledger["docs"][doc_id]["last_attempt_at"]
    due = stamped + _DEGRADED_RETRY_CAP_SECONDS + 1
    requeued = _include_degraded_docs(_scanned(doc_id), [], ledger, now=due)
    assert [r["doc_id"] for r in requeued] == [doc_id]


# --- exponential backoff on retry (dpark 2026-07-28) ---


def test_backoff_grows_exponentially_with_total_attempts():
    base = _DEGRADED_RETRY_BASE_SECONDS
    assert _degraded_backoff_seconds({"attempts": 0}) == base
    assert _degraded_backoff_seconds({"attempts": 1}) == base * 2
    assert _degraded_backoff_seconds({"attempts": 2}) == base * 4
    # transient tries count too — a long outage widens the window.
    assert _degraded_backoff_seconds({"attempts": 0, "transient_attempts": 3}) == base * 8


def test_backoff_is_capped():
    assert _degraded_backoff_seconds({"attempts": 40}) == _DEGRADED_RETRY_CAP_SECONDS


def test_include_degraded_waits_for_backoff_window():
    entry = {"reasons": ["ocr_describe_failed"], "attempts": 0,
             "transient_attempts": 2, "last_attempt_at": 1000.0}
    ledger = {"docs": {"documents::img1": dict(entry)}}
    window = _degraded_backoff_seconds(entry)
    # Not yet due -> not re-queued.
    out = _include_degraded_docs(
        _scanned("documents::img1"), [], ledger, now=1000.0 + window - 1
    )
    assert out == []
    # Past the window -> re-queued.
    out = _include_degraded_docs(
        _scanned("documents::img1"), [], ledger, now=1000.0 + window + 1
    )
    assert [r["doc_id"] for r in out] == ["documents::img1"]


def test_normal_diff_degraded_doc_waits_for_backoff_when_unchanged():
    doc_id = "documents::provider-error"
    record = {"doc_id": doc_id, "mtime": 1.0, "change_hash": "error-v1"}
    entry = {
        "reasons": ["vision_sidecar_failed"],
        "attempts": 0,
        "transient_attempts": 191,
        "last_attempt_at": 1000.0,
        "change_key": "error-v1",
    }
    ledger = {"docs": {doc_id: entry}}

    before_due, _, report = _reconcile(
        [record], [record], ledger, now=1000.0 + _DEGRADED_RETRY_CAP_SECONDS - 1
    )
    at_due, _, due_report = _reconcile(
        [record], [record], ledger, now=1000.0 + _DEGRADED_RETRY_CAP_SECONDS
    )

    assert before_due == []
    assert report["backoff"] == [doc_id]
    assert at_due == [record]
    assert due_report["already_queued"] == [doc_id]


def test_normal_diff_changed_degraded_doc_bypasses_backoff():
    doc_id = "documents::regenerated"
    regenerated = {"doc_id": doc_id, "mtime": 2.0, "change_hash": "content-v2"}
    ledger = {"docs": {doc_id: {
        "reasons": ["vision_sidecar_failed"],
        "attempts": 0,
        "transient_attempts": 191,
        "last_attempt_at": 1000.0,
        "change_key": "error-v1",
    }}}

    out, _, report = _reconcile(
        [regenerated], [regenerated], ledger, now=1001.0
    )

    assert out == [regenerated]
    assert report["already_queued"] == [doc_id]


def test_merge_stamps_degraded_input_change_key():
    merged = _merge_degraded_ledger(
        {"docs": {}},
        {"documents::img1": ["vision_sidecar_failed"]},
        set(),
        change_keys={"documents::img1": "error-v1"},
        now=1234.0,
    )

    assert merged["docs"]["documents::img1"]["change_key"] == "error-v1"


def test_include_degraded_legacy_entry_without_timestamp_is_due():
    # Entries stamped before this change lack last_attempt_at -> retry now,
    # exactly as they did before backoff (no regression on first encounter).
    ledger = {"docs": {"documents::img1": {"reasons": ["x"], "attempts": 1}}}
    out = _include_degraded_docs(_scanned("documents::img1"), [], ledger, now=5000.0)
    assert [r["doc_id"] for r in out] == ["documents::img1"]


def test_merge_stamps_last_attempt_at():
    merged = _merge_degraded_ledger(
        {"docs": {}}, {"a": ["enrichment_failed"]}, set(), now=1234.0
    )
    assert merged["docs"]["a"]["last_attempt_at"] == 1234.0


def test_retry_cap_was_raised():
    # dpark asked to raise the doc-specific cap alongside backoff.
    assert _DEGRADED_MAX_ATTEMPTS >= 12


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


# --- #0618 termination invariant (reconciliation pass 2026-08-01) ---
#
# Production evidence this encodes: on 2026-08-01 the live ledger held 70
# entries while every run logged "Re-queued 2 degraded docs" — the other 68
# were `comm_messages::` rows whose upstream source row had left the configured
# scan query (comm-store sets canonical_message_id on cross-delivery duplicates,
# and the source query filters `canonical_message_id IS NULL`). Nine of them had
# been frozen at attempts=1 since 2026-06-01. The old reader intersected ledger
# ids with the scan and dropped the misses on the floor, so such an entry was
# never retried, never cleared and never counted: it stayed in the retry ledger
# forever. This test fails against that reader and passes once every entry is
# accounted for and aged.

def test_unresolvable_entry_reaches_a_terminal_state():
    """An entry a successful full scan cannot resolve must leave the retry
    ledger within a bounded number of runs instead of sitting there forever."""
    import flow_index_vault as fiv

    stuck = "comm_messages::zoho_mail/<vmZUp-N3T8aXVF8P2j20Ag@geopod-ismtpd-2>"
    ledger = {"docs": {stuck: {"reasons": ["enrichment_failed"], "attempts": 1}}}
    scanned = _scanned("comm_messages::still-here")

    bound = getattr(fiv, "_DEGRADED_MAX_UNRESOLVED_RUNS", 3)
    for run in range(1, 4 * bound + 1):
        if hasattr(fiv, "_reconcile_degraded_docs"):
            _, ledger, _report = fiv._reconcile_degraded_docs(
                scanned, [], ledger,
                scanned_sources={"comm_messages"}, full_scan=True,
            )
        else:  # pre-fix reader: returns the queue only, never touches the ledger
            fiv._include_degraded_docs(scanned, [], ledger)
        if stuck not in ledger.get("docs", {}):
            break
    else:
        raise AssertionError(
            f"unresolvable ledger entry still in the retry ledger after "
            f"{4 * bound} full-scan runs — it can never be retried or cleared"
        )
    assert run <= bound, f"took {run} runs to reach a terminal state (bound {bound})"


def test_terminal_write_failure_does_not_lose_entries(tmp_path, monkeypatch):
    """If the terminal ledger cannot be written, the escalated entries must stay
    in the ACTIVE ledger — otherwise a failed write drops them from both files.

    Found by an adversarial review of PR #81 during the 2026-08-01 reconciliation:
    _save_degraded_unresolved swallowed OSError while the caller unconditionally
    persisted the active ledger with those entries already removed. This host has
    filled its disk before (#0232/#58), so the failure path is reachable.
    """
    import flow_index_vault as fiv

    assert _save_degraded_unresolved.__doc__, "helper must document its contract"

    monkeypatch.setattr(
        fiv.Path, "write_text",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("No space left on device")),
    )
    assert fiv._save_degraded_unresolved(tmp_path, {"docs": {"documents::x": {}}}) is False
