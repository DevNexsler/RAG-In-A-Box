"""Tests for the degraded-docs self-heal ledger.

Docs that index successfully but with transient degradations (OCR/vision
timeouts, enrichment failures) must be re-queued on later runs even though
their mtime is unchanged — otherwise they stay silently degraded until a
full rebuild (901 description-less images sat that way in production).
"""

import json
import socket
import threading
import time

import httpx

from extractors import (
    begin_degradation_capture,
    collect_degradations,
    extract_image,
    note_degradation,
)
from flow_index_vault import (
    _DEGRADED_MAX_ATTEMPTS,
    _DEGRADED_RETRY_BASE_SECONDS,
    _apply_degraded_ledger,
    _change_key,
    _degraded_delta,
    _degraded_state,
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
    merged = _merge_degraded_ledger(
        {"docs": {}}, {"b": ["enrichment_failed", "enrichment_failed"]}, set(), now=1000.0
    )
    entry = merged["docs"]["b"]
    assert entry["reasons"] == ["enrichment_failed"]
    assert entry["attempts"] == 1
    assert entry["failures"] == 1
    assert entry["last_attempt"] == 1000.0
    assert entry["retry_after"] >= 1000.0 + _DEGRADED_RETRY_BASE_SECONDS


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
    # Still never abandoned: the doc is re-queued once its backoff comes due
    # (#0583 added the schedule; it must not have turned into a cap).
    due_at = ledger["docs"][doc_id]["retry_after"]
    assert _include_degraded_docs(_scanned(doc_id), [], ledger, now=due_at) != []


# --- deterministic-failure repro (ticket #0583) ---

def _hung_vision_provider(calls: list):
    """An OCR provider whose describe() always times out — the deterministic
    provider-side failure of #0583 (57 images, ~908s each, every run)."""

    class _Hung:
        def extract(self, file_path, page=None):
            return ""

        def describe(self, file_path):
            calls.append(str(file_path))
            raise httpx.ReadTimeout("timed out")

    return _Hung()


def _simulate_run(index_root, scanned, diff_queue, provider, now):
    """One index run reduced to the seams under test: admit the degraded docs
    that are due on top of what the diff queued, extract each, fold the
    outcomes back into the ledger."""
    queued, stats = _apply_degraded_ledger(index_root, scanned, list(diff_queue), now=now)
    degraded_now, clean_now = {}, set()
    for record in queued:
        begin_degradation_capture()
        extract_image(record["path"], ocr_provider=provider)
        reasons = collect_degradations()
        if reasons:
            degraded_now[record["doc_id"]] = reasons
        else:
            clean_now.add(record["doc_id"])
    ledger = _merge_degraded_ledger(
        _load_degraded_ledger(index_root),
        degraded_now,
        clean_now,
        change_keys={r["doc_id"]: _change_key(r) for r in scanned},
        now=now,
    )
    _save_degraded_ledger(index_root, ledger)
    return queued, ledger, stats


def test_deterministic_describe_timeout_is_not_reattempted_next_run(tmp_path):
    """#0583 repro: a describe timeout marks the doc degraded, and the very
    next run re-queues it with no backoff — 57 docs x ~908s ate ~90% of every
    4h run. Two consecutive runs over an unchanged vault must produce at most
    ONE describe attempt; the doc is parked in cross-run backoff, not retried
    once per run. It is still not abandoned: a later run does retry it."""
    from PIL import Image

    img_path = tmp_path / "photo.png"
    Image.new("RGB", (8, 8), "white").save(img_path)
    doc_id = "documents::hung"
    scanned = [{"doc_id": doc_id, "mtime": 1.0, "path": img_path}]
    calls: list = []
    provider = _hung_vision_provider(calls)
    run_at = 1_000_000.0
    run_gap = 4 * 60 * 60  # the production indexing cadence

    # Run 1: the doc is new, so the diff queues it. It degrades (timeout).
    _, ledger, _ = _simulate_run(tmp_path, scanned, scanned, provider, run_at)
    assert calls == [str(img_path)]
    assert doc_id in ledger["docs"]

    # Run 2, one cadence later: nothing changed, so the diff queues nothing.
    # The degraded ledger must NOT hand the doc straight back.
    queued, ledger, stats = _simulate_run(
        tmp_path, scanned, [], provider, run_at + run_gap
    )
    assert queued == [], "a still-failing doc must not be re-attempted every run"
    assert len(calls) == 1, f"expected exactly 1 describe attempt, got {len(calls)}"
    assert stats == {"entries": 1, "requeued": 0, "parked": 1, "capped": 0}
    assert _degraded_state(ledger["docs"][doc_id], run_at + run_gap) == "parked"

    # ...but the backoff is a schedule, not a cap: once it comes due the doc is
    # re-attempted, so a doc still heals when the provider recovers (#0251).
    due_at = ledger["docs"][doc_id]["retry_after"]
    queued, ledger, _ = _simulate_run(tmp_path, scanned, [], provider, due_at)
    assert [r["doc_id"] for r in queued] == [doc_id]
    assert len(calls) == 2
    assert ledger["docs"][doc_id]["failures"] == 2


# --- persistence ---

def test_ledger_roundtrip(tmp_path):
    ledger = {
        "version": 3,
        "docs": {
            "a": {
                "reasons": ["x"],
                "attempts": 1,
                "failures": 1,
                "last_attempt": 10.0,
                "retry_after": 100.0,
                "change_key": "mtime:1.0",
            }
        },
    }
    assert _save_degraded_ledger(tmp_path, ledger) is True
    assert _load_degraded_ledger(tmp_path) == ledger


def test_load_missing_or_corrupt_ledger_returns_empty(tmp_path):
    assert _load_degraded_ledger(tmp_path) == {"version": 3, "docs": {}}
    (tmp_path / "degraded_docs.json").write_text("{broken")
    assert _load_degraded_ledger(tmp_path) == {"version": 3, "docs": {}}


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
    assert ledger["version"] == 3


def test_migration_stamps_version_on_disk_once(tmp_path):
    (tmp_path / "degraded_docs.json").write_text(json.dumps(
        {"docs": {"documents::ocr": {"reasons": ["ocr_describe_failed"], "attempts": 5}}}
    ))
    _load_degraded_ledger(tmp_path)
    on_disk = json.loads((tmp_path / "degraded_docs.json").read_text())
    assert on_disk["version"] == 3
    assert on_disk["docs"]["documents::ocr"]["attempts"] == 0


def test_migration_does_not_reopen_v2_capped_docs(tmp_path):
    # v2 attempts only count doc-specific failures — a v2 cap is genuine.
    v2 = {"version": 2, "docs": {
        "documents::ocr": {"reasons": ["ocr_describe_failed"], "attempts": 5},
    }}
    (tmp_path / "degraded_docs.json").write_text(json.dumps(v2))
    ledger = _load_degraded_ledger(tmp_path)
    assert ledger["docs"]["documents::ocr"]["attempts"] == 5


# --- v2 -> v3 migration: recorded history becomes a retry schedule ---

def test_migration_reopened_docs_are_due_immediately(tmp_path):
    # A v1 cap was the outage's fault, so the reopened doc earns no backoff.
    (tmp_path / "degraded_docs.json").write_text(json.dumps({"docs": {
        "documents::ocr": {"reasons": ["ocr_describe_failed"], "attempts": 5},
    }}))
    entry = _load_degraded_ledger(tmp_path)["docs"]["documents::ocr"]
    assert entry["failures"] == 0
    assert _degraded_state(entry, time.time()) == "due"


def test_migration_parks_v2_docs_with_a_failure_history(tmp_path):
    # The 57 docs of #0583 sat at transient_attempts~6 with nowhere to record a
    # next-attempt time, so the whole set came due every run. v3 schedules them
    # from that history — they park on the first run after the upgrade.
    (tmp_path / "degraded_docs.json").write_text(json.dumps({"version": 2, "docs": {
        "documents::hung": {
            "reasons": ["ocr_describe_failed"], "attempts": 0, "transient_attempts": 6,
        },
    }}))
    entry = _load_degraded_ledger(tmp_path)["docs"]["documents::hung"]
    assert entry["failures"] == 6
    assert entry["last_attempt"] > 0
    assert _degraded_state(entry, time.time()) == "parked"


# --- backoff schedule ---

def _entry(**over):
    base = {"reasons": ["ocr_describe_failed"], "attempts": 0, "failures": 1,
            "last_attempt": 1000.0, "retry_after": 1000.0 + _DEGRADED_RETRY_BASE_SECONDS}
    base.update(over)
    return base


def test_state_parked_before_due_and_due_after():
    entry = _entry()
    assert _degraded_state(entry, 1000.0) == "parked"
    assert _degraded_state(entry, entry["retry_after"]) == "due"


def test_state_capped_wins_over_schedule():
    assert _degraded_state(_entry(attempts=_DEGRADED_MAX_ATTEMPTS), 1e12) == "capped"


def test_changed_content_is_due_regardless_of_schedule_or_cap():
    # The failure was about the old bytes: a modified file is re-evaluated at
    # once, which is also how a doc escapes the attempts cap.
    parked = _entry(change_key="mtime:1.0")
    assert _degraded_state(parked, 1000.0, "mtime:1.0") == "parked"
    assert _degraded_state(parked, 1000.0, "mtime:2.0") == "due"
    capped = _entry(attempts=_DEGRADED_MAX_ATTEMPTS, change_key="mtime:1.0")
    assert _degraded_state(capped, 1e12, "mtime:2.0") == "due"


def test_changed_content_starts_fresh_failure_history():
    from extractors import Degradation

    previous = _entry(
        attempts=_DEGRADED_MAX_ATTEMPTS,
        failures=8,
        transient_attempts=3,
        change_key="mtime:1.0",
    )
    merged = _merge_degraded_ledger(
        {"docs": {"a": previous}},
        {"a": [Degradation("ocr_describe_failed", transient=True)]},
        set(),
        change_keys={"a": "mtime:2.0"},
        now=2000.0,
    )
    entry = merged["docs"]["a"]

    assert entry["change_key"] == "mtime:2.0"
    assert entry["attempts"] == 0
    assert entry["transient_attempts"] == 1
    assert entry["failures"] == 1
    assert _degraded_state(entry, 2000.0, "mtime:2.0") == "parked"


def test_changed_content_resets_migrated_failure_history(tmp_path):
    from PIL import Image

    img_path = tmp_path / "photo.png"
    Image.new("RGB", (8, 8), "white").save(img_path)
    doc_id = "documents::legacy"
    (tmp_path / "degraded_docs.json").write_text(json.dumps({
        "version": 2,
        "docs": {
            doc_id: {
                "reasons": ["ocr_describe_failed"],
                "attempts": _DEGRADED_MAX_ATTEMPTS,
                "transient_attempts": 3,
            },
        },
    }))
    scanned = [{"doc_id": doc_id, "mtime": 2.0, "path": img_path}]
    calls: list = []

    _, ledger, _ = _simulate_run(
        tmp_path, scanned, scanned, _hung_vision_provider(calls), 2000.0
    )

    entry = ledger["docs"][doc_id]
    assert calls == [str(img_path)]
    assert entry["change_key"] == "mtime:2.0"
    assert entry["attempts"] == 0
    assert entry["transient_attempts"] == 1
    assert entry["failures"] == 1
    assert _degraded_state(entry, 2000.0, "mtime:2.0") == "parked"


def test_backoff_grows_with_consecutive_failures_and_caps():
    intervals = []
    for failures in range(1, 12):
        entry = _merge_degraded_ledger(
            {"docs": {"a": {"reasons": ["x"], "attempts": 0, "failures": failures - 1}}},
            {"a": ["ocr_describe_failed"]},
            set(),
            now=0.0,
        )["docs"]["a"]
        intervals.append(entry["retry_after"])
    assert intervals[0] < intervals[1] < intervals[2]        # doubling
    assert intervals[-1] == intervals[-2]                    # capped, not unbounded
    assert max(intervals) < float("inf")                     # never "never" (#0251)


def test_transient_only_failures_still_earn_backoff():
    # The #0583 root cause: a transient failure must not cap a doc (#0251) but
    # it must still be scheduled, or a deterministic timeout retries forever.
    from extractors import Degradation

    entry = _merge_degraded_ledger(
        {"docs": {}},
        {"a": [Degradation("ocr_describe_failed", transient=True)]},
        set(),
        now=1000.0,
    )["docs"]["a"]
    assert entry["attempts"] == 0                            # never abandoned
    assert entry["failures"] == 1
    assert _degraded_state(entry, 1000.0) == "parked"


def test_backoff_is_spread_across_docs():
    # Entries merged in one batch share a stamp; a flat interval would bring the
    # whole set due in one later run (#0480).
    merged = _merge_degraded_ledger(
        {"docs": {}}, {f"doc{i}": ["ocr_describe_failed"] for i in range(20)}, set(),
        now=0.0,
    )
    assert len({e["retry_after"] for e in merged["docs"].values()}) > 1


def test_healed_doc_drops_out_and_loses_its_schedule():
    ledger = {"docs": {"a": _entry(failures=4)}}
    assert _merge_degraded_ledger(ledger, {}, {"a"})["docs"] == {}


# --- claim on handout ---

def test_apply_requeues_due_docs_and_claims_the_retry(tmp_path):
    _save_degraded_ledger(tmp_path, {"version": 3, "docs": {"documents::a": _entry()}})
    scanned = [{"doc_id": "documents::a", "mtime": 1.0}]
    due_at = _entry()["retry_after"]

    queued, stats = _apply_degraded_ledger(tmp_path, scanned, [], now=due_at)

    assert [r["doc_id"] for r in queued] == ["documents::a"]
    assert stats == {"entries": 1, "requeued": 1, "parked": 0, "capped": 0}
    # The retry is deferred at handout, so a run killed mid-queue does not hand
    # the same doc back to the next run (#0480).
    persisted = _load_degraded_ledger(tmp_path)["docs"]["documents::a"]
    assert persisted["retry_after"] > due_at
    assert persisted["last_attempt"] == due_at
    again, _ = _apply_degraded_ledger(tmp_path, scanned, [], now=due_at)
    assert again == []


def test_apply_reports_parked_entries_without_requeueing(tmp_path):
    _save_degraded_ledger(tmp_path, {"version": 3, "docs": {
        "documents::a": _entry(),
        "documents::dead": _entry(attempts=_DEGRADED_MAX_ATTEMPTS),
    }})
    scanned = [{"doc_id": "documents::a", "mtime": 1.0},
               {"doc_id": "documents::dead", "mtime": 1.0}]
    queued, stats = _apply_degraded_ledger(tmp_path, scanned, [], now=1000.0)
    assert queued == []
    assert stats == {"entries": 2, "requeued": 0, "parked": 1, "capped": 1}
    persisted = _load_degraded_ledger(tmp_path)["docs"]
    assert persisted["documents::a"]["change_key"] == "mtime:1.0"
    assert persisted["documents::dead"]["change_key"] == "mtime:1.0"


def test_apply_fails_closed_when_the_claim_cannot_be_persisted(tmp_path, monkeypatch):
    _save_degraded_ledger(tmp_path, {"version": 3, "docs": {"documents::a": _entry()}})
    monkeypatch.setattr(
        "flow_index_vault._save_degraded_ledger", lambda *_a, **_k: False
    )
    queued, stats = _apply_degraded_ledger(
        tmp_path, [{"doc_id": "documents::a", "mtime": 1.0}], [],
        now=_entry()["retry_after"],
    )
    assert queued == []            # undurable deferral -> do not hand out the retry
    assert stats["requeued"] == 0


def test_apply_leaves_diff_queued_docs_untouched(tmp_path):
    # A doc the diff already queued (changed file) is not a degraded re-heal and
    # must not be deferred by the ledger.
    _save_degraded_ledger(tmp_path, {"version": 3, "docs": {"documents::a": _entry()}})
    queued, stats = _apply_degraded_ledger(
        tmp_path,
        [{"doc_id": "documents::a", "mtime": 2.0}],
        [{"doc_id": "documents::a", "mtime": 2.0}],
        now=1000.0,
    )
    assert [r["doc_id"] for r in queued] == ["documents::a"]
    assert stats["requeued"] == 0
    assert _load_degraded_ledger(tmp_path)["docs"]["documents::a"] == _entry()


# --- convergence visibility ---

def test_delta_reports_healed_still_and_newly_degraded():
    previous = {"docs": {"a": _entry(), "b": _entry()}}
    updated = _merge_degraded_ledger(previous, {"b": ["x"], "c": ["x"]}, {"a"}, now=0.0)
    delta = _degraded_delta(previous, updated, {"b": ["x"], "c": ["x"]}, {"a"}, now=0.0)
    assert delta["healed"] == 1
    assert delta["still_degraded"] == 1
    assert delta["newly_degraded"] == 1
    assert delta["entries"] == 2
    assert delta["parked"] == 2


def test_delta_makes_a_non_converging_set_visible():
    # The #0583 signature: every degraded doc was already degraded, none healed.
    previous = {"docs": {f"d{i}": _entry() for i in range(55)}}
    degraded_now = {f"d{i}": ["ocr_describe_failed"] for i in range(55)}
    updated = _merge_degraded_ledger(previous, degraded_now, set(), now=0.0)
    delta = _degraded_delta(previous, updated, degraded_now, set(), now=0.0)
    assert delta["healed"] == 0
    assert delta["still_degraded"] == 55
    assert delta["newly_degraded"] == 0


def test_delta_counts_parked_docs_as_still_degraded():
    previous = {"docs": {"a": _entry()}}
    delta = _degraded_delta(previous, previous, {}, set(), now=0.0)

    assert delta["healed"] == 0
    assert delta["still_degraded"] == 1
    assert delta["newly_degraded"] == 0
