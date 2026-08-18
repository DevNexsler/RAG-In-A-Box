"""Exact-content dedupe gate, driven from outside against the staging stack.

Production indexes the same bytes arriving on several messages exactly once:
the first copy wins the canonical election, later copies skip the whole
extract/enrich/embed pipeline, and each keeps a context-only alias so the
conversation it was delivered into stays searchable. `config.staging.yaml`
carries production's `dedupe:` block so this tier covers that branch (#1188).

Several cohorts are deposited in one sweep on purpose: equal-content copies are
serialized by the registry's content-hash lock, so a single cohort never
exercises concurrent dispatch. Distinct cohorts do, at the configured
`enrichment.concurrency`.
"""
import json
import tempfile
import time
import uuid
from collections import Counter
from pathlib import Path

import anyio
import pytest

from tests.e2e.client import get_hook_events, open_mcp_session, search_hits
from tests.e2e.conftest import (
    EXPECTED_CORPUS_DOCS,
    _compose_cp_into_documents,
    indexer_log_lines,
    wait_for_index,
)

COHORTS = 3
COPIES_PER_COHORT = 2

# The two seeded `ops` fixture messages (staging/comm_postgres/init.sql) sit at
# 10:00:00 and 10:01:00, so a delivery timestamped between them has real
# conversation context on both sides — which is what the alias write embeds.
ORIGIN_SOURCE = "quo"
CHANNEL = "ops"

# Opening line of a duplicate's context-only alias (_index_duplicate_delivery_context).
ALIAS_QUERY = "attachment delivery canonical media attachment"
ALIAS_LOC = "context:c:0"


def _sidecar(stem: str, copy_index: int) -> dict:
    return {
        "schema_version": 1,
        "source": ORIGIN_SOURCE,
        "message": {
            "source_message_id": f"{stem}-msg",
            "sender": "Field Agent",
            "sent_at": f"2026-06-01T10:00:{20 + copy_index:02d}Z",
        },
        "channel": {"source_channel_id": CHANNEL},
        "media": {
            "media_index": 0,
            "media_type": "document",
            "original_filename": f"{stem}.md",
        },
    }


def _deposit_cohorts(run_id: str) -> list[dict]:
    """Deposit COHORTS cohorts of COPIES_PER_COHORT byte-identical documents.

    Each copy carries its own sidecar — one message each, in one shared channel:
    the production shape of a single attachment delivered several times.
    """
    cohorts = []
    with tempfile.TemporaryDirectory() as tmp:
        for cohort_index in range(COHORTS):
            phrase = f"cobalt vestibule dossier {run_id} {cohort_index}"
            # Identical across the cohort, distinct per cohort and per run: a
            # re-run against a live stack must not hit the unchanged-skip path.
            body = f"# Delivery\n\nThe {phrase} was attached to this message.\n"
            stems = []
            for copy_index in range(COPIES_PER_COHORT):
                stem = f"dedupe-{run_id}-c{cohort_index}-{copy_index}"
                doc = Path(tmp) / f"{stem}.md"
                doc.write_text(body)
                sidecar = Path(tmp) / f"{stem}.json"
                sidecar.write_text(json.dumps(_sidecar(stem, copy_index)))
                _compose_cp_into_documents(sidecar)
                _compose_cp_into_documents(doc)
                stems.append(stem)
            cohorts.append({"phrase": phrase, "stems": stems})
    return cohorts


async def _sweep_and_wait(session) -> dict:
    started = await session.call_tool_json("file_index_update", {})
    assert started.get("status") == "started", started
    # file_status can still read "idle" for a moment after launch, which would
    # let wait_for_index return before the sweep has processed anything.
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if (await session.call_tool_json("file_status", {})).get("indexer_running"):
            break
        await anyio.sleep(1)
    return await wait_for_index(session, min_docs=EXPECTED_CORPUS_DOCS)


async def _doc_ids_by_stem(session, stems: list[str]) -> dict[str, str]:
    """Map each deposited stem to its namespaced doc_id via the ID audit log.

    The registry keeps a bare and a namespaced row per document; only the
    namespaced id addresses the index.
    """
    log = await session.call_tool_json(
        "file_audit_log", {"event": "registered", "limit": 200})
    by_stem = {}
    for entry in log["entries"]:
        doc_id = str(entry.get("doc_id") or "")
        rel_path = str(entry.get("rel_path") or "")
        if "::" not in doc_id:
            continue
        for stem in stems:
            # Indexing assigns the ID-alias name (stem@<5char>@.md).
            if rel_path.startswith(stem):
                by_stem[stem] = doc_id
    return by_stem


async def _index_duplicate_cohorts() -> dict:
    cohorts = _deposit_cohorts(uuid.uuid4().hex[:8])
    stems = [stem for cohort in cohorts for stem in cohort["stems"]]
    async with open_mcp_session("duplicate_cohorts") as session:
        await _sweep_and_wait(session)
        doc_ids = await _doc_ids_by_stem(session, stems)
        chunks = {
            stem: await session.call_tool_json(
                "file_get_doc_chunks", {"doc_id": doc_id})
            for stem, doc_id in doc_ids.items()
        }
        alias_hits = search_hits(await session.call_tool_json(
            "file_search", {"query": ALIAS_QUERY, "top_k": 50}))
    return {
        "cohorts": cohorts,
        "doc_ids": doc_ids,
        "chunks": chunks,
        "alias_doc_ids": {hit["doc_id"] for hit in alias_hits},
        "hook_events": await get_hook_events(),
    }


@pytest.fixture(scope="session")
def duplicate_cohorts(indexed_corpus):
    """Deposit byte-identical cohorts, sweep once, and snapshot the evidence.

    Sync session fixture driving its own event loop (same shape as
    ``indexed_corpus``): the webhook-sink snapshot must be taken before any
    per-test ``/admin/reset`` wipes it.
    """
    return anyio.run(_index_duplicate_cohorts)


def _split_cohort(snapshot: dict, cohort: dict) -> tuple[list[str], list[str]]:
    """Cohort stems split into those holding the content and those that don't."""
    canonical, duplicate = [], []
    for stem in cohort["stems"]:
        chunks = snapshot["chunks"].get(stem)
        assert isinstance(chunks, list) and chunks, (stem, chunks)
        text = "\n".join(chunk.get("text") or "" for chunk in chunks)
        (canonical if cohort["phrase"] in text else duplicate).append(stem)
    return canonical, duplicate


def test_every_deposited_copy_reached_the_registry(duplicate_cohorts):
    """Guards every other assertion here: a stem that never registered would
    otherwise read as a successfully skipped duplicate."""
    missing = [
        stem
        for cohort in duplicate_cohorts["cohorts"]
        for stem in cohort["stems"]
        if stem not in duplicate_cohorts["doc_ids"]
    ]
    assert not missing, f"deposited copies never registered: {missing}"


def test_byte_identical_deliveries_are_indexed_exactly_once(duplicate_cohorts):
    """Only one copy of identical bytes carries the document's content."""
    for cohort in duplicate_cohorts["cohorts"]:
        canonical, _duplicate = _split_cohort(duplicate_cohorts, cohort)
        assert len(canonical) == 1, (
            f"{len(canonical)} of {COPIES_PER_COHORT} byte-identical copies were "
            f"indexed under their own content — the dedupe gate elected no single "
            f"canonical: {canonical}"
        )


def test_skipped_duplicates_keep_a_searchable_context_alias(duplicate_cohorts):
    """A skipped duplicate holds exactly its context alias — no second extraction."""
    for cohort in duplicate_cohorts["cohorts"]:
        canonical, duplicates = _split_cohort(duplicate_cohorts, cohort)
        assert duplicates, f"cohort {cohort['phrase']!r} produced no duplicate"
        canonical_doc_id = duplicate_cohorts["doc_ids"][canonical[0]]

        for stem in duplicates:
            chunks = duplicate_cohorts["chunks"][stem]
            assert [chunk.get("loc") for chunk in chunks] == [ALIAS_LOC], (stem, chunks)
            text = chunks[0].get("text") or ""
            assert canonical_doc_id in text, (stem, canonical[0], text[:400])
            assert "[Conversation context]" in text, (stem, text[:400])
            # The alias exists so the conversation this copy was delivered into
            # stays retrievable — assert presence, never rank (sim embeddings
            # carry no semantic signal).
            assert duplicate_cohorts["doc_ids"][stem] in duplicate_cohorts["alias_doc_ids"], (
                f"alias for {stem} is not searchable: "
                f"{sorted(duplicate_cohorts['alias_doc_ids'])}"
            )


def test_dedupe_sweep_delivers_one_document_indexed_per_document(duplicate_cohorts):
    """One `document.indexed` delivery per document, counted at the receiver.

    What a downstream consumer depends on is that one indexing decision reaches
    it once. This tree's payload carries no `event_id`, so the identity of a
    delivery is the document it is about: a sweep makes one decision per
    document, so the same doc_id twice at the sink is a repeat delivery. That is
    the shape the duplicate-delivery defect took (#1174 — one outbox row sent by
    two drains), and it clusters on dedup-skipped documents because their fast
    path lets several dispatches overlap inside one drain window.
    """
    events = [event for event in duplicate_cohorts["hook_events"]
              if event.get("event") == "document.indexed"]
    assert events, "the dedupe sweep delivered no document.indexed events"

    repeated = {
        doc_id: count
        for doc_id, count in Counter(e.get("doc_id") for e in events).items()
        if count > 1
    }
    assert not repeated, f"document.indexed delivered more than once: {repeated}"

    # Every cohort's canonical is what downstream systems are told about.
    delivered = {event.get("doc_id") for event in events}
    for cohort in duplicate_cohorts["cohorts"]:
        canonical, _duplicates = _split_cohort(duplicate_cohorts, cohort)
        doc_id = duplicate_cohorts["doc_ids"][canonical[0]]
        assert doc_id in delivered, (
            f"no document.indexed for canonical {doc_id}: {sorted(delivered)}"
        )


def test_staging_sweeps_dispatch_documents_concurrently(duplicate_cohorts):
    """The staging pool must match production's, or the races prod hits stay invisible.

    ``_process_docs`` logs its worker count only when it is above one, so the
    line's presence is the proof that ``enrichment.concurrency`` is
    production-shaped rather than serialized down to 1.
    """
    dispatched = [line for line in indexer_log_lines() if "with concurrency=" in line]
    assert dispatched, (
        "no sweep logged a concurrent document pool — enrichment.concurrency in "
        "config.staging.yaml is 1, so the concurrent-dispatch shape production "
        "runs is never exercised"
    )
