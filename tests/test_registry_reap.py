"""Registry lifecycle reaping (#0382).

The store diff only reaps docs that made it into the Lance table
(to_delete = stored - scanned). A doc that was never indexed — skip-ledgered,
degraded-capped, duplicate, or deleted before its first successful run — has
no store row, so when its backing object disappears its registry row survives
forever and deep health counts it as an indexable gap. The reap removes those
rows from the scan diff instead, tombstoning their IDs.
"""

import logging

import pytest

from doc_id_store import DocIDStore
from flow_index_vault import _reap_vanished_registry_rows

_LOGGER = logging.getLogger("test_registry_reap")


@pytest.fixture
def registry(tmp_path):
    store = DocIDStore(tmp_path / "doc_registry.db")
    yield store
    store.close()


def _reap(registry, scanned, stored, roots=None, **kwargs):
    kwargs.setdefault("source_scope", None)
    kwargs.setdefault("max_delete_ratio", 0.5)
    kwargs.setdefault("min_docs_for_ratio", 20)
    kwargs.setdefault("logger", _LOGGER)
    return _reap_vanished_registry_rows(
        registry, set(scanned), set(stored), roots or {}, **kwargs
    )


def test_list_rows_exposes_exact_keys_and_namespaced_ids(registry):
    registry.register("00001", "docs/a@00001@.pdf", source_name="documents")
    registry.register("documents::00001", "docs/a@00001@.pdf", source_name="documents")
    registry.register("comm::msg-1", "cesar/msg-1", source_name="comm")

    rows = {(r["doc_id"], r["namespaced_doc_id"]) for r in registry.list_rows()}

    assert rows == {
        ("00001", "documents::00001"),
        ("documents::00001", "documents::00001"),
        ("comm::msg-1", "comm::msg-1"),
    }


def test_reap_deletes_rows_for_vanished_database_objects(registry):
    # Deleted-database-object class: a Comm/SOR row whose backing DB object is
    # gone is absent from the scan and was never stored — reap and tombstone.
    registry.register("comm::msg-kept", "cesar/msg-kept", source_name="comm")
    registry.register("comm::msg-deleted", "cesar/msg-deleted", source_name="comm")

    reaped, blocked = _reap(registry, scanned={"comm::msg-kept"}, stored=set())

    assert reaped == {"comm::msg-deleted"}
    assert blocked == {}
    assert registry.lookup_path("comm::msg-deleted") is None
    assert registry.is_retired("comm::msg-deleted")
    assert registry.lookup_path("comm::msg-kept") == "cesar/msg-kept"


def test_reap_spares_stored_rows_even_if_unscanned(registry):
    # Stored-but-unscanned docs belong to the store diff (and its mass-delete
    # guard) — the reap must never touch them.
    registry.register("comm::msg-1", "cesar/msg-1", source_name="comm")

    reaped, blocked = _reap(registry, scanned=set(), stored={"comm::msg-1"})

    assert reaped == set()
    assert registry.lookup_path("comm::msg-1") == "cesar/msg-1"


def test_reap_deletes_dual_legacy_and_namespaced_rows(registry, tmp_path):
    # Deleted-filesystem-object class, legacy shape: one document can hold a
    # bare legacy row AND a namespaced row. Both must die, by exact stored key.
    root = tmp_path / "docs"
    root.mkdir()
    registry.register("00001", "probe@00001@.md", source_name="documents")
    registry.register("documents::00001", "probe@00001@.md", source_name="documents")

    reaped, _ = _reap(
        registry, scanned=set(), stored=set(), roots={"documents": root}
    )

    assert reaped == {"documents::00001"}
    assert registry.count() == 0
    assert registry.is_retired("00001")
    assert registry.is_retired("documents::00001")


def test_reap_spares_present_but_unscanned_files(registry, tmp_path):
    # Zero-byte-file class: the scan intentionally skips empty files, but the
    # backing object still exists — presence is deep health's classification
    # job, not lifecycle's. Identity must survive for when content arrives.
    root = tmp_path / "docs"
    root.mkdir()
    (root / "empty@00001@.pdf").write_bytes(b"")
    registry.register("documents::00001", "empty@00001@.pdf", source_name="documents")

    reaped, _ = _reap(
        registry, scanned=set(), stored=set(), roots={"documents": root}
    )

    assert reaped == set()
    assert registry.lookup_path("documents::00001") == "empty@00001@.pdf"


def test_reap_spares_source_when_root_unavailable(registry, tmp_path):
    # Unmounted/unavailable root: absence of files proves nothing — leave the
    # whole source untouched.
    registry.register("documents::00001", "a.md", source_name="documents")

    reaped, _ = _reap(
        registry,
        scanned=set(),
        stored=set(),
        roots={"documents": tmp_path / "unmounted"},
    )

    assert reaped == set()
    assert registry.lookup_path("documents::00001") == "a.md"


def test_reap_mass_orphan_guard_blocks_partial_scan(registry):
    # A scan that suddenly misses most of a source's registry rows looks like
    # a partial scan (source unreachable) — mirror the store diff's per-source
    # mass-delete guard and leave the registry alone.
    for i in range(20):
        registry.register(f"comm::msg-{i}", f"cesar/msg-{i}", source_name="comm")

    reaped, blocked = _reap(registry, scanned={"comm::msg-0"}, stored=set())

    assert reaped == set()
    assert blocked == {"comm": (19, 20)}
    assert registry.count() == 20


def test_reap_honors_mass_delete_signal_from_stored_rows(registry):
    # Partial scan misses 95 stored rows and 5 never-indexed rows. Looking only
    # at the 5 reap candidates makes this appear safely below 50%, even though
    # the store deletion guard has correctly identified a source-wide outage.
    for i in range(100):
        registry.register(f"comm::msg-{i}", f"cesar/msg-{i}", source_name="comm")
    stored = {f"comm::msg-{i}" for i in range(95)}

    reaped, blocked = _reap(registry, scanned=set(), stored=stored)

    assert reaped == set()
    assert blocked == {"comm": (5, 100)}
    assert registry.count() == 100


def test_reap_small_sources_bypass_ratio_guard(registry):
    # Mirrors the store guard's min_docs_for_ratio: tiny sources reap freely.
    registry.register("probe::doc-1", "probe/1", source_name="probe")
    registry.register("probe::doc-2", "probe/2", source_name="probe")

    reaped, blocked = _reap(registry, scanned=set(), stored=set())

    assert reaped == {"probe::doc-1", "probe::doc-2"}
    assert blocked == {}


def test_reap_source_scope_limits_to_that_source(registry):
    # A source-scoped run only scanned one source — other sources' rows are
    # out of scope even though they are absent from this run's scan.
    registry.register("comm::msg-gone", "cesar/msg-gone", source_name="comm")
    registry.register("sor::task-1", "task/1", source_name="sor")

    reaped, _ = _reap(registry, scanned=set(), stored=set(), source_scope="comm")

    assert reaped == {"comm::msg-gone"}
    assert registry.lookup_path("sor::task-1") == "task/1"
