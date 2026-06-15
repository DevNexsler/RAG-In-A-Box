"""Safety guards against destructive actions from anomalous scans.

Three real risks, all the same root flaw — the indexer trusting an anomalous
scan as truth and acting destructively:

1. Mass deletion: a partial/empty source scan makes to_delete balloon toward
   the whole corpus -> those docs get deleted, then fully re-enriched.
2. Registry-empty wipe: an empty doc_registry beside a populated table drops
   the whole table (a lost registry looks like the legacy migration case).
3. Shadow rebuild: a large diff against a populated index reprocesses every
   doc into a fresh table instead of upserting in place.
"""

from unittest.mock import MagicMock, patch

import pytest

from sources.base import SourceRecord
from flow_index_vault import index_vault_flow, _should_use_shadow_rebuild


# ---------------------------------------------------------------------------
# Guard 3 (decision function) — shadow is explicit-only, never auto-escalated
# ---------------------------------------------------------------------------

def test_shadow_never_auto_triggers_on_populated_index():
    # No diff size or corpus size can trigger it without the explicit flag.
    assert _should_use_shadow_rebuild(force_full_rebuild=False, stored_doc_count=30000) is False


def test_shadow_only_on_explicit_force_flag():
    assert _should_use_shadow_rebuild(force_full_rebuild=True, stored_doc_count=30000) is True


def test_shadow_forced_but_empty_builds_in_place():
    assert _should_use_shadow_rebuild(force_full_rebuild=True, stored_doc_count=0) is False


# ---------------------------------------------------------------------------
# Guards 1 & 2 (flow integration)
# ---------------------------------------------------------------------------

def _run_flow(tmp_path, *, scanned_doc_ids, stored_mtimes, registry_count,
              config_extra=None, deletes_capture=None):
    active_store = MagicMock()
    active_store.list_doc_ids.return_value = list(stored_mtimes.keys())
    active_store.list_doc_mtimes.return_value = dict(stored_mtimes)
    active_store.count_chunks.return_value = len(stored_mtimes)
    active_store.fts_available.return_value = True

    fake_registry = MagicMock()
    fake_registry.count.return_value = registry_count
    fake_taxonomy = MagicMock()
    fake_taxonomy.count.return_value = 0

    class _FakeSource:
        name = "documents"

        def scan(self):
            return iter([
                SourceRecord(
                    doc_id=d.split("::", 1)[1], natural_key=f"{d}.txt",
                    source_type="txt", mtime=1.0, size=4,
                    metadata={"abs_path": str(tmp_path / f"{d}.txt"), "ext": "txt"},
                )
                for d in scanned_doc_ids
            ])

        def set_ocr_provider(self, p):
            return None

        def close(self):
            return None

    config = {
        "index_root": str(tmp_path / "index"),
        "sources": [{"type": "filesystem", "name": "documents", "root": str(tmp_path)}],
        "chunking": {"max_chars": 1800, "overlap": 200, "semantic": {"enabled": False}},
        "enrichment": {"enabled": False},
        "ocr": {"enabled": False},
        "lancedb": {"table": "chunks"},
        "pdf": {},
        "logging": {"level": "WARNING"},
        **(config_extra or {}),
    }

    delete_mock = MagicMock()
    if deletes_capture is not None:
        delete_mock.side_effect = lambda ids: deletes_capture.extend(ids)

    with patch("flow_index_vault.get_run_logger", return_value=MagicMock()):
        with patch("flow_index_vault.load_config", return_value=config):
            with patch("flow_index_vault.open_store_with_recovery", return_value=active_store):
                with patch("flow_index_vault.LanceDBStore", return_value=active_store) as ldb:
                    with patch("flow_index_vault.DocIDStore", return_value=fake_registry):
                        with patch("flow_index_vault.build_embed_provider", return_value=MagicMock()):
                            with patch("flow_index_vault.build_ocr_provider", return_value=None):
                                with patch("sources.build_source", return_value=_FakeSource()):
                                    with patch("core.taxonomy.load_taxonomy_store", return_value=fake_taxonomy):
                                        with patch("flow_index_vault._process_docs", return_value=[]):
                                            with patch("flow_index_vault.delete_docs_task", delete_mock):
                                                with patch("flow_index_vault.index_stats_task"):
                                                    with patch("flow_index_vault.write_index_metadata_task"):
                                                        index_vault_flow.fn("dummy.yaml")
    return active_store, delete_mock, ldb


def test_partial_scan_does_not_mass_delete(tmp_path):
    # 100 docs indexed; scan returns only 2 (source hiccup). Without the guard,
    # 98 would be deleted. With it, deletion is blocked.
    stored = {f"documents::doc-{i}": 1.0 for i in range(100)}
    deletes = []
    _run_flow(tmp_path, scanned_doc_ids=["documents::doc-0", "documents::doc-1"],
              stored_mtimes=stored, registry_count=100, deletes_capture=deletes)
    assert deletes == []  # mass deletion blocked


def test_normal_small_deletion_still_proceeds(tmp_path):
    # 100 indexed, scan returns 98 (2 genuinely removed) → deletion allowed.
    stored = {f"documents::doc-{i}": 1.0 for i in range(100)}
    scanned = [f"documents::doc-{i}" for i in range(98)]
    deletes = []
    _run_flow(tmp_path, scanned_doc_ids=scanned, stored_mtimes=stored,
              registry_count=100, deletes_capture=deletes)
    assert set(deletes) == {"documents::doc-98", "documents::doc-99"}


def test_tiny_corpus_not_blocked_by_ratio(tmp_path):
    # 3 docs, 2 removed → over 50% but under the min-docs floor, so allowed.
    stored = {f"documents::doc-{i}": 1.0 for i in range(3)}
    deletes = []
    _run_flow(tmp_path, scanned_doc_ids=["documents::doc-0"],
              stored_mtimes=stored, registry_count=3, deletes_capture=deletes)
    assert set(deletes) == {"documents::doc-1", "documents::doc-2"}


def test_empty_registry_does_not_wipe_populated_table(tmp_path):
    # registry count 0 + 500 docs in table → refuse auto-wipe (lost registry).
    stored = {f"documents::doc-{i}": 1.0 for i in range(500)}
    scanned = list(stored.keys())
    active_store, delete_mock, ldb = _run_flow(
        tmp_path, scanned_doc_ids=scanned, stored_mtimes=stored, registry_count=0,
    )
    active_store.reset_table.assert_not_called()  # no drop/rebuild


def test_empty_registry_wipe_allowed_with_explicit_optin(tmp_path):
    stored = {f"documents::doc-{i}": 1.0 for i in range(500)}
    scanned = list(stored.keys())
    with patch("flow_index_vault.lancedb" if False else "lancedb.connect", MagicMock()):
        active_store, delete_mock, ldb = _run_flow(
            tmp_path, scanned_doc_ids=scanned, stored_mtimes=stored, registry_count=0,
            config_extra={"safety": {"allow_registry_empty_wipe": True,
                                     "registry_wipe_max_docs": 100}},
        )
    # When opted in, the table is reconstructed via LanceDBStore(...) after drop.
    assert ldb.called


def test_partial_source_deletion_caught_per_source_not_global(tmp_path):
    """The 'only part of the data is missing' case.

    comm has 200 docs indexed but its scan returns only 90 (postgres partial);
    documents (100 docs) scans fully. to_delete = 110 comm docs. That is only
    110/300 = 37% of the WHOLE corpus — a global ratio guard would let it
    through and delete 110 good comm docs. The per-source guard sees 110/200 =
    55% of comm and blocks it, while leaving documents' (zero) deletes alone.
    """
    stored = {}
    stored.update({f"comm_messages::c{i}": 1.0 for i in range(200)})
    stored.update({f"documents::d{i}": 1.0 for i in range(100)})

    # Two sources: documents scans all 100; comm returns only 90 of 200.
    class _Src:
        def __init__(self, name, ids):
            self.name = name
            self._ids = ids

        def scan(self):
            return iter([
                SourceRecord(
                    doc_id=i, natural_key=f"{i}.txt", source_type="txt",
                    mtime=1.0, size=4,
                    metadata={"abs_path": str(tmp_path / f"{i}.txt"), "ext": "txt"},
                )
                for i in self._ids
            ])

        def set_ocr_provider(self, p):
            return None

        def set_media_provider(self, p):
            return None

        def close(self):
            return None

    docs_src = _Src("documents", [f"d{i}" for i in range(100)])
    comm_src = _Src("comm_messages", [f"c{i}" for i in range(90)])  # partial!

    def build_fake(cfg, registry=None, pdf_config=None):
        return {"documents": docs_src, "comm_messages": comm_src}[cfg["name"]]

    active_store = MagicMock()
    active_store.list_doc_ids.return_value = list(stored.keys())
    active_store.list_doc_mtimes.return_value = dict(stored)
    active_store.count_chunks.return_value = len(stored)
    active_store.fts_available.return_value = True

    fake_registry = MagicMock()
    fake_registry.count.return_value = 300
    fake_taxonomy = MagicMock()
    fake_taxonomy.count.return_value = 0

    config = {
        "index_root": str(tmp_path / "index"),
        "sources": [
            {"type": "filesystem", "name": "documents", "root": str(tmp_path)},
            {"type": "filesystem", "name": "comm_messages", "root": str(tmp_path)},
        ],
        "chunking": {"max_chars": 1800, "overlap": 200, "semantic": {"enabled": False}},
        "enrichment": {"enabled": False},
        "ocr": {"enabled": False},
        "media": {"enabled": False},
        "lancedb": {"table": "chunks"},
        "pdf": {},
        "logging": {"level": "WARNING"},
    }

    deletes = []
    delete_mock = MagicMock(side_effect=lambda ids: deletes.extend(ids))

    with patch("flow_index_vault.get_run_logger", return_value=MagicMock()):
        with patch("flow_index_vault.load_config", return_value=config):
            with patch("flow_index_vault.open_store_with_recovery", return_value=active_store):
                with patch("flow_index_vault.LanceDBStore", return_value=active_store):
                    with patch("flow_index_vault.DocIDStore", return_value=fake_registry):
                        with patch("flow_index_vault.build_embed_provider", return_value=MagicMock()):
                            with patch("flow_index_vault.build_ocr_provider", return_value=None):
                                with patch("flow_index_vault.build_media_provider", return_value=None):
                                    with patch("sources.build_source", side_effect=build_fake):
                                        with patch("core.taxonomy.load_taxonomy_store", return_value=fake_taxonomy):
                                            with patch("flow_index_vault._process_docs", return_value=[]):
                                                with patch("flow_index_vault.delete_docs_task", delete_mock):
                                                    with patch("flow_index_vault.index_stats_task"):
                                                        with patch("flow_index_vault.write_index_metadata_task"):
                                                            index_vault_flow.fn("dummy.yaml")

    # The 110 missing comm docs must NOT be deleted (partial-scan protection).
    assert not any(d.startswith("comm_messages::") for d in deletes), \
        f"partial comm scan must not delete comm docs, got {len(deletes)} deletes"
