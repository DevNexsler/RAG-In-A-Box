# Persistent Doc ID Integration Tests
#
# Tests the full indexing pipeline with persistent document IDs:
#   scan_vault_task (with file renaming) → diff_index_task → process_doc_task
#   → LanceDB store → search → verify audit trail
#
# Uses REAL LanceDB, REAL Prefect tasks, REAL extractors, but MOCK embeddings
# so no API keys are needed.
#
# Run with: pytest tests/test_doc_id_integration.int.test.py -v
#
# Edge cases covered:
#   1. New file → gets ID, renamed on disk, indexed with persistent ID
#   2. Re-scan unchanged vault → idempotent, no re-indexing
#   3. File moved between folders → keeps ID, rel_path updated, re-indexed
#   4. File deleted → chunks removed, registry cleaned, audit logged
#   5. ID collision (two files same @XXXXX@) → second gets fresh ID
#   6. Someone strips ID from filename → gets fresh ID, old orphan cleaned up
#   7. New file added to existing vault → only new file indexed
#   8. File content modified → re-indexed with same ID
#   9. Audit trail traces full lifecycle
#  10. rel_path filter (doc_id_prefix) works for path-based browsing

import logging
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# These imports require prefect + llama_index — skip entire module if missing
pytest.importorskip("prefect")
pytest.importorskip("llama_index")

from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.node_parser import SentenceSplitter

from doc_id_store import (
    DocIDStore,
    extract_id_from_filename,
    inject_id_into_filename,
    strip_id_from_filename,
)
from extractors import extract_markdown, extract_title, normalize_tags, derive_folder
from flow_index_vault import (
    scan_vault_task,
    diff_index_task,
    process_doc_task,
    delete_docs_task,
    _RUNTIME,
)
from lancedb_store import LanceDBStore
from providers.embed.base import EmbedProvider


# ---------------------------------------------------------------------------
# Mock providers
# ---------------------------------------------------------------------------


class MockEmbedProvider(EmbedProvider):
    """Deterministic 768d vectors for offline testing."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 768 for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [0.1] * 768


# ---------------------------------------------------------------------------
# Test documents
# ---------------------------------------------------------------------------

_MD_RECIPE = """\
---
title: Korean Kimchi Recipe
tags: [cooking, korean]
status: active
---

# Korean Kimchi Recipe

## Ingredients

- 1 large napa cabbage
- Korean red pepper flakes (gochugaru)
- Fish sauce
- Garlic and ginger

## Instructions

Salt the cabbage thoroughly and let sit for two hours.
Mix gochugaru, fish sauce, garlic, and ginger into a paste.
Apply paste between cabbage leaves. Ferment for 3-5 days.
"""

_MD_NOTES = """\
---
title: Meeting Notes March 2026
tags: [meeting, planning]
status: draft
---

# Meeting Notes

## Action Items

- Review Q1 budget proposal by Friday
- Schedule design review for the new dashboard
- Follow up with legal on compliance requirements
"""

_MD_REPORT = """\
---
title: Infrastructure Report
tags: [devops, infrastructure]
status: active
---

# Infrastructure Report

## Current State

All production servers running on Kubernetes 1.28.
Database cluster healthy with three replicas.
CDN cache hit ratio at 94%.
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_test_logger = logging.getLogger("test_doc_id_integration")


@pytest.fixture(autouse=True)
def _mock_prefect_logger():
    """Patch get_run_logger so tasks can run outside a Prefect flow context."""
    with patch("flow_index_vault.get_run_logger", return_value=_test_logger):
        yield


@pytest.fixture
def vault_and_index(tmp_path):
    """Create a temp vault with markdown files and a separate index directory."""
    vault = tmp_path / "vault"
    vault.mkdir()
    index = tmp_path / "index"
    index.mkdir()
    return vault, index


def _setup_runtime(index_dir: Path, vault_dir: Path):
    """Set up _RUNTIME with mock providers — mirrors index_vault_flow setup."""
    _RUNTIME.clear()
    store = LanceDBStore(str(index_dir), "test_chunks")
    doc_id_store = DocIDStore(index_dir / "doc_registry.db")
    embed = MockEmbedProvider()
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)

    _RUNTIME["store"] = store
    _RUNTIME["doc_id_store"] = doc_id_store
    _RUNTIME["embed_provider"] = embed
    _RUNTIME["splitter"] = splitter
    _RUNTIME["ocr_provider"] = None
    _RUNTIME["llm_generator"] = None
    _RUNTIME["taxonomy_store"] = None
    _RUNTIME["semantic_splitter"] = None
    _RUNTIME["semantic_threshold"] = 0
    _RUNTIME["config"] = {
        "documents_root": str(vault_dir),
        "index_root": str(index_dir),
    }
    return store, doc_id_store, embed


def _teardown_runtime():
    doc_id_store = _RUNTIME.get("doc_id_store")
    if doc_id_store:
        doc_id_store.close()
    _RUNTIME.clear()


# ---------------------------------------------------------------------------
# Test 1: New file gets ID assigned, renamed, and indexed
# ---------------------------------------------------------------------------


class TestNewFileIndexing:
    def test_new_file_gets_persistent_id(self, vault_and_index):
        vault, index = vault_and_index
        (vault / "recipe.md").write_text(_MD_RECIPE)

        store, doc_id_store, embed = _setup_runtime(index, vault)
        try:
            records = scan_vault_task.fn(vault, ["**/*.md"], [])

            assert len(records) == 1
            rec = records[0]
            # doc_id is a 5-char base-62 ID, not a path
            assert len(rec["doc_id"]) == 5
            assert rec["doc_id"] == "00001"
            # rel_path contains the @XXXXX@ tag
            assert "@00001@" in rec["rel_path"]
            # Original file was renamed on disk
            assert not (vault / "recipe.md").exists()
            assert (vault / "recipe@00001@.md").exists()
            # Registry has the mapping
            assert doc_id_store.lookup_path("00001") == "recipe@00001@.md"

            # Now index it
            process_doc_task.fn(rec)

            # Verify in LanceDB
            doc_ids = store.list_doc_ids()
            assert "00001" in doc_ids
            chunks = store.get_doc_chunks("00001")
            assert len(chunks) > 0
            # Every chunk has rel_path in metadata
            for chunk in chunks:
                assert chunk.rel_path == "recipe@00001@.md"
                assert chunk.title == "Korean Kimchi Recipe"
        finally:
            _teardown_runtime()

    def test_multiple_files_get_sequential_ids(self, vault_and_index):
        vault, index = vault_and_index
        (vault / "recipe.md").write_text(_MD_RECIPE)
        (vault / "notes.md").write_text(_MD_NOTES)
        sub = vault / "reports"
        sub.mkdir()
        (sub / "infra.md").write_text(_MD_REPORT)

        store, doc_id_store, embed = _setup_runtime(index, vault)
        try:
            records = scan_vault_task.fn(vault, ["**/*.md"], [])

            assert len(records) == 3
            ids = sorted(r["doc_id"] for r in records)
            assert ids == ["00001", "00002", "00003"]

            # Index all
            for rec in records:
                process_doc_task.fn(rec)

            assert len(store.list_doc_ids()) == 3
        finally:
            _teardown_runtime()


# ---------------------------------------------------------------------------
# Test 2: Re-scan unchanged vault is idempotent
# ---------------------------------------------------------------------------


class TestIdempotentRescan:
    def test_rescan_no_changes(self, vault_and_index):
        vault, index = vault_and_index
        (vault / "recipe.md").write_text(_MD_RECIPE)

        store, doc_id_store, embed = _setup_runtime(index, vault)
        try:
            # First scan + index
            records1 = scan_vault_task.fn(vault, ["**/*.md"], [])
            for rec in records1:
                process_doc_task.fn(rec)
            stored_mtimes = store.list_doc_mtimes()

            # Second scan
            records2 = scan_vault_task.fn(vault, ["**/*.md"], [])
            to_add, to_delete = diff_index_task.fn(records2, stored_mtimes)

            # Nothing changed
            assert len(to_add) == 0
            assert len(to_delete) == 0
            # Same IDs
            assert records1[0]["doc_id"] == records2[0]["doc_id"]
            assert records1[0]["rel_path"] == records2[0]["rel_path"]
        finally:
            _teardown_runtime()


# ---------------------------------------------------------------------------
# Test 3: File moved between folders
# ---------------------------------------------------------------------------


class TestFileMoved:
    def test_moved_file_keeps_id(self, vault_and_index):
        vault, index = vault_and_index
        (vault / "recipe@abc12@.md").write_text(_MD_RECIPE)

        store, doc_id_store, embed = _setup_runtime(index, vault)
        try:
            # First scan + index
            records1 = scan_vault_task.fn(vault, ["**/*.md"], [])
            assert records1[0]["doc_id"] == "abc12"
            process_doc_task.fn(records1[0])

            # Move file to subfolder
            archive = vault / "archive"
            archive.mkdir()
            (vault / "recipe@abc12@.md").rename(archive / "recipe@abc12@.md")

            # Second scan
            records2 = scan_vault_task.fn(vault, ["**/*.md"], [])
            assert len(records2) == 1
            assert records2[0]["doc_id"] == "abc12"  # same ID
            assert records2[0]["rel_path"] == "archive/recipe@abc12@.md"

            # Registry updated
            assert doc_id_store.lookup_path("abc12") == "archive/recipe@abc12@.md"

            # Diff detects the path change via mtime or new doc_id
            stored_mtimes = store.list_doc_mtimes()
            to_add, to_delete = diff_index_task.fn(records2, stored_mtimes)
            # The old doc_id "abc12" is in both scan and store, mtime unchanged
            # so it should NOT re-index (mtime didn't change)

            # Audit trail shows the move
            moved = doc_id_store.audit_log(event="moved")
            assert len(moved) == 1
            assert moved[0]["doc_id"] == "abc12"
            assert "archive/" in moved[0]["rel_path"]
        finally:
            _teardown_runtime()


# ---------------------------------------------------------------------------
# Test 4: File deleted
# ---------------------------------------------------------------------------


class TestFileDeleted:
    def test_deleted_file_cleaned_up(self, vault_and_index):
        vault, index = vault_and_index
        (vault / "recipe@abc12@.md").write_text(_MD_RECIPE)

        store, doc_id_store, embed = _setup_runtime(index, vault)
        try:
            # Scan + index
            records = scan_vault_task.fn(vault, ["**/*.md"], [])
            process_doc_task.fn(records[0])
            assert len(store.list_doc_ids()) == 1

            # Delete the file
            (vault / "recipe@abc12@.md").unlink()

            # Re-scan + diff
            records2 = scan_vault_task.fn(vault, ["**/*.md"], [])
            stored_mtimes = store.list_doc_mtimes()
            to_add, to_delete = diff_index_task.fn(records2, stored_mtimes)

            assert to_delete == ["abc12"]
            assert len(to_add) == 0

            # Delete from store
            delete_docs_task.fn(to_delete)
            assert len(store.list_doc_ids()) == 0

            # Clean registry
            for did in to_delete:
                doc_id_store.delete(did)

            assert doc_id_store.count() == 0

            # Audit trail
            deleted = doc_id_store.audit_log(event="deleted")
            assert len(deleted) == 1
            assert deleted[0]["doc_id"] == "abc12"
        finally:
            _teardown_runtime()


# ---------------------------------------------------------------------------
# Test 5: ID collision
# ---------------------------------------------------------------------------


class TestIdCollision:
    def test_collision_resolved_and_both_indexed(self, vault_and_index):
        vault, index = vault_and_index
        # Two files with the same ID — copy-paste accident
        (vault / "recipe@abc12@.md").write_text(_MD_RECIPE)
        (vault / "notes@abc12@.md").write_text(_MD_NOTES)

        store, doc_id_store, embed = _setup_runtime(index, vault)
        try:
            records = scan_vault_task.fn(vault, ["**/*.md"], [])

            assert len(records) == 2
            ids = [r["doc_id"] for r in records]
            assert len(set(ids)) == 2  # all unique

            # One keeps abc12, other gets fresh ID
            assert "abc12" in ids
            fresh = [i for i in ids if i != "abc12"][0]
            assert len(fresh) == 5

            # The reassigned file was renamed on disk
            reassigned = [r for r in records if r["doc_id"] != "abc12"][0]
            assert f"@{fresh}@" in reassigned["rel_path"]
            assert "@abc12@" not in reassigned["rel_path"]

            # Both can be indexed without conflict
            for rec in records:
                process_doc_task.fn(rec)

            assert len(store.list_doc_ids()) == 2

            # Audit trail records the collision
            collisions = doc_id_store.audit_log(event="collision")
            assert len(collisions) == 1
            assert collisions[0]["doc_id"] == "abc12"
            assert "already claimed by" in collisions[0]["detail"]
        finally:
            _teardown_runtime()


# ---------------------------------------------------------------------------
# Test 6: ID stripped from filename
# ---------------------------------------------------------------------------


class TestIdStripped:
    def test_stripped_file_gets_new_id(self, vault_and_index):
        vault, index = vault_and_index

        store, doc_id_store, embed = _setup_runtime(index, vault)
        try:
            # First scan: file with ID
            (vault / "recipe@abc12@.md").write_text(_MD_RECIPE)
            records1 = scan_vault_task.fn(vault, ["**/*.md"], [])
            process_doc_task.fn(records1[0])
            assert records1[0]["doc_id"] == "abc12"

            # User strips the ID
            (vault / "recipe@abc12@.md").rename(vault / "recipe.md")

            # Re-scan
            records2 = scan_vault_task.fn(vault, ["**/*.md"], [])
            assert len(records2) == 1
            new_id = records2[0]["doc_id"]
            assert new_id != "abc12"
            assert len(new_id) == 5

            # Diff: old abc12 is gone, new ID is new
            stored_mtimes = store.list_doc_mtimes()
            to_add, to_delete = diff_index_task.fn(records2, stored_mtimes)
            assert "abc12" in to_delete
            assert len(to_add) == 1
            assert to_add[0]["doc_id"] == new_id

            # Process new, delete old
            process_doc_task.fn(to_add[0])
            delete_docs_task.fn(to_delete)
            for did in to_delete:
                doc_id_store.delete(did)

            # Only new ID in store
            assert store.list_doc_ids() == [new_id]
        finally:
            _teardown_runtime()


# ---------------------------------------------------------------------------
# Test 7: New file added to existing vault
# ---------------------------------------------------------------------------


class TestIncrementalAdd:
    def test_only_new_file_indexed(self, vault_and_index):
        vault, index = vault_and_index
        (vault / "recipe.md").write_text(_MD_RECIPE)

        store, doc_id_store, embed = _setup_runtime(index, vault)
        try:
            # First scan + index
            records1 = scan_vault_task.fn(vault, ["**/*.md"], [])
            for rec in records1:
                process_doc_task.fn(rec)
            stored_mtimes = store.list_doc_mtimes()

            # Add a new file
            (vault / "notes.md").write_text(_MD_NOTES)

            # Second scan + diff
            records2 = scan_vault_task.fn(vault, ["**/*.md"], [])
            to_add, to_delete = diff_index_task.fn(records2, stored_mtimes)

            # Only the new file needs indexing
            assert len(to_add) == 1
            assert "notes" in to_add[0]["rel_path"].lower()
            assert len(to_delete) == 0

            process_doc_task.fn(to_add[0])
            assert len(store.list_doc_ids()) == 2
        finally:
            _teardown_runtime()


# ---------------------------------------------------------------------------
# Test 8: File content modified
# ---------------------------------------------------------------------------


class TestContentModified:
    def test_modified_file_reindexed_same_id(self, vault_and_index):
        vault, index = vault_and_index

        store, doc_id_store, embed = _setup_runtime(index, vault)
        try:
            # Create and index
            (vault / "recipe.md").write_text(_MD_RECIPE)
            records1 = scan_vault_task.fn(vault, ["**/*.md"], [])
            process_doc_task.fn(records1[0])
            original_id = records1[0]["doc_id"]
            stored_mtimes = store.list_doc_mtimes()

            # Modify the file (need to find it by its new name)
            renamed_file = [f for f in vault.glob("*.md")][0]
            time.sleep(0.05)  # ensure mtime changes
            renamed_file.write_text(_MD_RECIPE + "\n## New Section\n\nAdditional content here.\n")

            # Re-scan + diff
            records2 = scan_vault_task.fn(vault, ["**/*.md"], [])
            to_add, to_delete = diff_index_task.fn(records2, stored_mtimes)

            # Same ID, flagged for re-index due to mtime change
            assert len(to_add) == 1
            assert to_add[0]["doc_id"] == original_id
            assert len(to_delete) == 0

            process_doc_task.fn(to_add[0])

            # Still one doc, same ID
            assert store.list_doc_ids() == [original_id]
        finally:
            _teardown_runtime()


# ---------------------------------------------------------------------------
# Test 9: Full lifecycle audit trail
# ---------------------------------------------------------------------------


class TestFullLifecycleAudit:
    def test_complete_audit_trail(self, vault_and_index):
        vault, index = vault_and_index

        store, doc_id_store, embed = _setup_runtime(index, vault)
        try:
            # 1. Create file → registered
            (vault / "recipe.md").write_text(_MD_RECIPE)
            records = scan_vault_task.fn(vault, ["**/*.md"], [])
            doc_id = records[0]["doc_id"]
            process_doc_task.fn(records[0])

            # 2. Move file → moved
            archive = vault / "archive"
            archive.mkdir()
            current_file = [f for f in vault.glob("*.md")][0]
            current_file.rename(archive / current_file.name)
            scan_vault_task.fn(vault, ["**/*.md"], [])

            # 3. Delete file → deleted
            moved_file = [f for f in archive.glob("*.md")][0]
            moved_file.unlink()
            scan_vault_task.fn(vault, ["**/*.md"], [])
            # Simulate flow cleanup
            doc_id_store.delete(doc_id)

            # Verify full audit trail
            trail = doc_id_store.audit_log(doc_id=doc_id)
            events = [e["event"] for e in trail]
            assert events == ["deleted", "moved", "registered"]

            # registered: has the original path
            reg = trail[2]
            assert "@" in reg["rel_path"]
            assert reg["old_path"] == ""

            # moved: has old and new paths
            move = trail[1]
            assert "archive/" in move["rel_path"]
            assert move["old_path"] != ""
            assert "archive/" not in move["old_path"]

            # deleted: has the last known path
            delete = trail[0]
            assert "archive/" in delete["rel_path"]
        finally:
            _teardown_runtime()


# ---------------------------------------------------------------------------
# Test 10: rel_path filter (doc_id_prefix) for path-based browsing
# ---------------------------------------------------------------------------


class TestRelPathFilter:
    def test_doc_id_prefix_filters_on_rel_path(self, vault_and_index):
        vault, index = vault_and_index
        # Create files in different folders
        recipes = vault / "recipes"
        recipes.mkdir()
        reports = vault / "reports"
        reports.mkdir()
        (recipes / "kimchi.md").write_text(_MD_RECIPE)
        (reports / "infra.md").write_text(_MD_REPORT)

        store, doc_id_store, embed = _setup_runtime(index, vault)
        try:
            records = scan_vault_task.fn(vault, ["**/*.md"], [])
            for rec in records:
                process_doc_task.fn(rec)
            store.create_fts_index()

            # Build a WHERE clause filtering by rel_path prefix
            where = store._build_where_clause(doc_id_prefix="recipes/")
            assert "rel_path" in where
            assert "recipes/" in where

            # Search with prefix filter — should only find recipe
            hits = store.vector_search(
                embed.embed_query("korean food"),
                top_k=10,
                where=where,
            )
            # All hits should be from recipes/ folder
            for h in hits:
                assert "recipes/" in h.rel_path
        finally:
            _teardown_runtime()


# ---------------------------------------------------------------------------
# Test 11: Copy-paste of deleted file — retired ID must not be reused
# ---------------------------------------------------------------------------


class TestRetiredIdReuse:
    def test_copy_paste_deleted_file_gets_fresh_id(self, vault_and_index):
        """Full E2E: delete a doc, copy-paste with stale ID, verify fresh ID assigned."""
        vault, index = vault_and_index

        store, doc_id_store, embed = _setup_runtime(index, vault)
        try:
            # 1. Create and index a file
            (vault / "recipe.md").write_text(_MD_RECIPE)
            records1 = scan_vault_task.fn(vault, ["**/*.md"], [])
            original_id = records1[0]["doc_id"]
            process_doc_task.fn(records1[0])
            assert original_id in store.list_doc_ids()

            # 2. Delete the file and clean up (full flow simulation)
            renamed_file = [f for f in vault.glob("*.md")][0]
            renamed_file.unlink()
            stored_mtimes = store.list_doc_mtimes()
            records_empty = scan_vault_task.fn(vault, ["**/*.md"], [])
            to_add, to_delete = diff_index_task.fn(records_empty, stored_mtimes)
            assert to_delete == [original_id]
            delete_docs_task.fn(to_delete)
            doc_id_store.delete(original_id)

            # Verify: ID is retired, store is empty
            assert doc_id_store.is_retired(original_id)
            assert len(store.list_doc_ids()) == 0

            # 3. Someone copies an old backup with the stale @XXXXX@ ID
            (vault / f"recipe@{original_id}@.md").write_text(_MD_RECIPE)
            records2 = scan_vault_task.fn(vault, ["**/*.md"], [])

            assert len(records2) == 1
            new_id = records2[0]["doc_id"]
            # Must NOT reuse the retired ID
            assert new_id != original_id
            assert f"@{new_id}@" in records2[0]["rel_path"]
            assert f"@{original_id}@" not in records2[0]["rel_path"]

            # 4. Index the new file and verify it's searchable under the new ID
            process_doc_task.fn(records2[0])
            assert new_id in store.list_doc_ids()
            assert original_id not in store.list_doc_ids()

            # 5. Audit trail records the retired-ID collision
            collisions = doc_id_store.audit_log(event="collision")
            assert len(collisions) >= 1
            retired_collision = [c for c in collisions if "retired" in c["detail"]]
            assert len(retired_collision) == 1
            assert original_id in retired_collision[0]["doc_id"]
        finally:
            _teardown_runtime()


class TestZeroByteFiles:
    def test_zero_byte_file_is_ignored_and_does_not_burn_doc_id(self, vault_and_index):
        vault, index = vault_and_index

        store, doc_id_store, embed = _setup_runtime(index, vault)
        try:
            empty_pdf = vault / "empty.pdf"
            empty_pdf.write_bytes(b"")

            empty_records = scan_vault_task.fn(vault, ["**/*.pdf"], [])

            assert empty_records == []
            assert empty_pdf.exists()
            assert empty_pdf.name == "empty.pdf"

            (vault / "recipe.md").write_text(_MD_RECIPE)
            valid_records = scan_vault_task.fn(vault, ["**/*.pdf", "**/*.md"], [])

            assert len(valid_records) == 1
            assert valid_records[0]["doc_id"] == "00001"
            assert valid_records[0]["rel_path"] == "recipe@00001@.md"
            assert (vault / "recipe@00001@.md").exists()
        finally:
            _teardown_runtime()
