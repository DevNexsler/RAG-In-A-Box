"""Tests for doc_id_store: base-62 encoding, DocIDStore, filename helpers, and
end-to-end scan scenarios covering all real-world use cases.

Scenarios covered:
  1. Agent drops a new file (no ID) → indexer assigns ID and renames
  2. File with existing ID is re-indexed → ID preserved
  3. File moved to different folder → ID survives, rel_path updated
  4. File disappears between runs → deleted from store + registry
  5. Two files end up with the same @XXXXX@ ID (collision) → second gets new ID
  6. Someone strips the ID from a filename manually → gets a fresh ID
  7. Rename fails (read-only FS) → falls back gracefully
  8. Mix of new files, existing IDs, and collisions in one scan pass
  9. Counter persists across close/reopen
"""

import os
import stat
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from doc_id_store import (
    DocIDStore,
    _int_to_base62,
    _base62_to_int,
    extract_id_from_filename,
    inject_id_into_filename,
    strip_id_from_filename,
)


# =========================================================================
# Base-62 encoding
# =========================================================================


class TestBase62:
    def test_zero(self):
        assert _int_to_base62(0) == "00000"

    def test_one(self):
        assert _int_to_base62(1) == "00001"

    def test_62(self):
        # 62 in base-62 = "00010"
        assert _int_to_base62(62) == "00010"

    def test_roundtrip(self):
        for n in [0, 1, 61, 62, 63, 999, 12345, 916132831]:
            assert _base62_to_int(_int_to_base62(n)) == n

    def test_max_5_chars(self):
        # 62^5 - 1 = 916,132,831
        result = _int_to_base62(916132831)
        assert len(result) == 5
        assert result == "ZZZZZ"

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            _int_to_base62(-1)

    def test_charset_order(self):
        # 0-9, a-z, A-Z
        assert _int_to_base62(9) == "00009"
        assert _int_to_base62(10) == "0000a"
        assert _int_to_base62(35) == "0000z"
        assert _int_to_base62(36) == "0000A"
        assert _int_to_base62(61) == "0000Z"

    def test_monotonic(self):
        """Sequential IDs sort lexicographically."""
        ids = [_int_to_base62(n) for n in range(200)]
        for i in range(1, len(ids)):
            assert _base62_to_int(ids[i]) > _base62_to_int(ids[i - 1])


# =========================================================================
# Filename helpers
# =========================================================================


class TestExtractId:
    def test_with_id(self):
        assert extract_id_from_filename("recipe@00001@.md") == "00001"

    def test_without_id(self):
        assert extract_id_from_filename("recipe.md") is None

    def test_nested_path(self):
        assert extract_id_from_filename("notes/recipe@aB3Zx@.md") == "aB3Zx"

    def test_no_extension(self):
        assert extract_id_from_filename("README@00001@") == "00001"

    def test_multiple_dots(self):
        assert extract_id_from_filename("archive@00002@.tar.gz") == "00002"

    def test_wrong_length_short(self):
        assert extract_id_from_filename("file@0001@.md") is None

    def test_wrong_length_long(self):
        assert extract_id_from_filename("file@000001@.md") is None

    def test_invalid_chars(self):
        assert extract_id_from_filename("file@ab!cd@.md") is None

    def test_empty_string(self):
        assert extract_id_from_filename("") is None

    def test_at_signs_in_name_no_id(self):
        """@ signs that don't form a valid ID pattern are ignored."""
        assert extract_id_from_filename("user@email.md") is None

    def test_multiple_id_patterns(self):
        """If somehow two IDs exist, first match wins."""
        assert extract_id_from_filename("file@00001@@00002@.md") == "00001"


class TestInjectId:
    def test_normal(self):
        assert inject_id_into_filename("recipe.md", "00001") == "recipe@00001@.md"

    def test_no_extension(self):
        assert inject_id_into_filename("README", "0000a") == "README@0000a@"

    def test_pdf(self):
        assert inject_id_into_filename("report.pdf", "abc12") == "report@abc12@.pdf"

    def test_compound_extension(self):
        # Only last suffix treated as extension
        assert inject_id_into_filename("data.tar.gz", "00001") == "data.tar@00001@.gz"

    def test_dotfile(self):
        assert inject_id_into_filename(".gitignore", "00001") == ".gitignore@00001@"

    def test_spaces_in_name(self):
        assert inject_id_into_filename("my notes.md", "00001") == "my notes@00001@.md"


class TestStripId:
    def test_strip_existing(self):
        assert strip_id_from_filename("recipe@00001@.md") == "recipe.md"

    def test_strip_no_id(self):
        assert strip_id_from_filename("recipe.md") == "recipe.md"

    def test_strip_no_extension(self):
        assert strip_id_from_filename("README@00001@") == "README"

    def test_strip_nested(self):
        assert strip_id_from_filename("notes/recipe@aB3Zx@.md") == "notes/recipe.md"

    def test_roundtrip_inject_strip(self):
        """inject then strip recovers original name."""
        original = "my-doc.pdf"
        injected = inject_id_into_filename(original, "Xz9Aa")
        assert strip_id_from_filename(injected) == original

    def test_roundtrip_strip_inject(self):
        """strip then inject produces clean single-ID name."""
        tagged = "recipe@00001@.md"
        clean = strip_id_from_filename(tagged)
        re_tagged = inject_id_into_filename(clean, "00002")
        assert re_tagged == "recipe@00002@.md"
        # Only one ID in the result
        assert re_tagged.count("@") == 2


# =========================================================================
# DocIDStore CRUD
# =========================================================================


@pytest.fixture
def id_store(tmp_path):
    db_path = tmp_path / "doc_registry.db"
    s = DocIDStore(db_path)
    yield s
    s.close()


class TestDocIDStore:
    def test_next_id_sequential(self, id_store):
        id1 = id_store.next_id()
        id2 = id_store.next_id()
        id3 = id_store.next_id()
        assert id1 == "00001"
        assert id2 == "00002"
        assert id3 == "00003"

    def test_register_and_lookup(self, id_store):
        id_store.register("00001", "notes/recipe.md")
        assert id_store.lookup_path("00001") == "notes/recipe.md"
        assert id_store.lookup_id("notes/recipe.md") == "00001"

    def test_lookup_missing(self, id_store):
        assert id_store.lookup_path("XXXXX") is None
        assert id_store.lookup_id("nonexistent.md") is None

    def test_update_path(self, id_store):
        id_store.register("00001", "old/path.md")
        id_store.update_path("00001", "new/path.md")
        assert id_store.lookup_path("00001") == "new/path.md"

    def test_delete(self, id_store):
        id_store.register("00001", "notes/recipe.md")
        id_store.delete("00001")
        assert id_store.lookup_path("00001") is None

    def test_delete_nonexistent_is_noop(self, id_store):
        """Deleting an ID that doesn't exist should not raise."""
        id_store.delete("XXXXX")  # should not raise

    def test_all_mappings(self, id_store):
        id_store.register("00001", "a.md")
        id_store.register("00002", "b.md")
        mappings = id_store.all_mappings()
        assert mappings == {"00001": "a.md", "00002": "b.md"}

    def test_count(self, id_store):
        assert id_store.count() == 0
        id_store.register("00001", "a.md")
        assert id_store.count() == 1
        id_store.register("00002", "b.md")
        assert id_store.count() == 2

    def test_register_upsert(self, id_store):
        """register() with same doc_id updates the path."""
        id_store.register("00001", "old.md")
        id_store.register("00001", "new.md")
        assert id_store.lookup_path("00001") == "new.md"
        assert id_store.count() == 1

    def test_persistence(self, tmp_path):
        """Data survives close + reopen."""
        db_path = tmp_path / "doc_registry.db"
        s1 = DocIDStore(db_path)
        s1.next_id()
        s1.register("00001", "notes/recipe.md")
        s1.close()

        s2 = DocIDStore(db_path)
        assert s2.lookup_path("00001") == "notes/recipe.md"
        # Counter should continue from where it left off
        id2 = s2.next_id()
        assert id2 == "00002"
        s2.close()

    def test_counter_survives_many_ids(self, tmp_path):
        """Counter doesn't reset even after hundreds of IDs."""
        db_path = tmp_path / "doc_registry.db"
        s = DocIDStore(db_path)
        for _ in range(150):
            s.next_id()
        s.close()

        s2 = DocIDStore(db_path)
        assert s2.next_id() == _int_to_base62(151)
        s2.close()


# =========================================================================
# Real-world scan scenarios
# =========================================================================


def _simulate_scan(vault_root: Path, id_store: DocIDStore) -> list[dict]:
    """Simulate the scan_vault_task logic for testing without Prefect deps.

    Mirrors the actual scan logic in flow_index_vault.py: walk files,
    assign IDs, rename, handle collisions — including audit logging.
    """
    records = []
    seen_ids: dict[str, str] = {}

    for dirpath, _dirnames, filenames in os.walk(vault_root):
        for fname in sorted(filenames):  # sorted for deterministic test order
            full_path = Path(dirpath) / fname
            if fname.endswith((".db", ".db-shm", ".db-wal", ".db-journal")):
                continue
            rel_str = str(full_path.relative_to(vault_root)).replace("\\", "/")

            try:
                st = full_path.stat()
            except OSError:
                continue

            existing_id = extract_id_from_filename(fname)
            if existing_id:
                # Check for retired ID first (deleted doc — must not reuse)
                if id_store.is_retired(existing_id):
                    retired = id_store.retired_info(existing_id)
                    last_path = retired["last_path"] if retired else "unknown"
                    id_store.log_event(
                        DocIDStore.COLLISION, existing_id, rel_str,
                        detail=f"retired ID, previously used by {last_path}",
                    )
                    existing_id = None
                elif existing_id in seen_ids:
                    # Collision — log it, strip old ID, will get new one below
                    id_store.log_event(
                        DocIDStore.COLLISION, existing_id, rel_str,
                        detail=f"already claimed by {seen_ids[existing_id]}",
                    )
                    existing_id = None
                else:
                    doc_id = existing_id
                    seen_ids[doc_id] = rel_str
                    stored_path = id_store.lookup_path(doc_id)
                    if stored_path != rel_str:
                        id_store.register(doc_id, rel_str)

            if existing_id is None:
                doc_id = id_store.next_id()
                clean_fname = strip_id_from_filename(fname)
                new_fname = inject_id_into_filename(clean_fname, doc_id)
                new_full_path = full_path.parent / new_fname
                try:
                    full_path.rename(new_full_path)
                except OSError as exc:
                    # Rename failed — log and fall back to path-based ID
                    id_store.log_event(
                        DocIDStore.RENAME_FAILED, doc_id, rel_str,
                        detail=str(exc),
                    )
                    doc_id = rel_str
                    id_store.register(doc_id, rel_str)
                    seen_ids[doc_id] = rel_str
                else:
                    full_path = new_full_path
                    rel_str = str(full_path.relative_to(vault_root)).replace("\\", "/")
                    id_store.register(doc_id, rel_str)
                    seen_ids[doc_id] = rel_str

            records.append({
                "doc_id": doc_id,
                "rel_path": rel_str,
                "abs_path": str(full_path.resolve()),
                "mtime": st.st_mtime,
                "size": st.st_size,
                "ext": full_path.suffix.lower().lstrip(".") or "bin",
            })
    return records


class TestScenarioNewFile:
    """Scenario 1: Agent drops a brand-new file with no ID."""

    def test_new_file_gets_id_and_renamed(self, tmp_path):
        (tmp_path / "recipe.md").write_text("# Recipe")
        store = DocIDStore(tmp_path / "reg.db")

        records = _simulate_scan(tmp_path, store)

        assert len(records) == 1
        rec = records[0]
        assert rec["doc_id"] == "00001"
        assert "@00001@" in rec["rel_path"]
        # Old file gone, new file present
        assert not (tmp_path / "recipe.md").exists()
        assert Path(rec["abs_path"]).exists()
        # Registry has it
        assert store.lookup_path("00001") == rec["rel_path"]
        store.close()

    def test_multiple_new_files_get_sequential_ids(self, tmp_path):
        (tmp_path / "a.md").write_text("A")
        (tmp_path / "b.md").write_text("B")
        (tmp_path / "c.pdf").write_text("C")
        store = DocIDStore(tmp_path / "reg.db")

        records = _simulate_scan(tmp_path, store)

        ids = sorted(r["doc_id"] for r in records)
        assert ids == ["00001", "00002", "00003"]
        assert store.count() == 3
        store.close()

    def test_new_file_in_subfolder(self, tmp_path):
        sub = tmp_path / "notes" / "cooking"
        sub.mkdir(parents=True)
        (sub / "recipe.md").write_text("# Recipe")
        store = DocIDStore(tmp_path / "reg.db")

        records = _simulate_scan(tmp_path, store)

        assert len(records) == 1
        assert "notes/cooking/" in records[0]["rel_path"]
        assert "@00001@" in records[0]["rel_path"]
        store.close()


class TestScenarioExistingId:
    """Scenario 2: File already has @XXXXX@ in its name."""

    def test_existing_id_preserved(self, tmp_path):
        (tmp_path / "recipe@abc12@.md").write_text("# Recipe")
        store = DocIDStore(tmp_path / "reg.db")

        records = _simulate_scan(tmp_path, store)

        assert len(records) == 1
        assert records[0]["doc_id"] == "abc12"
        assert records[0]["rel_path"] == "recipe@abc12@.md"
        # File not renamed
        assert (tmp_path / "recipe@abc12@.md").exists()
        assert store.lookup_path("abc12") == "recipe@abc12@.md"
        store.close()

    def test_existing_id_no_counter_waste(self, tmp_path):
        """Files with existing IDs should not consume a counter value."""
        (tmp_path / "a@abc12@.md").write_text("A")
        (tmp_path / "b.md").write_text("B")  # this one needs an ID
        store = DocIDStore(tmp_path / "reg.db")

        records = _simulate_scan(tmp_path, store)

        # b.md should get 00001 (counter starts at 0, first next_id = 1)
        new_file_rec = [r for r in records if r["doc_id"] != "abc12"][0]
        assert new_file_rec["doc_id"] == "00001"
        store.close()


class TestScenarioFileMoved:
    """Scenario 3: File moved to a different folder between index runs."""

    def test_moved_file_keeps_id_path_updated(self, tmp_path):
        # First scan: file is in notes/
        notes = tmp_path / "notes"
        notes.mkdir()
        (notes / "recipe@abc12@.md").write_text("# Recipe")
        store = DocIDStore(tmp_path / "reg.db")

        records1 = _simulate_scan(tmp_path, store)
        assert records1[0]["rel_path"] == "notes/recipe@abc12@.md"
        assert store.lookup_path("abc12") == "notes/recipe@abc12@.md"

        # Move file to archive/
        archive = tmp_path / "archive"
        archive.mkdir()
        (notes / "recipe@abc12@.md").rename(archive / "recipe@abc12@.md")

        records2 = _simulate_scan(tmp_path, store)
        assert records2[0]["doc_id"] == "abc12"  # same ID
        assert records2[0]["rel_path"] == "archive/recipe@abc12@.md"  # new path
        assert store.lookup_path("abc12") == "archive/recipe@abc12@.md"
        store.close()


class TestScenarioFileDeleted:
    """Scenario 4: File disappears between index runs."""

    def test_deleted_file_cleaned_from_registry(self, tmp_path):
        (tmp_path / "recipe@abc12@.md").write_text("# Recipe")
        store = DocIDStore(tmp_path / "reg.db")

        records = _simulate_scan(tmp_path, store)
        assert store.count() == 1

        # Delete the file
        (tmp_path / "recipe@abc12@.md").unlink()

        # Simulate what index_vault_flow does: scan, diff, delete
        records2 = _simulate_scan(tmp_path, store)
        assert len(records2) == 0

        # In the real flow, deleted IDs are cleaned up after delete_docs_task
        old_ids = set(store.all_mappings().keys())
        scanned_ids = {r["doc_id"] for r in records2}
        for did in old_ids - scanned_ids:
            store.delete(did)

        assert store.count() == 0
        assert store.lookup_path("abc12") is None
        store.close()


class TestScenarioCollision:
    """Scenario 5: Two files end up with the same @XXXXX@ ID."""

    def test_duplicate_id_second_file_gets_new_id(self, tmp_path):
        # Both files have the same ID embedded — e.g. copy-paste accident
        (tmp_path / "recipe@abc12@.md").write_text("# Recipe")
        (tmp_path / "soup@abc12@.md").write_text("# Soup")
        store = DocIDStore(tmp_path / "reg.db")

        records = _simulate_scan(tmp_path, store)

        ids = [r["doc_id"] for r in records]
        assert len(ids) == 2
        # One keeps abc12, the other gets a fresh ID
        assert "abc12" in ids
        fresh_ids = [i for i in ids if i != "abc12"]
        assert len(fresh_ids) == 1
        assert len(fresh_ids[0]) == 5  # valid base-62 ID
        assert fresh_ids[0] != "abc12"

        # The reassigned file was renamed on disk with new ID
        reassigned = [r for r in records if r["doc_id"] != "abc12"][0]
        assert f"@{reassigned['doc_id']}@" in reassigned["rel_path"]
        # Old @abc12@ was stripped from the renamed file
        assert "@abc12@" not in reassigned["rel_path"]

        assert store.count() == 2
        store.close()

    def test_collision_in_different_folders(self, tmp_path):
        """Collision across folders: both notes/ and archive/ have same ID."""
        notes = tmp_path / "notes"
        archive = tmp_path / "archive"
        notes.mkdir()
        archive.mkdir()
        (notes / "a@abc12@.md").write_text("A")
        (archive / "b@abc12@.md").write_text("B")
        store = DocIDStore(tmp_path / "reg.db")

        records = _simulate_scan(tmp_path, store)

        ids = [r["doc_id"] for r in records]
        assert len(set(ids)) == 2  # all unique
        assert "abc12" in ids
        store.close()


class TestScenarioStrippedId:
    """Scenario 6: Someone manually strips the ID from a filename."""

    def test_stripped_id_gets_fresh_assignment(self, tmp_path):
        store = DocIDStore(tmp_path / "reg.db")

        # First scan: file has ID
        (tmp_path / "recipe@abc12@.md").write_text("# Recipe v1")
        records1 = _simulate_scan(tmp_path, store)
        assert records1[0]["doc_id"] == "abc12"

        # User manually renames: strips the ID
        (tmp_path / "recipe@abc12@.md").rename(tmp_path / "recipe.md")

        records2 = _simulate_scan(tmp_path, store)
        assert len(records2) == 1
        new_id = records2[0]["doc_id"]
        # Got a fresh ID (not abc12)
        assert new_id != "abc12"
        assert f"@{new_id}@" in records2[0]["rel_path"]

        # Old abc12 is still in registry (orphaned), cleaned up by diff/delete step
        # New ID is registered
        assert store.lookup_path(new_id) is not None
        store.close()


class TestScenarioRenameFails:
    """Scenario 7: File rename fails (e.g. read-only filesystem)."""

    def test_rename_failure_falls_back_to_path_id(self, tmp_path):
        (tmp_path / "recipe.md").write_text("# Recipe")
        store = DocIDStore(tmp_path / "reg.db")

        # Patch Path.rename to fail
        original_rename = Path.rename

        def failing_rename(self, target):
            if "recipe" in str(self):
                raise OSError("Permission denied")
            return original_rename(self, target)

        with patch.object(Path, "rename", failing_rename):
            records = _simulate_scan(tmp_path, store)

        assert len(records) == 1
        # Falls back to using rel_path as doc_id
        assert records[0]["doc_id"] == "recipe.md"
        # File was NOT renamed on disk
        assert (tmp_path / "recipe.md").exists()
        store.close()


class TestScenarioMixedScan:
    """Scenario 8: Mix of new files, existing IDs, and collisions in one pass."""

    def test_mixed_vault(self, tmp_path):
        # Existing with ID
        (tmp_path / "notes@aaa11@.md").write_text("Notes")
        # New file, no ID
        (tmp_path / "report.pdf").write_text("PDF")
        # Collision pair
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "dup1@bbb22@.md").write_text("Dup1")
        (tmp_path / "dup2@bbb22@.md").write_text("Dup2")

        store = DocIDStore(tmp_path / "reg.db")
        records = _simulate_scan(tmp_path, store)

        ids = [r["doc_id"] for r in records]
        assert len(ids) == 4
        assert len(set(ids)) == 4  # ALL unique

        # Existing ID preserved
        assert "aaa11" in ids
        # One of the collision pair keeps bbb22
        assert "bbb22" in ids

        # New file (report.pdf) got an ID
        report_rec = [r for r in records if "report" in r["rel_path"]][0]
        assert len(report_rec["doc_id"]) == 5
        assert "@" in report_rec["rel_path"]

        assert store.count() == 4
        store.close()


class TestScenarioRescan:
    """Scenario 9: Re-scan after initial indexing — idempotent behavior."""

    def test_rescan_is_idempotent(self, tmp_path):
        (tmp_path / "a.md").write_text("A")
        (tmp_path / "b.md").write_text("B")
        store = DocIDStore(tmp_path / "reg.db")

        records1 = _simulate_scan(tmp_path, store)
        # Second scan of the same vault
        records2 = _simulate_scan(tmp_path, store)

        # Same IDs, same paths
        r1 = {r["doc_id"]: r["rel_path"] for r in records1}
        r2 = {r["doc_id"]: r["rel_path"] for r in records2}
        assert r1 == r2

        # Counter didn't advance on second scan (no new files)
        store.close()

    def test_add_file_between_scans(self, tmp_path):
        (tmp_path / "a.md").write_text("A")
        store = DocIDStore(tmp_path / "reg.db")

        records1 = _simulate_scan(tmp_path, store)
        assert len(records1) == 1

        # Add another file
        (tmp_path / "b.md").write_text("B")
        records2 = _simulate_scan(tmp_path, store)
        assert len(records2) == 2

        # First file kept its ID
        a_id_1 = [r for r in records1 if "a" in r["rel_path"].lower().split("@")[0]][0]["doc_id"]
        a_id_2 = [r for r in records2 if "a" in r["rel_path"].lower().split("@")[0]][0]["doc_id"]
        assert a_id_1 == a_id_2

        # New file got a different ID
        b_rec = [r for r in records2 if "b" in r["rel_path"].lower().split("@")[0]][0]
        assert b_rec["doc_id"] != a_id_1
        store.close()


# =========================================================================
# Audit log
# =========================================================================


class TestAuditLogBasic:
    """Unit tests for the audit log table and query API."""

    def test_register_logs_registered_event(self, id_store):
        id_store.register("00001", "recipe.md")
        log = id_store.audit_log()
        assert len(log) == 1
        assert log[0]["event"] == "registered"
        assert log[0]["doc_id"] == "00001"
        assert log[0]["rel_path"] == "recipe.md"

    def test_register_same_path_no_duplicate_log(self, id_store):
        """Re-registering the same doc_id + same path should not add a log entry."""
        id_store.register("00001", "recipe.md")
        id_store.register("00001", "recipe.md")  # same path
        log = id_store.audit_log()
        assert len(log) == 1  # only the initial "registered"

    def test_register_new_path_logs_moved(self, id_store):
        id_store.register("00001", "old/recipe.md")
        id_store.register("00001", "new/recipe.md")
        log = id_store.audit_log()
        assert len(log) == 2
        # Newest first
        assert log[0]["event"] == "moved"
        assert log[0]["old_path"] == "old/recipe.md"
        assert log[0]["rel_path"] == "new/recipe.md"
        assert log[1]["event"] == "registered"

    def test_delete_logs_deleted_event(self, id_store):
        id_store.register("00001", "recipe.md")
        id_store.delete("00001")
        log = id_store.audit_log()
        assert len(log) == 2
        assert log[0]["event"] == "deleted"
        assert log[0]["doc_id"] == "00001"
        assert log[0]["rel_path"] == "recipe.md"

    def test_delete_nonexistent_no_log(self, id_store):
        id_store.delete("XXXXX")
        log = id_store.audit_log()
        assert len(log) == 0

    def test_log_event_public_api(self, id_store):
        id_store.log_event("collision", "abc12", "dup.md", detail="already claimed by orig.md")
        log = id_store.audit_log()
        assert len(log) == 1
        assert log[0]["event"] == "collision"
        assert log[0]["detail"] == "already claimed by orig.md"

    def test_audit_log_filter_by_doc_id(self, id_store):
        id_store.register("00001", "a.md")
        id_store.register("00002", "b.md")
        log = id_store.audit_log(doc_id="00001")
        assert len(log) == 1
        assert log[0]["doc_id"] == "00001"

    def test_audit_log_filter_by_event(self, id_store):
        id_store.register("00001", "a.md")
        id_store.register("00001", "b.md")  # moved
        log = id_store.audit_log(event="moved")
        assert len(log) == 1
        assert log[0]["event"] == "moved"

    def test_audit_log_pagination(self, id_store):
        for i in range(10):
            id_store.register(f"0000{i}", f"file{i}.md")
        # First page
        page1 = id_store.audit_log(limit=3, offset=0)
        assert len(page1) == 3
        # Second page
        page2 = id_store.audit_log(limit=3, offset=3)
        assert len(page2) == 3
        # No overlap
        ids1 = {e["doc_id"] for e in page1}
        ids2 = {e["doc_id"] for e in page2}
        assert ids1.isdisjoint(ids2)

    def test_audit_log_count(self, id_store):
        id_store.register("00001", "a.md")
        id_store.register("00002", "b.md")
        id_store.register("00001", "c.md")  # moved
        assert id_store.audit_log_count() == 3
        assert id_store.audit_log_count(doc_id="00001") == 2
        assert id_store.audit_log_count(event="moved") == 1

    def test_audit_log_newest_first(self, id_store):
        id_store.register("00001", "a.md")
        id_store.register("00002", "b.md")
        id_store.register("00003", "c.md")
        log = id_store.audit_log()
        # Timestamps should be non-increasing (newest first)
        for i in range(1, len(log)):
            assert log[i - 1]["ts"] >= log[i]["ts"]

    def test_audit_log_persists(self, tmp_path):
        db_path = tmp_path / "reg.db"
        s1 = DocIDStore(db_path)
        s1.register("00001", "a.md")
        s1.log_event("collision", "00002", "b.md", detail="test")
        s1.close()

        s2 = DocIDStore(db_path)
        log = s2.audit_log()
        assert len(log) == 2
        s2.close()


class TestAuditLogScenarios:
    """Verify audit log entries are written during real-world scan scenarios."""

    def test_new_file_logged_as_registered(self, tmp_path):
        (tmp_path / "recipe.md").write_text("# Recipe")
        store = DocIDStore(tmp_path / "reg.db")
        _simulate_scan(tmp_path, store)

        log = store.audit_log(event="registered")
        assert len(log) == 1
        assert log[0]["doc_id"] == "00001"
        assert "@00001@" in log[0]["rel_path"]
        store.close()

    def test_file_move_logged(self, tmp_path):
        notes = tmp_path / "notes"
        notes.mkdir()
        (notes / "recipe@abc12@.md").write_text("# Recipe")
        store = DocIDStore(tmp_path / "reg.db")

        _simulate_scan(tmp_path, store)

        archive = tmp_path / "archive"
        archive.mkdir()
        (notes / "recipe@abc12@.md").rename(archive / "recipe@abc12@.md")
        _simulate_scan(tmp_path, store)

        moved = store.audit_log(event="moved")
        assert len(moved) == 1
        assert moved[0]["doc_id"] == "abc12"
        assert "notes/" in moved[0]["old_path"]
        assert "archive/" in moved[0]["rel_path"]
        store.close()

    def test_deletion_logged(self, tmp_path):
        (tmp_path / "recipe@abc12@.md").write_text("# Recipe")
        store = DocIDStore(tmp_path / "reg.db")
        _simulate_scan(tmp_path, store)

        # Delete the file
        (tmp_path / "recipe@abc12@.md").unlink()
        _simulate_scan(tmp_path, store)

        # Simulate flow cleanup
        scanned_ids = set()  # nothing scanned
        for did in list(store.all_mappings().keys()):
            if did not in scanned_ids:
                store.delete(did)

        deleted = store.audit_log(event="deleted")
        assert len(deleted) == 1
        assert deleted[0]["doc_id"] == "abc12"
        assert deleted[0]["rel_path"] == "recipe@abc12@.md"
        store.close()

    def test_collision_logged(self, tmp_path):
        (tmp_path / "a@abc12@.md").write_text("A")
        (tmp_path / "b@abc12@.md").write_text("B")
        store = DocIDStore(tmp_path / "reg.db")
        _simulate_scan(tmp_path, store)

        collisions = store.audit_log(event="collision")
        assert len(collisions) == 1
        assert collisions[0]["doc_id"] == "abc12"
        assert "already claimed by" in collisions[0]["detail"]
        store.close()

    def test_rename_failure_logged(self, tmp_path):
        (tmp_path / "recipe.md").write_text("# Recipe")
        store = DocIDStore(tmp_path / "reg.db")

        original_rename = Path.rename

        def failing_rename(self, target):
            if "recipe" in str(self):
                raise OSError("Permission denied")
            return original_rename(self, target)

        with patch.object(Path, "rename", failing_rename):
            _simulate_scan(tmp_path, store)

        failures = store.audit_log(event="rename_failed")
        assert len(failures) == 1
        assert "Permission denied" in failures[0]["detail"]
        store.close()

    def test_full_lifecycle_audit_trail(self, tmp_path):
        """Walk through a complete document lifecycle and verify the full trail."""
        store = DocIDStore(tmp_path / "reg.db")

        # 1. New file appears
        (tmp_path / "recipe.md").write_text("# Recipe")
        _simulate_scan(tmp_path, store)

        # 2. File is moved (simulate by finding the renamed file and moving it)
        renamed = [f for f in tmp_path.iterdir() if f.suffix == ".md"][0]
        sub = tmp_path / "archive"
        sub.mkdir()
        renamed.rename(sub / renamed.name)
        _simulate_scan(tmp_path, store)

        # 3. File is deleted
        moved_file = [f for f in sub.iterdir()][0]
        moved_file.unlink()
        _simulate_scan(tmp_path, store)
        # Cleanup registry
        for did in list(store.all_mappings().keys()):
            store.delete(did)

        # Verify full trail for doc_id "00001"
        trail = store.audit_log(doc_id="00001")
        events = [e["event"] for e in trail]
        # Newest first: deleted, moved, registered
        assert events == ["deleted", "moved", "registered"]

        # Total events
        assert store.audit_log_count() == 3
        store.close()


# =========================================================================
# Retired IDs — deleted IDs must never be reused
# =========================================================================


class TestRetiredIds:
    """Deleted IDs are retired and cannot be reclaimed by copy-pasted files."""

    def test_delete_retires_id(self, id_store):
        """Deleting a doc_id moves it to retired_ids."""
        id_store.register("00001", "recipe.md")
        id_store.delete("00001")
        assert id_store.is_retired("00001")
        assert id_store.lookup_path("00001") is None  # gone from active registry

    def test_unregistered_id_not_retired(self, id_store):
        assert not id_store.is_retired("XXXXX")

    def test_retired_info(self, id_store):
        id_store.register("00001", "notes/recipe.md")
        id_store.delete("00001")
        info = id_store.retired_info("00001")
        assert info is not None
        assert info["doc_id"] == "00001"
        assert info["last_path"] == "notes/recipe.md"
        assert info["retired_at"] > 0

    def test_retired_info_missing(self, id_store):
        assert id_store.retired_info("XXXXX") is None

    def test_active_id_not_retired(self, id_store):
        """An ID that is still active should not be retired."""
        id_store.register("00001", "recipe.md")
        assert not id_store.is_retired("00001")


class TestScenarioCopyPasteDeletedFile:
    """Scenario: user copies a file whose ID was previously deleted."""

    def test_copy_of_deleted_file_gets_fresh_id(self, tmp_path):
        """If a file carries a retired @XXXXX@ suffix, it must get a new ID."""
        store = DocIDStore(tmp_path / "reg.db")

        # 1. Create and scan a file
        (tmp_path / "recipe.md").write_text("# Recipe")
        records1 = _simulate_scan(tmp_path, store)
        original_id = records1[0]["doc_id"]
        original_path = records1[0]["rel_path"]

        # 2. Delete the file and clean up registry (simulates full flow)
        Path(records1[0]["abs_path"]).unlink()
        store.delete(original_id)
        assert store.is_retired(original_id)

        # 3. Someone copies an old backup that still has the deleted ID in its name
        (tmp_path / f"recipe@{original_id}@.md").write_text("# Old copy")
        records2 = _simulate_scan(tmp_path, store)

        assert len(records2) == 1
        new_id = records2[0]["doc_id"]
        # Must NOT reuse the retired ID
        assert new_id != original_id
        assert len(new_id) == 5
        # File was renamed on disk with the fresh ID
        assert f"@{new_id}@" in records2[0]["rel_path"]
        assert f"@{original_id}@" not in records2[0]["rel_path"]

        # Audit trail shows the collision with retired ID
        collisions = store.audit_log(event="collision")
        assert len(collisions) == 1
        assert "retired ID" in collisions[0]["detail"]
        assert original_path in collisions[0]["detail"] or "previously used by" in collisions[0]["detail"]

        store.close()

    def test_retired_id_persists_across_reopen(self, tmp_path):
        """Retired IDs survive database close/reopen."""
        db_path = tmp_path / "reg.db"
        s1 = DocIDStore(db_path)
        s1.register("abc12", "recipe.md")
        s1.delete("abc12")
        s1.close()

        s2 = DocIDStore(db_path)
        assert s2.is_retired("abc12")
        info = s2.retired_info("abc12")
        assert info["last_path"] == "recipe.md"
        s2.close()

    def test_multiple_deletions_all_retired(self, tmp_path):
        """Multiple deleted IDs are all retired and blocked from reuse."""
        store = DocIDStore(tmp_path / "reg.db")
        store.register("00001", "a.md")
        store.register("00002", "b.md")
        store.register("00003", "c.md")
        store.delete("00001")
        store.delete("00003")

        assert store.is_retired("00001")
        assert not store.is_retired("00002")  # still active
        assert store.is_retired("00003")


class TestSourceNameMigration:
    """Schema migration adds source_name column, backfills existing rows."""

    def test_new_store_has_source_name_column(self, tmp_path):
        from doc_id_store import DocIDStore

        store = DocIDStore(tmp_path / "reg.db")
        store.register("abc", "note.md")

        # Column exists and defaults to 'documents' for rows registered
        # without an explicit source_name (backward compat — FilesystemSource
        # will pass it explicitly in a later task).
        cur = store._conn.execute("SELECT source_name FROM doc_registry WHERE doc_id = 'abc'")
        assert cur.fetchone()[0] == "documents"
        store.close()

    def test_legacy_store_gets_backfilled_on_open(self, tmp_path):
        """A pre-migration DB (no source_name column) opens cleanly and
        existing rows get backfilled to 'documents'."""
        import sqlite3

        db = tmp_path / "legacy.db"
        # Simulate a pre-migration DB: create doc_registry without source_name
        conn = sqlite3.connect(db)
        conn.executescript("""
            CREATE TABLE doc_registry (
                doc_id TEXT PRIMARY KEY,
                rel_path TEXT NOT NULL,
                created REAL NOT NULL
            );
            CREATE TABLE counter (id INTEGER PRIMARY KEY CHECK (id = 1), value INTEGER NOT NULL DEFAULT 0);
            INSERT INTO counter VALUES (1, 0);
            INSERT INTO doc_registry VALUES ('old1', 'old.md', 0.0);
        """)
        conn.commit()
        conn.close()

        # Opening via DocIDStore should run the migration
        from doc_id_store import DocIDStore
        store = DocIDStore(db)

        cur = store._conn.execute("SELECT source_name FROM doc_registry WHERE doc_id = 'old1'")
        assert cur.fetchone()[0] == "documents"
        store.close()

    def test_register_accepts_source_name(self, tmp_path):
        """New register() signature accepts source_name kwarg."""
        from doc_id_store import DocIDStore

        store = DocIDStore(tmp_path / "reg.db")
        store.register("pg1", "quo/abc123", source_name="comm_messages")

        cur = store._conn.execute(
            "SELECT rel_path, source_name FROM doc_registry WHERE doc_id = 'pg1'"
        )
        row = cur.fetchone()
        assert row[0] == "quo/abc123"
        assert row[1] == "comm_messages"
        store.close()

    def test_distinct_source_names(self, tmp_path):
        """Helper for mcp_server to derive _VALID_SOURCE_NAMES at request time."""
        from doc_id_store import DocIDStore

        store = DocIDStore(tmp_path / "reg.db")
        store.register("a", "x.md", source_name="documents")
        store.register("b", "y.md", source_name="documents")
        store.register("c", "quo/123", source_name="comm_messages")

        assert store.distinct_source_names() == {"documents", "comm_messages"}
        store.close()

    def test_register_without_source_name_preserves_existing(self, tmp_path):
        """Calling register() without source_name must not reset the existing
        source_name on the row (prevents accidental clobber from legacy callers)."""
        from doc_id_store import DocIDStore

        store = DocIDStore(tmp_path / "reg.db")
        store.register("x", "path1", source_name="comm_messages")
        store.register("x", "path2")  # no source_name kwarg

        cur = store._conn.execute(
            "SELECT rel_path, source_name FROM doc_registry WHERE doc_id = 'x'"
        )
        row = cur.fetchone()
        assert row[0] == "path2"              # path was updated
        assert row[1] == "comm_messages"      # source_name was preserved, not reset
        store.close()
