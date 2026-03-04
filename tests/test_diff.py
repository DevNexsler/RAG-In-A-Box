"""Tests for diff_index_task (mtime-aware diffing)."""

from flow_index_vault import diff_index_task


def test_all_new():
    """First run: everything is new, nothing to delete."""
    scanned = [
        {"doc_id": "a.md", "abs_path": "/a.md", "mtime": 1.0, "size": 10, "ext": "md"},
        {"doc_id": "b.md", "abs_path": "/b.md", "mtime": 2.0, "size": 20, "ext": "md"},
    ]
    to_add, to_delete = diff_index_task.fn(scanned, {})  # empty store
    assert len(to_add) == 2
    assert to_delete == []


def test_nothing_changed():
    """Same files with same mtime — nothing to reprocess."""
    scanned = [{"doc_id": "a.md", "abs_path": "/a.md", "mtime": 1.0, "size": 10, "ext": "md"}]
    stored_mtimes = {"a.md": 1.0}
    to_add, to_delete = diff_index_task.fn(scanned, stored_mtimes)
    assert len(to_add) == 0, "Unchanged file should NOT be reprocessed"
    assert to_delete == []


def test_file_modified():
    """File exists in store but mtime changed — should reprocess."""
    scanned = [{"doc_id": "a.md", "abs_path": "/a.md", "mtime": 5.0, "size": 10, "ext": "md"}]
    stored_mtimes = {"a.md": 1.0}  # old mtime
    to_add, to_delete = diff_index_task.fn(scanned, stored_mtimes)
    assert len(to_add) == 1
    assert to_add[0]["doc_id"] == "a.md"
    assert to_delete == []


def test_file_deleted():
    """File removed from vault but still in store."""
    scanned = [{"doc_id": "a.md", "abs_path": "/a.md", "mtime": 1.0, "size": 10, "ext": "md"}]
    stored_mtimes = {"a.md": 1.0, "deleted.md": 2.0}
    to_add, to_delete = diff_index_task.fn(scanned, stored_mtimes)
    assert set(to_delete) == {"deleted.md"}


def test_file_added():
    """New file in vault, not in store."""
    scanned = [
        {"doc_id": "a.md", "abs_path": "/a.md", "mtime": 1.0, "size": 10, "ext": "md"},
        {"doc_id": "new.md", "abs_path": "/new.md", "mtime": 3.0, "size": 30, "ext": "md"},
    ]
    stored_mtimes = {"a.md": 1.0}
    to_add, to_delete = diff_index_task.fn(scanned, stored_mtimes)
    add_ids = {r["doc_id"] for r in to_add}
    assert add_ids == {"new.md"}, "Only new file should be in to_add"
    assert to_delete == []


def test_empty_vault():
    """Vault is empty; everything in store should be deleted."""
    stored_mtimes = {"old1.md": 1.0, "old2.md": 2.0}
    to_add, to_delete = diff_index_task.fn([], stored_mtimes)
    assert to_add == []
    assert set(to_delete) == {"old1.md", "old2.md"}


def test_mixed_scenario():
    """Mix of new, modified, unchanged, and deleted files."""
    scanned = [
        {"doc_id": "unchanged.md", "abs_path": "/u.md", "mtime": 1.0, "size": 10, "ext": "md"},
        {"doc_id": "modified.md", "abs_path": "/m.md", "mtime": 9.0, "size": 10, "ext": "md"},
        {"doc_id": "new.md", "abs_path": "/n.md", "mtime": 3.0, "size": 30, "ext": "md"},
    ]
    stored_mtimes = {"unchanged.md": 1.0, "modified.md": 2.0, "deleted.md": 4.0}
    to_add, to_delete = diff_index_task.fn(scanned, stored_mtimes)

    add_ids = {r["doc_id"] for r in to_add}
    assert add_ids == {"modified.md", "new.md"}, f"Got {add_ids}"
    assert set(to_delete) == {"deleted.md"}
