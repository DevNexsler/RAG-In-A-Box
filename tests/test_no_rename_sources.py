"""Tests for deposit-owned no-rename paths.

External depositors (email-attachment capture, quo-attachment sync) own the
filenames in their directories and re-deposit any attachment they no longer
see. Renaming those files to inject @XXXXX@ doc IDs created an endless
duplicate loop: deposit -> rename -> depositor sees file "missing" ->
re-deposit -> new ID -> ... Paths under scan.no_rename keep their filenames;
identity lives in the registry keyed by relative path.
"""

from pathlib import Path

from doc_id_store import DocIDStore
from flow_index_vault import scan_filesystem_records


def _setup(tmp_path):
    (tmp_path / "email-attachments" / "dan").mkdir(parents=True)
    (tmp_path / "notes").mkdir()
    (tmp_path / "email-attachments" / "dan" / "msg1__mm0.pdf").write_text("attachment body")
    (tmp_path / "notes" / "todo.md").write_text("# todo")
    registry = DocIDStore(tmp_path / "registry.db")
    return registry


def test_no_rename_path_keeps_filename_and_registers_id(tmp_path):
    registry = _setup(tmp_path)
    records = scan_filesystem_records(
        tmp_path, ["**/*"], ["registry.db*"],
        doc_id_store=registry,
        no_rename_prefixes=["email-attachments/"],
    )
    # File on disk untouched
    assert (tmp_path / "email-attachments" / "dan" / "msg1__mm0.pdf").exists()
    att = next(r for r in records if "msg1" in r["rel_path"])
    assert "@" not in Path(att["rel_path"]).name
    # but it has a registry-tracked doc_id
    assert registry.lookup_path(att["doc_id"]) == "email-attachments/dan/msg1__mm0.pdf"
    registry.close()


def test_no_rename_path_id_is_stable_across_scans(tmp_path):
    registry = _setup(tmp_path)
    kwargs = dict(doc_id_store=registry, no_rename_prefixes=["email-attachments/"])
    first = scan_filesystem_records(tmp_path, ["**/*"], ["registry.db*"], **kwargs)
    second = scan_filesystem_records(tmp_path, ["**/*"], ["registry.db*"], **kwargs)
    id1 = next(r["doc_id"] for r in first if "msg1" in r["rel_path"])
    id2 = next(r["doc_id"] for r in second if "msg1" in r["rel_path"])
    assert id1 == id2


def test_paths_outside_no_rename_still_get_renamed(tmp_path):
    registry = _setup(tmp_path)
    scan_filesystem_records(
        tmp_path, ["**/*"], ["registry.db*"],
        doc_id_store=registry,
        no_rename_prefixes=["email-attachments/"],
    )
    notes = list((tmp_path / "notes").iterdir())
    assert len(notes) == 1
    assert "@" in notes[0].name  # normal path: ID injected via rename
    registry.close()


def test_no_prefixes_means_legacy_behavior_everywhere(tmp_path):
    registry = _setup(tmp_path)
    scan_filesystem_records(
        tmp_path, ["**/*"], ["registry.db*"], doc_id_store=registry,
    )
    att_files = list((tmp_path / "email-attachments" / "dan").iterdir())
    assert all("@" in f.name for f in att_files)
    registry.close()


def test_deposit_file_already_carrying_id_keeps_it(tmp_path):
    registry = _setup(tmp_path)
    # Simulate a file renamed by an older run: ID in filename, registered
    pre_id = registry.next_id()
    old = tmp_path / "email-attachments" / "dan" / f"msg2__mm1@{pre_id}@.pdf"
    old.write_text("previously renamed")
    registry.register(pre_id, f"email-attachments/dan/msg2__mm1@{pre_id}@.pdf")
    records = scan_filesystem_records(
        tmp_path, ["**/*"], ["registry.db*"],
        doc_id_store=registry,
        no_rename_prefixes=["email-attachments/"],
    )
    rec = next(r for r in records if "msg2" in r["rel_path"])
    assert rec["doc_id"] == pre_id
    assert old.exists()  # untouched
    registry.close()
