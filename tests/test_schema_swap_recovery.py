"""Recovery from a metadata-widening swap that died between its two renames.

`_evolve_metadata_schema` installs the widened table with
`move(<table> -> <table>__schema_backup)` followed by
`move(<table>__schema_tmp -> <table>)`. Dying in between leaves no active table;
before this recovery existed, opening the store then created an empty table and
served 0 rows, and the next widening dropped the backup as stale — turning a
sub-millisecond crash window into permanent data loss.
"""

import errno
import json
import shutil
import subprocess
import sys
from pathlib import Path

import lancedb
import pyarrow as pa
import pytest

import lancedb_store as lancedb_store_module
from lancedb_store import (
    LanceDBStore,
    restore_interrupted_schema_swap,
    schema_swap_lock,
)


TABLE = "test_chunks"
ROWS = 12


def _write_table(index_root: Path, name: str, rows: int) -> None:
    schema = pa.schema(
        [
            pa.field("id", pa.utf8()),
            pa.field("doc_id", pa.utf8()),
            pa.field("text", pa.utf8()),
            pa.field("vector", pa.list_(pa.float32(), 4)),
            pa.field("metadata", pa.struct([pa.field("doc_id", pa.utf8())])),
        ]
    )
    if rows == 0:
        lancedb.connect(str(index_root)).create_table(name, schema=schema)
        return
    ids = [f"documents::{index:03d}" for index in range(rows)]
    batch = pa.RecordBatch.from_arrays(
        [
            pa.array([f"{doc}::c:0" for doc in ids]),
            pa.array(ids),
            pa.array([f"chunk {index}" for index in range(rows)]),
            pa.FixedSizeListArray.from_arrays(
                pa.array([float(index % 7) for index in range(rows * 4)], type=pa.float32()),
                4,
            ),
            pa.StructArray.from_arrays(
                [pa.array(ids)], fields=[pa.field("doc_id", pa.utf8())]
            ),
        ],
        schema=schema,
    )
    lancedb.connect(str(index_root)).create_table(name, batch)


def _mark_swap_in_flight(index_root: Path) -> Path:
    marker = index_root / f"{TABLE}__schema_swap.json"
    marker.write_text(json.dumps({"table": TABLE, "pid": 1, "new_fields": ["x"]}))
    return marker


def _interrupt_between_renames(index_root: Path, *, leave_empty_active: bool) -> None:
    """Reproduce the crash state: widened copy written, first rename done."""
    table_path = index_root / f"{TABLE}.lance"
    temp_path = index_root / f"{TABLE}__schema_tmp.lance"
    backup_path = index_root / f"{TABLE}__schema_backup.lance"
    shutil.copytree(table_path, temp_path)
    _mark_swap_in_flight(index_root)
    shutil.move(str(table_path), str(backup_path))
    if leave_empty_active:
        # One restart later: the process came back and created a fresh table.
        _write_table(index_root, TABLE, 0)


def test_interrupted_swap_is_restored_when_no_active_table(tmp_path):
    _write_table(tmp_path, TABLE, ROWS)
    _interrupt_between_renames(tmp_path, leave_empty_active=False)

    assert restore_interrupted_schema_swap(tmp_path, TABLE) is True

    assert lancedb.connect(str(tmp_path)).open_table(TABLE).count_rows() == ROWS
    assert not (tmp_path / f"{TABLE}__schema_backup.lance").exists()
    assert not (tmp_path / f"{TABLE}__schema_tmp.lance").exists()


def test_opening_the_store_recovers_an_interrupted_swap(tmp_path):
    _write_table(tmp_path, TABLE, ROWS)
    _interrupt_between_renames(tmp_path, leave_empty_active=False)

    store = LanceDBStore(tmp_path, TABLE)

    assert store._vs.table.count_rows() == ROWS
    assert not (tmp_path / f"{TABLE}__schema_backup.lance").exists()


def test_existing_active_table_is_kept_and_leftovers_are_cleaned(tmp_path):
    """An active table that exists is authoritative, empty or not.

    Replacing it from a backup is how a legitimately emptied index gets its rows
    resurrected; the widened table is already committed by the time the active
    path exists, so the only correct action is to drop the leftovers.
    """
    _write_table(tmp_path, TABLE, ROWS)
    _interrupt_between_renames(tmp_path, leave_empty_active=True)
    assert lancedb.connect(str(tmp_path)).open_table(TABLE).count_rows() == 0

    assert restore_interrupted_schema_swap(tmp_path, TABLE) is False

    assert lancedb.connect(str(tmp_path)).open_table(TABLE).count_rows() == 0
    assert not (tmp_path / f"{TABLE}__schema_backup.lance").exists()
    assert not (tmp_path / f"{TABLE}__schema_swap.json").exists()


def test_stale_backup_without_a_marker_never_resurrects_rows(tmp_path):
    """A swap that completed but died before cleanup leaves a stale backup.

    If a later, legitimate deletion empties the table, the stale backup must not
    bring the deleted rows back. The marker is gone once the swap finished, and
    that absence is what makes this safe.
    """
    _write_table(tmp_path, TABLE, ROWS)
    backup_path = tmp_path / f"{TABLE}__schema_backup.lance"
    shutil.copytree(tmp_path / f"{TABLE}.lance", backup_path)
    lancedb.connect(str(tmp_path)).open_table(TABLE).delete("true")
    assert lancedb.connect(str(tmp_path)).open_table(TABLE).count_rows() == 0

    assert restore_interrupted_schema_swap(tmp_path, TABLE) is False
    store = LanceDBStore(tmp_path, TABLE)

    assert store._vs.table.count_rows() == 0
    assert backup_path.exists()


def test_unreadable_active_table_is_never_deleted(tmp_path):
    """A table that fails to READ is a corruption case, not an empty table.

    Deleting it and promoting the backup would throw away rows that
    open_store_with_recovery can repair in place.
    """
    _write_table(tmp_path, TABLE, ROWS)
    _interrupt_between_renames(tmp_path, leave_empty_active=True)
    real_row_count = lancedb_store_module._lance_row_count

    def unreadable(path: Path):
        if path.name == f"{TABLE}.lance":
            return None
        return real_row_count(path)

    lancedb_store_module._lance_row_count = unreadable
    try:
        assert restore_interrupted_schema_swap(tmp_path, TABLE) is False
    finally:
        lancedb_store_module._lance_row_count = real_row_count

    assert (tmp_path / f"{TABLE}.lance").exists()


def test_unreadable_backup_is_left_for_manual_recovery(tmp_path):
    _write_table(tmp_path, TABLE, ROWS)
    _interrupt_between_renames(tmp_path, leave_empty_active=False)
    real_row_count = lancedb_store_module._lance_row_count
    lancedb_store_module._lance_row_count = lambda path: None
    try:
        assert restore_interrupted_schema_swap(tmp_path, TABLE) is False
    finally:
        lancedb_store_module._lance_row_count = real_row_count

    assert (tmp_path / f"{TABLE}__schema_backup.lance").exists()
    assert (tmp_path / f"{TABLE}__schema_swap.json").exists()


def test_swap_lock_fails_closed_on_a_non_contention_error(monkeypatch, tmp_path):
    """Only contention may yield False; anything else must propagate.

    A caller that reads "could not lock" as "lock held" runs the rename pair
    unprotected, which is worse than refusing to run at all.
    """
    def unsupported(fileno, flags):
        raise OSError(errno.ENOTSUP, "operation not supported")

    monkeypatch.setattr("lancedb_store.fcntl.flock", unsupported)
    with pytest.raises(OSError):
        with schema_swap_lock(tmp_path, TABLE):
            pass
    with pytest.raises(OSError):
        with schema_swap_lock(tmp_path, TABLE, blocking=False):
            pass


def test_recovery_does_nothing_without_a_marker(tmp_path):
    _write_table(tmp_path, TABLE, ROWS)
    _interrupt_between_renames(tmp_path, leave_empty_active=False)
    (tmp_path / f"{TABLE}__schema_swap.json").unlink()

    assert restore_interrupted_schema_swap(tmp_path, TABLE) is False
    assert not (tmp_path / f"{TABLE}.lance").exists()


def test_a_swap_in_progress_is_left_alone(tmp_path):
    """A live widening between its two renames looks identical on disk.

    The swap holds the cross-process lock across both renames, so the recovery
    must decline rather than restore the backup out from under it.
    """
    _write_table(tmp_path, TABLE, ROWS)
    _interrupt_between_renames(tmp_path, leave_empty_active=False)

    with schema_swap_lock(tmp_path, TABLE) as held:
        assert held is True
        assert restore_interrupted_schema_swap(tmp_path, TABLE) is False

    assert (tmp_path / f"{TABLE}__schema_backup.lance").exists()
    assert not (tmp_path / f"{TABLE}.lance").exists()

    # Once the holder is gone — kernel releases the lock when a process dies —
    # the same state is recovered.
    assert restore_interrupted_schema_swap(tmp_path, TABLE) is True
    assert lancedb.connect(str(tmp_path)).open_table(TABLE).count_rows() == ROWS


def test_populated_active_table_keeps_its_rows(tmp_path):
    _write_table(tmp_path, TABLE, ROWS)
    backup_path = tmp_path / f"{TABLE}__schema_backup.lance"
    shutil.copytree(tmp_path / f"{TABLE}.lance", backup_path)
    lancedb.connect(str(tmp_path)).open_table(f"{TABLE}__schema_backup").delete("true")
    _mark_swap_in_flight(tmp_path)

    assert restore_interrupted_schema_swap(tmp_path, TABLE) is False
    assert lancedb.connect(str(tmp_path)).open_table(TABLE).count_rows() == ROWS


_DIE_BETWEEN_RENAMES = """
import os, sys
sys.path.insert(0, {repo!r})
import lancedb_store as module

real_move = module.shutil.move
calls = {{"count": 0}}


def move(src, dst):
    calls["count"] += 1
    if calls["count"] == 2:
        os._exit(9)
    return real_move(src, dst)


module.shutil.move = move
store = module.LanceDBStore({root!r}, {table!r})
store._evolve_metadata_schema({{"content_status"}})
"""


def test_fault_injected_death_between_renames_is_recovered(tmp_path):
    """Kill a real widening in its rename window, then reopen the store.

    Everything above builds the crash state by hand; this drives the shipped
    `_evolve_metadata_schema` in a separate interpreter and takes it out with
    `os._exit`, so no exception handler runs — which is what a container kill
    actually looks like.
    """
    _write_table(tmp_path, TABLE, ROWS)
    script = _DIE_BETWEEN_RENAMES.format(
        repo=str(Path(lancedb_store_module.__file__).parent),
        root=str(tmp_path),
        table=TABLE,
    )

    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180
    )

    assert completed.returncode == 9, (
        f"child did not die in the rename window: rc={completed.returncode} "
        f"stderr={completed.stderr[-2000:]}"
    )
    assert not (tmp_path / f"{TABLE}.lance").exists()
    assert (tmp_path / f"{TABLE}__schema_swap.json").exists()

    store = LanceDBStore(tmp_path, TABLE)

    assert store._vs.table.count_rows() == ROWS
    assert "content_status" not in {
        field.name for field in store._vs.table.schema.field("metadata").type
    }
    assert not (tmp_path / f"{TABLE}__schema_backup.lance").exists()
    assert not (tmp_path / f"{TABLE}__schema_tmp.lance").exists()
    assert not (tmp_path / f"{TABLE}__schema_swap.json").exists()
