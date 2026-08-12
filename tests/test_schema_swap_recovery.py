"""Recovery from a metadata-widening swap that died between its two renames.

`_evolve_metadata_schema` installs the widened table with
`move(<table> -> <table>__schema_backup)` followed by
`move(<table>__schema_tmp -> <table>)`. Dying in between leaves no active table;
before this recovery existed, opening the store then created an empty table and
served 0 rows, and the next widening dropped the backup as stale — turning a
sub-millisecond crash window into permanent data loss.
"""

import shutil
from pathlib import Path

import lancedb
import pyarrow as pa
import pytest

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


def _interrupt_between_renames(index_root: Path, *, leave_empty_active: bool) -> None:
    """Reproduce the crash state: widened copy written, first rename done."""
    table_path = index_root / f"{TABLE}.lance"
    temp_path = index_root / f"{TABLE}__schema_tmp.lance"
    backup_path = index_root / f"{TABLE}__schema_backup.lance"
    shutil.copytree(table_path, temp_path)
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


def test_empty_active_table_beside_populated_backup_is_restored(tmp_path):
    _write_table(tmp_path, TABLE, ROWS)
    _interrupt_between_renames(tmp_path, leave_empty_active=True)
    assert lancedb.connect(str(tmp_path)).open_table(TABLE).count_rows() == 0

    store = LanceDBStore(tmp_path, TABLE)

    assert store._vs.table.count_rows() == ROWS
    assert not (tmp_path / f"{TABLE}__schema_backup.lance").exists()


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


def test_populated_active_table_is_never_replaced_by_a_backup(tmp_path):
    _write_table(tmp_path, TABLE, ROWS)
    backup_path = tmp_path / f"{TABLE}__schema_backup.lance"
    shutil.copytree(tmp_path / f"{TABLE}.lance", backup_path)

    assert restore_interrupted_schema_swap(tmp_path, TABLE) is False
    assert (tmp_path / f"{TABLE}.lance").exists()
    assert backup_path.exists()


@pytest.mark.parametrize("rows", [0])
def test_empty_backup_is_not_promoted(tmp_path, rows):
    _write_table(tmp_path, TABLE, ROWS)
    table_path = tmp_path / f"{TABLE}.lance"
    backup_path = tmp_path / f"{TABLE}__schema_backup.lance"
    shutil.move(str(table_path), str(backup_path))
    # Truncate the backup to zero rows: nothing worth restoring.
    lancedb.connect(str(tmp_path)).open_table(f"{TABLE}__schema_backup").delete("true")

    assert restore_interrupted_schema_swap(tmp_path, TABLE) is False
