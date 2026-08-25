"""Short-lived Lance maintenance subprocess contract."""

from unittest.mock import MagicMock, patch


def test_compact_dataset_uses_streaming_binary_copy():
    from core.lance_maintenance import compact_dataset

    dataset = MagicMock()
    with patch("lance.dataset", return_value=dataset) as open_dataset:
        compact_dataset("/data/index/chunks.lance")

    open_dataset.assert_called_once_with("/data/index/chunks.lance")
    dataset.optimize.compact_files.assert_called_once_with(
        compaction_mode="try_binary_copy"
    )


def _index_dirs(dataset_path):
    from pathlib import Path

    return {entry.name for entry in (Path(dataset_path) / "_indices").iterdir()}


def _reachable_index_uuids(dataset_path):
    from core.lance_maintenance import reachable_index_uuids

    return reachable_index_uuids(dataset_path)


def _dataset_with_orphan_indices(tmp_path):
    """Build a dataset whose `_indices` holds superseded index generations."""
    from datetime import timedelta

    import lance
    import pyarrow as pa

    path = str(tmp_path / "orphans.lance")
    rows = pa.table({
        "id": pa.array(range(64)),
        "text": pa.array([f"row {i} alpha" for i in range(64)]),
    })
    dataset = lance.write_dataset(rows, path)
    dataset.create_scalar_index("text", index_type="INVERTED")
    for batch in range(3):
        dataset = lance.write_dataset(
            pa.table({
                "id": pa.array(range(64 * (batch + 1), 64 * (batch + 2))),
                "text": pa.array([f"row {i} beta" for i in range(64)]),
            }),
            path,
            mode="append",
        )
        dataset.optimize.optimize_indices()
    # Drop every superseded version so the old index generations are provably
    # unreachable; Lance still leaves their directories behind.
    lance.dataset(path).cleanup_old_versions(
        older_than=timedelta(0), error_if_tagged_old_versions=False
    )
    return path


def test_prune_orphan_indices_removes_unreferenced_index_directories(tmp_path):
    """Index directories no retained version references are reclaimed."""
    from core.lance_maintenance import prune_orphan_indices

    path = _dataset_with_orphan_indices(tmp_path)
    reachable = _reachable_index_uuids(path)
    orphans = _index_dirs(path) - reachable
    assert orphans, "fixture must leave superseded index directories behind"

    removed, reclaimed = prune_orphan_indices(path)

    assert removed == len(orphans)
    assert reclaimed >= 0
    assert _index_dirs(path) == reachable


def test_prune_orphan_indices_keeps_directories_inside_the_grace_window(tmp_path):
    """A grace window protects an index build that has not committed yet."""
    from core.lance_maintenance import prune_orphan_indices

    path = _dataset_with_orphan_indices(tmp_path)
    orphans = _index_dirs(path) - _reachable_index_uuids(path)

    removed, reclaimed = prune_orphan_indices(path, min_age_seconds=3600)

    assert (removed, reclaimed) == (0, 0)
    assert orphans <= _index_dirs(path)


def test_prune_orphan_indices_reports_reclaimed_bytes(tmp_path):
    """Reclaimed bytes cover the files inside the removed directories."""
    from pathlib import Path

    from core.lance_maintenance import prune_orphan_indices

    path = _dataset_with_orphan_indices(tmp_path)
    orphan = sorted(_index_dirs(path) - _reachable_index_uuids(path))[0]
    (Path(path) / "_indices" / orphan / "part_0_invert.lance").write_bytes(b"x" * 4096)

    removed, reclaimed = prune_orphan_indices(path)

    assert removed >= 1
    assert reclaimed >= 4096


def test_prune_orphan_indices_keeps_every_segment_of_a_delta_index(tmp_path):
    """A live index split across delta segments keeps all of its directories.

    Each delta is its own `_indices/<uuid>` directory under one index name, so
    a sweep that enumerated indices instead of segments would delete the parts
    of a live index it never listed."""
    import lance
    import pyarrow as pa

    from core.lance_maintenance import prune_orphan_indices

    def _rows(count, offset=0):
        flat = pa.array(
            [float((i % 17) + 1) for i in range(count * 16)], type=pa.float32()
        )
        return pa.table({
            "id": pa.array(range(offset, offset + count)),
            "v": pa.FixedSizeListArray.from_arrays(flat, 16),
        })

    path = str(tmp_path / "deltas.lance")
    dataset = lance.write_dataset(_rows(512), path)
    dataset.create_index("v", index_type="IVF_PQ", num_partitions=2, num_sub_vectors=4)
    for batch in range(3):
        dataset = lance.write_dataset(
            _rows(256, 512 + 256 * batch), path, mode="append"
        )
        dataset.optimize.optimize_indices(num_indices_to_merge=0)

    segments = _reachable_index_uuids(path)
    assert len(segments) > 1, "fixture must produce a multi-segment index"

    removed, _ = prune_orphan_indices(path)

    assert removed == 0
    assert segments <= _index_dirs(path)
