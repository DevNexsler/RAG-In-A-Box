"""Regression tests for #3143: mid-sweep queue service vs sweep insert race."""

import tempfile
from unittest.mock import MagicMock, patch

import lance
import pytest

import flow_index_vault as fiv
from core.index_request_queue import IndexRequestQueue
from lancedb_store import LanceDBStore
from tests.test_store import _lance_path, _make_node


@pytest.fixture
def config(tmp_path):
    return {
        "index_root": str(tmp_path),
        "lancedb": {"table": "chunks"},
        "index_queue": {},
    }


def test_service_index_queue_reconciles_sweep_insert_snapshot(tmp_path, config):
    """Serving a queued request drops the doc from the sweep's insert set."""
    fiv._RUNTIME.clear()
    fiv._RUNTIME.update(
        {
            "store": object(),
            "doc_id_store": object(),
            "index_root": tmp_path,
            "storage_insert_doc_ids": {"documents::queued"},
            "config": config,
        }
    )
    queue = IndexRequestQueue(tmp_path)
    queue.enqueue("chunks", "documents", "queued.bin")

    with patch.object(
        fiv,
        "_index_document_unlocked",
        return_value={
            "status": "indexed",
            "doc_id": "documents::queued",
            "rel_path": "queued.bin",
        },
    ):
        assert fiv._service_index_queue(config, "chunks") == 1

    assert "documents::queued" not in fiv._RUNTIME["storage_insert_doc_ids"]
    assert "documents::queued" in fiv._RUNTIME["sweep_served_doc_ids"]
    fiv._RUNTIME.clear()


def test_process_doc_task_skips_sweep_served_doc():
    """A sweep worker must not touch the store for a checkpoint-served doc."""
    store = MagicMock()
    fiv._RUNTIME.clear()
    fiv._RUNTIME.update(
        {
            "store": store,
            "config": {},
            "sweep_served_doc_ids": {"documents::queued"},
        }
    )
    try:
        fiv._process_doc_task(
            {
                "doc_id": "documents::queued",
                "rel_path": "queued.bin",
                "mtime": 1.0,
                "size": 10,
                "ext": "bin",
                "source_type": "bin",
                "source_name": "documents",
            }
        )
    finally:
        fiv._RUNTIME.clear()

    store.insert_nodes.assert_not_called()
    store.upsert_nodes.assert_not_called()


def test_queued_request_during_sweep_leaves_one_chunk_row():
    """Reproduce #3143: queue service upserts, sweep insert must not duplicate."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        doc_id = "documents::queued"
        with store.exclusive_writer_session():
            store.upsert_nodes([
                _make_node(doc_id, "c:0", "request lane", [0.2] * 768)
            ])
            wrote = store.insert_nodes(
                [_make_node(doc_id, "c:0", "sweep lane", [0.3] * 768)],
                known_absent=True,
            )

        dataset = lance.dataset(_lance_path(tmpdir))
        assert wrote is False
        assert dataset.count_rows(f"doc_id = '{doc_id}'") == 1
        chunk_ids = [row["id"] for row in dataset.to_table(columns=["id"]).to_pylist()]
        assert len(chunk_ids) == len(set(chunk_ids))
