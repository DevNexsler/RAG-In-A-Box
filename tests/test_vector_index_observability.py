"""Read-only health telemetry and configured/live ANN drift (#2226)."""

import logging
from unittest.mock import patch

import pytest
from llama_index.core.schema import TextNode

import mcp_server
from lancedb_store import LanceDBStore


@pytest.fixture
def indexed_store(tmp_path):
    store = LanceDBStore(tmp_path, "custom_chunks")
    store.upsert_nodes([
        TextNode(text=f"doc {i}", id_=str(i), embedding=[float(i == j) for j in range(8)])
        for i in range(8)
    ])
    store.ensure_vector_index(index_type="IVF_FLAT")
    return store


@pytest.mark.parametrize("stalled", [False, True])
def test_health_reports_live_index_without_store_init_or_writer_lock(
    tmp_path, indexed_store, stalled,
):
    expected = indexed_store.vector_index_stats()
    config = {"index_root": str(tmp_path), "lancedb": {"table": "custom_chunks"}}
    with (
        patch("mcp_server._index_disk_usage", return_value={}),
        patch("mcp_server._get_index_run_supervisor") as supervisor,
        patch("mcp_server._resolve_indexer_pid", return_value=(stalled, 123 if stalled else None)),
        patch.object(LanceDBStore, "__init__", side_effect=AssertionError("store init writes")),
        patch("fcntl.flock", side_effect=AssertionError("writer lock")),
    ):
        supervisor.return_value.status_summary.return_value = {"unresolved_failure": False}
        payload, code = mcp_server._health_probe(config)
    assert code == (503 if stalled else 200)
    assert payload["vector_index"] == expected
    assert payload["vector_index"]["index_type"] == "IVF_FLAT"


def test_health_missing_table_does_not_create_index_directory(tmp_path):
    root = tmp_path / "absent"
    payload, _ = mcp_server._health_probe({"index_root": str(root)})
    assert payload["vector_index"]["available"] is False
    assert payload["vector_index"]["index_type"] is None
    assert not (root / "chunks.lance").exists()


def test_health_metadata_failure_is_unknown_without_leaking_error(tmp_path, indexed_store):
    with patch("core.lance_session.connect", side_effect=RuntimeError("private path or secret")):
        payload, _ = mcp_server._health_probe({
            "index_root": str(tmp_path), "lancedb": {"table": "custom_chunks"},
        })
    assert payload["vector_index"]["available"] is None
    assert payload["vector_index"]["error"] == "metadata_unavailable"
    assert "private path" not in str(payload)


@pytest.mark.parametrize("configured, warns", [("IVF_PQ", True), ("ivf_flat", False)])
def test_existing_index_type_drift_is_logged_without_rebuild(indexed_store, caplog, configured, warns):
    with caplog.at_level(logging.WARNING, logger="lancedb_store"), patch.object(
        indexed_store._vs.table, "create_index", side_effect=AssertionError("must not rebuild"),
    ):
        assert indexed_store.ensure_vector_index(index_type=configured) is False
    messages = [r.getMessage() for r in caplog.records if "Vector index type mismatch" in r.getMessage()]
    assert len(messages) == int(warns)
    if warns:
        assert "configured=IVF_PQ" in messages[0]
        assert "live=IVF_FLAT" in messages[0]
        assert "--rebuild" in messages[0]
