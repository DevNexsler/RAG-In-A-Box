"""Every LanceDB connection in the process shares one bounded session (#1157)."""

from __future__ import annotations

import random

import lancedb
import numpy as np
import pyarrow as pa
import pytest

from core import lance_session
from lancedb_store import LanceDBStore
from taxonomy_store import TaxonomyStore


ROWS = 2_000
DIM = 32


@pytest.fixture(autouse=True)
def restore_process_session():
    """Leave the process-wide session at its shipped defaults for other tests."""
    yield
    lance_session.configure()


@pytest.fixture
def session():
    return lance_session.configure(index_cache_mb=4, metadata_cache_mb=2)


def _build_table(index_root, table_name: str = "chunks") -> None:
    """Write a small production-shaped chunks table straight through LanceDB."""
    rng = np.random.default_rng(1157)
    vectors = rng.standard_normal((ROWS, DIM), dtype=np.float32)
    words = ["alpha", "beta", "gamma", "delta", "epsilon"]
    table = pa.table(
        {
            "id": pa.array([f"doc-{i}::0" for i in range(ROWS)]),
            "doc_id": pa.array([f"doc-{i}" for i in range(ROWS)]),
            "text": pa.array(
                [" ".join(random.Random(i).choices(words, k=20)) for i in range(ROWS)]
            ),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(vectors.reshape(-1)), DIM
            ),
            "metadata": pa.array(
                [{"doc_id": f"doc-{i}", "source_type": "md"} for i in range(ROWS)],
                type=pa.struct(
                    [pa.field("doc_id", pa.utf8()), pa.field("source_type", pa.utf8())]
                ),
            ),
        }
    )
    connection = lancedb.connect(str(index_root))
    created = connection.create_table(table_name, table)
    created.create_fts_index("text", replace=True)


def _search(store: LanceDBStore, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(5):
        store.vector_search(list(rng.standard_normal(DIM).astype(float)), top_k=5)
        store.keyword_search("alpha", top_k=5)


def test_configure_sizes_the_process_wide_session():
    configured = lance_session.configure(index_cache_mb=4, metadata_cache_mb=2)
    assert lance_session.get_session() is configured


def test_connect_shares_one_session_across_connections(session, tmp_path):
    _build_table(tmp_path)
    idle_bytes = session.size_bytes

    for _ in range(3):
        lance_session.connect(tmp_path).open_table("chunks").count_rows()

    assert session.size_bytes > idle_bytes


def test_store_reads_land_in_the_shared_session(session, tmp_path):
    _build_table(tmp_path)

    _search(LanceDBStore(tmp_path, "chunks"))

    assert session.approx_num_items > 0


def test_reconnect_reuses_the_shared_session(session, tmp_path):
    """A reconnect must not start a second default-sized Lance cache."""
    _build_table(tmp_path)

    store = LanceDBStore(tmp_path, "chunks")
    for seed in range(5):
        store._reconnect()
        _search(store, seed=seed)

    assert lance_session.get_session() is session
    assert session.approx_num_items > 0


def test_taxonomy_store_shares_the_session(session, tmp_path):
    idle_bytes = session.size_bytes

    store = TaxonomyStore(tmp_path, "taxonomy", embed_fn=lambda text: [0.0] * 8)
    store.add("building", "Maple Court", "a building")

    assert store.count() == 1
    assert session.size_bytes > idle_bytes


def test_importing_lance_session_opts_out_of_centroid_statistics(monkeypatch):
    """index_stats() otherwise prints a raw WARN to stderr on first use — an
    untimestamped line in indexer.log (#0546) from every sweep's health read."""
    import importlib
    import os

    monkeypatch.delenv("LANCE_INCLUDE_VECTOR_CENTROIDS", raising=False)
    importlib.reload(lance_session)
    assert os.environ["LANCE_INCLUDE_VECTOR_CENTROIDS"] == "false"

    monkeypatch.setenv("LANCE_INCLUDE_VECTOR_CENTROIDS", "true")
    importlib.reload(lance_session)  # an explicit operator setting wins
    assert os.environ["LANCE_INCLUDE_VECTOR_CENTROIDS"] == "true"
