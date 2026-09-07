"""The shared Lance session actually caps cache growth on a real index (#1157).

Production's server process grew to ~6 GiB RSS and the memcg OOM-killer took
the whole container down.  The growth is LanceDB's per-connection session
cache, which is sized for a dedicated host when nobody passes a session.  This
test is self-calibrating: it first proves the workload's cache footprint
exceeds the cap, then proves the shared bounded session holds it under.
"""

from __future__ import annotations

import random

import lancedb
import numpy as np
import pyarrow as pa
import pytest

from core import lance_session
from lancedb_store import LanceDBStore


ROWS = 60_000
DIM = 256
INDEX_CACHE_MB = 4
METADATA_CACHE_MB = 2
CAP_BYTES = (INDEX_CACHE_MB + METADATA_CACHE_MB) * 1024 * 1024
# The cache is evicted lazily, so a bounded session overshoots its ceiling by
# roughly one batch of index partitions (measured: 5-14 MB readings against a
# 6 MB cap) before eviction pulls it back. A single reading against the cap is
# therefore a coin flip; the bound shows as the cache RETURNING under the cap
# and never reaching the unbounded footprint. Anything near the cap is
# bounded, 6 GiB is not.
CAP_TOLERANCE = 1.5


@pytest.fixture(autouse=True)
def restore_process_session():
    yield
    lance_session.configure()


def _build_indexed_table(index_root) -> None:
    rng = np.random.default_rng(1157)
    vectors = rng.standard_normal((ROWS, DIM), dtype=np.float32)
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    table = pa.table(
        {
            "id": pa.array([f"doc-{i}::0" for i in range(ROWS)]),
            "doc_id": pa.array([f"doc-{i}" for i in range(ROWS)]),
            "text": pa.array(
                [" ".join(random.Random(i).choices(words, k=40)) for i in range(ROWS)]
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
    created = connection.create_table("chunks", table)
    created.create_index(num_partitions=64, num_sub_vectors=32, index_type="IVF_PQ")
    created.create_fts_index("text", replace=True)


def _drive_reads(search, queries: int = 60) -> None:
    rng = np.random.default_rng(3)
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    for _ in range(queries):
        search(list(rng.standard_normal(DIM).astype(float)), str(rng.choice(words)))


def test_shared_session_caps_lance_cache_growth(tmp_path):
    _build_indexed_table(tmp_path)

    # Calibration: without an explicit session this workload caches past the cap.
    unbounded = lancedb.Session.default()
    unbounded_table = lancedb.connect(str(tmp_path), session=unbounded).open_table(
        "chunks"
    )
    _drive_reads(
        lambda vector, word: (
            unbounded_table.search(vector, query_type="vector").limit(10).to_list(),
            unbounded_table.search(word, query_type="fts").limit(10).to_list(),
        )
    )
    assert unbounded.size_bytes > CAP_BYTES, (
        "workload too small to prove a bound; grow ROWS/DIM"
    )

    session = lance_session.configure(
        index_cache_mb=INDEX_CACHE_MB, metadata_cache_mb=METADATA_CACHE_MB
    )
    store = LanceDBStore(tmp_path, "chunks")
    # Reconnects are the multi-day-uptime path: schema swaps, shadow promotions
    # and stale-read recovery each used to open a fresh default-sized cache.
    readings: list[int] = []
    for _ in range(4):
        _drive_reads(
            lambda vector, word: (
                store.vector_search(vector, top_k=10),
                store.keyword_search(word, top_k=10),
            ),
            queries=20,
        )
        readings.append(session.size_bytes)
        store._reconnect()
    assert session.approx_num_items > 0
    # Bounded: the cache comes back under the cap between rounds. (Its transient
    # overshoot can reach most of the unbounded footprint at this table size —
    # the whole index is ~19 MB — so the peak carries no signal; the return does.)
    assert min(readings) <= CAP_BYTES * CAP_TOLERANCE, (readings, unbounded.size_bytes)
