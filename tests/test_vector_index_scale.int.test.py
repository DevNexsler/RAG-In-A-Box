"""Vector index on a table big enough to matter: exact results, one index, merged tail.

The unit tests prove the plan and the stats on six rows. This proves, on 4,000
rows x 512 dims (the same shape as production at 1/30 the size), that the ANN
path returns what the brute-force scan returns, that the sweep's merge keeps it
one index, and that rows added after the build are found through it.
"""

import tempfile

import numpy as np
import pytest
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

import lancedb_store as lancedb_store_module
from lancedb_store import LanceDBStore

DIM = 512
DOCS = 20
CHUNKS_PER_DOC = 200
TOP_K = 10
QUERIES = 25


def _node(doc_id: str, loc: str, vector: list[float]) -> TextNode:
    node = TextNode(
        text=f"{doc_id} {loc}",
        id_=f"{doc_id}::{loc}",
        embedding=vector,
        metadata={"doc_id": doc_id, "source_type": "md", "loc": loc, "mtime": 1.0, "size": 8},
    )
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
    return node


def _flat_top_k(matrix: np.ndarray, ids: list[str], query: np.ndarray, k: int) -> set[str]:
    distances = ((matrix - query) ** 2).sum(axis=1)
    return {ids[i] for i in np.argsort(distances, kind="stable")[:k]}


def _table_rows(store: LanceDBStore) -> tuple[np.ndarray, list[str]]:
    table = store._vs.table.to_lance().to_table(columns=["id", "vector"])
    ids = table["id"].to_pylist()
    matrix = np.stack([np.asarray(v, dtype=np.float32) for v in table["vector"].to_pylist()])
    return matrix, ids


def _ann_top_k(store: LanceDBStore, query: np.ndarray, k: int) -> set[str]:
    return {f"{hit.doc_id}::{hit.loc}" for hit in store.vector_search(query.tolist(), top_k=k)}


def _recall(store: LanceDBStore, rng: np.random.Generator) -> float:
    matrix, ids = _table_rows(store)
    picks = rng.choice(len(ids), size=QUERIES, replace=False)
    hits = 0
    for row in picks:
        query = matrix[row]
        hits += len(_flat_top_k(matrix, ids, query, TOP_K) & _ann_top_k(store, query, TOP_K))
    return hits / (QUERIES * TOP_K)


@pytest.fixture(scope="module")
def scaled_store():
    rng = np.random.default_rng(20260906)
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "chunks")
        for d in range(DOCS):
            vectors = rng.standard_normal((CHUNKS_PER_DOC, DIM)).astype(np.float32)
            store.upsert_nodes(
                [_node(f"doc{d}.md", f"c:{c}", vectors[c].tolist()) for c in range(CHUNKS_PER_DOC)]
            )
        store.create_fts_index()
        yield store, rng


def test_ann_index_is_exact_at_this_scale_and_uses_the_index(scaled_store):
    store, rng = scaled_store
    assert store.count_chunks() == DOCS * CHUNKS_PER_DOC
    assert store.vector_index_available() is False
    assert "KNNVectorDistance" in store.explain_vector_search([0.0] * DIM)

    assert store.ensure_vector_index() is True
    stats = store.vector_index_stats()
    assert stats["indexed_rows"] == DOCS * CHUNKS_PER_DOC
    assert stats["unindexed_rows"] == 0 and stats["num_indices"] == 1
    # partitions = min(sqrt(4000), 4000 // 256) = 15, all probed at nprobes 40:
    # exact within float32 ordering, so the ANN answer is the flat answer.
    assert lancedb_store_module.ivf_partitions_for(DOCS * CHUNKS_PER_DOC) == 15
    plan = store.explain_vector_search([0.0] * DIM)
    assert "ANNSubIndex" in plan and "KNNVectorDistance" not in plan, plan
    assert _recall(store, rng) >= 0.98


def test_tail_added_after_the_build_is_merged_and_found(scaled_store):
    store, rng = scaled_store
    assert store.vector_index_available() is True
    tail = rng.standard_normal((50, DIM)).astype(np.float32)
    store.upsert_nodes([_node("tail.md", f"c:{c}", tail[c].tolist()) for c in range(50)])
    assert store.vector_index_stats()["unindexed_rows"] == 50

    store.ensure_fts_index()  # the sweep's incremental step

    stats = store.vector_index_stats()
    assert stats["unindexed_rows"] == 0
    assert stats["indexed_rows"] == DOCS * CHUNKS_PER_DOC + 50
    assert stats["num_indices"] == 1
    assert _ann_top_k(store, tail[7], 1) == {"tail.md::c:7"}
    assert _recall(store, rng) >= 0.98
