import json

import pytest

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

from lancedb_store import LanceDBStore


def _make_node(doc_id: str, loc: str, text: str, vector: list[float]) -> TextNode:
    node = TextNode(
        text=text,
        id_=f"{doc_id}::{loc}",
        embedding=vector,
        metadata={
            "doc_id": doc_id,
            "source_type": "md",
            "loc": loc,
            "snippet": text[:200],
            "mtime": 100.0,
            "size": len(text),
            "title": "Canonical doc",
            "folder": "Projects",
        },
    )
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
    return node


def test_update_canonical_duplicate_metadata_surfaces_dup_fields_in_recent_docs(tmp_path):
    store = LanceDBStore(tmp_path, "test_chunks")
    vector = [0.1] * 768
    canonical_doc_id = "documents::00001"
    store.upsert_nodes([
        _make_node(canonical_doc_id, "c:0", "alpha", vector),
        _make_node(canonical_doc_id, "c:1", "beta", vector),
    ])

    duplicate_refs = [
        {
            "doc_id": "documents::00002",
            "source_name": "documents",
            "rel_path": "Archive/alpha-copy.md",
            "archive_path": "/archive/filesystem/documents%3A%3A00001/alpha-copy.md",
        },
        {
            "doc_id": "comm_messages::msg-42",
            "source_name": "comm_messages",
            "rel_path": "postgres/comm_messages/msg-42",
            "archive_path": "/archive/postgres/comm_messages/documents%3A%3A00001/msg-42.json",
            "natural_key": "msg-42",
        },
    ]

    store.update_canonical_duplicate_metadata(canonical_doc_id, duplicate_refs)

    recent_docs = store.list_recent_docs(limit=10)
    assert len(recent_docs) == 1

    recent_doc = recent_docs[0]
    assert recent_doc["doc_id"] == canonical_doc_id
    assert recent_doc["dup_count"] == "2"
    assert json.loads(recent_doc["dup_sources"]) == ["documents", "comm_messages"]
    assert json.loads(recent_doc["dup_locations"]) == [
        "Archive/alpha-copy.md",
        "postgres/comm_messages/msg-42",
    ]
    assert json.loads(recent_doc["dup_archive_paths"]) == [
        "/archive/filesystem/documents%3A%3A00001/alpha-copy.md",
        "/archive/postgres/comm_messages/documents%3A%3A00001/msg-42.json",
    ]
    assert json.loads(recent_doc["dup_natural_keys"]) == ["msg-42"]

    chunks = store.get_doc_chunks(canonical_doc_id)
    assert len(chunks) == 2
    assert {chunk.dup_count for chunk in chunks} == {"2"}
    assert {chunk.dup_locations for chunk in chunks} == {
        json.dumps(["Archive/alpha-copy.md", "postgres/comm_messages/msg-42"])
    }


def _raw_node_content(store: LanceDBStore, doc_id: str) -> str:
    rows = (
        store._vs.table.search(None)
        .where(f"doc_id = '{doc_id}'", prefilter=True)
        .select(["id", "metadata"])
        .to_list()
    )
    assert rows, f"no rows for {doc_id}"
    metadata = rows[0]["metadata"] or {}
    return metadata.get("_node_content") or ""


def test_update_canonical_duplicate_metadata_does_not_nest_node_content(tmp_path):
    """Regression: repeated duplicate merges must not re-embed the stored
    _node_content inside itself.

    The merge path reads chunk metadata back from LanceDB (which includes the
    LlamaIndex-managed `_node_content` serialization) and rebuilds a TextNode
    from it. If the stale `_node_content` is carried into the new node's
    metadata, the re-serialization nests it inside itself and JSON escaping
    roughly doubles it on every merge — growing to hundreds of MB per row and
    breaking Lance page decoding (Arrow i32 offset overflow) during
    table.optimize()/compaction.
    """
    store = LanceDBStore(tmp_path, "test_chunks")
    vector = [0.1] * 768
    canonical_doc_id = "documents::00001"
    store.upsert_nodes([_make_node(canonical_doc_id, "c:0", "alpha " * 50, vector)])

    baseline = len(_raw_node_content(store, canonical_doc_id))
    sizes = [baseline]
    for i in range(5):
        store.update_canonical_duplicate_metadata(
            canonical_doc_id,
            [
                {
                    "doc_id": f"documents::dup-{i}",
                    "source_name": "documents",
                    "rel_path": f"Archive/copy-{i}.md",
                    "archive_path": f"/archive/filesystem/copy-{i}.md",
                }
            ],
        )
        sizes.append(len(_raw_node_content(store, canonical_doc_id)))

    inner = json.loads(_raw_node_content(store, canonical_doc_id))
    inner_meta = inner.get("metadata") or {}
    assert "_node_content" not in inner_meta, (
        f"_node_content is nested inside itself; bytes per merge: {sizes}"
    )
    # Dup provenance lists grow linearly (a few hundred bytes per merge), so
    # allow slack — but geometric doubling per merge must fail.
    assert sizes[-1] < baseline + 10_000, f"unbounded _node_content growth: {sizes}"


def test_update_canonical_duplicate_metadata_compacts_duplicate_chunk_ids(tmp_path):
    """Legacy physical duplication must self-heal without losing real chunks."""
    store = LanceDBStore(tmp_path, "test_chunks")
    canonical_doc_id = "documents::00001"
    vector = [0.1] * 768
    unique_nodes = [
        _make_node(canonical_doc_id, "c:0", "alpha", vector),
        _make_node(canonical_doc_id, "c:1", "beta", vector),
    ]
    store.upsert_nodes(unique_nodes)
    store._vs.add(
        [_make_node(canonical_doc_id, "c:0", "alpha", vector) for _ in range(128)]
    )

    before = (
        store._vs.table.search(None)
        .where(f"doc_id = '{canonical_doc_id}'", prefilter=True)
        .select(["id"])
        .to_list()
    )
    assert len(before) == 130

    store.update_canonical_duplicate_metadata(
        canonical_doc_id,
        [{"doc_id": "documents::duplicate", "rel_path": "duplicate.jpg"}],
    )

    after = (
        store._vs.table.search(None)
        .where(f"doc_id = '{canonical_doc_id}'", prefilter=True)
        .select(["id"])
        .to_list()
    )
    assert sorted(row["id"] for row in after) == [
        f"{canonical_doc_id}::c:0",
        f"{canonical_doc_id}::c:1",
    ]


def test_upsert_nodes_strips_llama_managed_metadata_keys(tmp_path):
    """Defense in depth: a node whose metadata carries LlamaIndex-managed keys
    from a read-back row (stale _node_content etc.) must not persist them into
    the fresh serialization."""
    store = LanceDBStore(tmp_path, "test_chunks")
    node = _make_node("documents::00001", "c:0", "alpha", [0.1] * 768)
    node.metadata["_node_content"] = '{"text": "stale copy of a whole node"}'
    node.metadata["_node_type"] = "TextNode"
    node.metadata["ref_doc_id"] = "documents::00001"
    store.upsert_nodes([node])

    node_content = _raw_node_content(store, "documents::00001")
    inner_meta = json.loads(node_content).get("metadata") or {}
    assert "_node_content" not in inner_meta
    assert "stale copy" not in node_content


def test_update_canonical_duplicate_metadata_raises_when_canonical_missing(tmp_path):
    store = LanceDBStore(tmp_path, "test_chunks")

    with pytest.raises(LookupError, match="documents::missing"):
        store.update_canonical_duplicate_metadata(
            "documents::missing",
            [
                {
                    "doc_id": "documents::00002",
                    "source_name": "documents",
                    "rel_path": "Archive/missing-copy.md",
                }
            ],
        )
