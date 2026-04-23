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
