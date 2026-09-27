"""A committed write followed by process death must replay without duplicate chunks."""

import subprocess
import sys

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

from core.index_request_queue import IndexRequestQueue
from lancedb_store import LanceDBStore


def test_crash_between_store_commit_and_queue_ack_replays_idempotently(tmp_path):
    queue = IndexRequestQueue(tmp_path)
    original = queue.enqueue("chunks", "documents", "invoice.md")
    process = subprocess.run([sys.executable, "-c", """
import os, sys
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from lancedb_store import LanceDBStore
nodes=[]
for loc,text in [("c:0","Invoice total is 42 dollars."),("c:1","Payment due Friday.")]:
    node=TextNode(id_="documents::one::"+loc,text=text,embedding=[0.1]*768,
                  metadata={"doc_id":"documents::one","loc":loc,"mtime":1.0})
    node.relationships[NodeRelationship.SOURCE]=RelatedNodeInfo(node_id="documents::one")
    nodes.append(node)
store=LanceDBStore(sys.argv[1],"chunks")
store.upsert_nodes(nodes)
assert len(store.get_doc_chunks("documents::one")) == 2
os._exit(17)
""", str(tmp_path)], timeout=60, capture_output=True, text=True)
    assert process.returncode == 17, process.stderr
    assert IndexRequestQueue(tmp_path).pending("chunks", limit=10) == [original]

    # A newer forced request arrives while the old worker's snapshot survives.
    current = queue.enqueue("chunks", "documents", "invoice.md", force=True)
    assert queue.complete(original) is False
    reopened = LanceDBStore(tmp_path, "chunks")
    nodes = []
    for loc, text in [("c:0", "Invoice total is 42 dollars."), ("c:1", "Payment due Friday.")]:
        node = TextNode(id_=f"documents::one::{loc}", text=text, embedding=[0.1] * 768,
                        metadata={"doc_id": "documents::one", "loc": loc, "mtime": 1.0})
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id="documents::one")
        nodes.append(node)
    reopened.upsert_nodes(nodes)
    assert queue.complete(current) is True

    final = LanceDBStore(tmp_path, "chunks")
    assert final.list_doc_ids() == ["documents::one"]
    assert [(chunk.loc, chunk.text) for chunk in final.get_doc_chunks("documents::one")] == [
        ("c:0", "Invoice total is 42 dollars."), ("c:1", "Payment due Friday."),
    ]
    assert final.count_chunks() == 2
    assert IndexRequestQueue(tmp_path).pending("chunks", limit=10) == []
