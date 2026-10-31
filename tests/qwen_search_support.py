"""Isolated corpus for real search-consumer and gateway qualification."""

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

from lancedb_store import LanceDBStore

QUERY = "Which tenant actually paid the outstanding invoice?"
ANSWER_ID = "qwen/tenant-payment"


class FixedEmbeddings:
    def embed_query(self, query):
        return [1.0] + [0.0] * 7


def corpus(root, count=72, dense=True):
    nodes = []
    for i in range(count):
        paid = i == count - 1
        doc_id = ANSWER_ID if paid else f"qwen/invoice-{i}"
        text = ("Tenant Mira actually paid the outstanding invoice in full; payment received and cleared."
                if paid else f"Tenant Sam promised to pay an outstanding invoice but no payment was received. Record {i}.")
        if dense:
            text += "\nLedger entries: " + "1234567890 9876543210 " * 300
        vector = [1.0] + [0.0] * 7
        # Distinct directions retain all occurrences through MMR.
        vector[1 + i % 7] = .5 + i / count
        node = TextNode(text=text, id_=f"{doc_id}::0", embedding=vector,
                        metadata={"doc_id": doc_id, "loc": "0", "snippet": text[:200],
                                  "source_type": "md", "mtime": 1.0, "size": len(text)})
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
        nodes.append(node)
    store = LanceDBStore(str(root), "qwen_contract")
    store.upsert_nodes(nodes)
    store.create_fts_index()
    config = {"index_root": str(root), "embeddings": {"provider": "fixed-test"}, "search": {
        "vector_top_k": count, "keyword_top_k": count,
        "importance": {"weight": 0}, "recency": {"weight": 0},
        "reranker": {"enabled": True, "provider": "qwen-bulk"},
    }}
    return store, FixedEmbeddings(), config
