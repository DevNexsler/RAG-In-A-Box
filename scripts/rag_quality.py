#!/usr/bin/env python3
"""Offline RAG regression judgments; real LanceDB/FTS, deterministic embedding double."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.enrichment_postprocess import ground_card_suffixes, repair_enrichment

CORPUS = ROOT / "tests/fixtures/quality/rag-v1.json"


class LexicalEmbeddings:
    """Stable provider double. Measures retrieval plumbing, not embedding-model quality."""

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * 768
        for token in re.findall(r"\w+", text.lower()):
            slot = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=4).digest(), "big") % 768
            vector[slot] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


class UnavailableEmbeddings(LexicalEmbeddings):
    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("embedding provider unavailable")


def ranking_metrics(returned: list[str], relevant: list[str], k: int) -> dict[str, float]:
    """Document-level binary relevance; duplicate chunks never earn extra credit."""
    if k <= 0 or len(relevant) != len(set(relevant)):
        raise ValueError("positive k and unique judgments required")
    ranked = returned[:k]
    expected = set(relevant)
    if not expected:
        score = float(not ranked)
        return {"recall_at_k": score, "reciprocal_rank": score, "ndcg_at_k": score}
    seen = set()
    positions = []
    for i, doc_id in enumerate(ranked):
        if doc_id in expected and doc_id not in seen:
            positions.append(i + 1)
        seen.add(doc_id)
    dcg = sum(1 / math.log2(position + 1) for position in positions)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(expected))))
    return {"recall_at_k": len(positions) / len(expected),
            "reciprocal_rank": 1 / positions[0] if positions else 0.0,
            "ndcg_at_k": dcg / ideal}


def make_node(doc_id: str, text: str, *, source_type: str = "md", vector=None):
    from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
    node = TextNode(id_=f"{doc_id}::c:0", text=text,
                    embedding=LexicalEmbeddings().embed_query(text) if vector is None else vector,
                    metadata={"doc_id": doc_id, "rel_path": doc_id, "loc": "c:0", "source_type": source_type, "mtime": 1.0})
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
    return node


def run_quality(index_root: Path, corpus_path: Path = CORPUS, *, embed_provider=None) -> dict:
    from lancedb_store import LanceDBStore
    from search_hybrid import hybrid_search
    raw = corpus_path.read_bytes()
    corpus = json.loads(raw)
    if corpus.get("version") != 1 or not corpus.get("queries") or not corpus.get("grounding"):
        raise ValueError("versioned, nonempty retrieval and grounding judgments required")
    documents = {doc["id"]: doc for doc in corpus["documents"]}
    if len(documents) != len(corpus["documents"]):
        raise ValueError("duplicate document judgments")
    if index_root.exists() and any(index_root.iterdir()):
        raise ValueError("quality index must be empty")
    store = LanceDBStore(index_root, "quality")
    configured_provider = embed_provider or LexicalEmbeddings()
    vectors = configured_provider.embed_texts([doc["text"] for doc in documents.values()])
    store.upsert_nodes([make_node(doc["id"], doc["text"], source_type=doc["source_type"], vector=vector)
                        for doc, vector in zip(documents.values(), vectors, strict=True)])
    store.create_fts_index()
    retrieval = []
    for case in corpus["queries"]:
        if not set(case["relevant"]) <= documents.keys():
            raise ValueError("judgment references missing document")
        provider = UnavailableEmbeddings() if case.get("embedding_failure") else configured_provider
        hits = hybrid_search(store, provider, case["query"], final_top_k=case["k"],
                             importance_weight=0.0, media_intent_weight=0.0,
                             **{key: case[key] for key in ("source_type", "doc_id_prefix") if key in case})
        returned = [hit.doc_id for hit in hits]
        metrics = ranking_metrics(returned, case["relevant"], case["k"])
        grounded = all(hit.doc_id in documents and hit.text == documents[hit.doc_id]["text"] for hit in hits)
        passed = (metrics["recall_at_k"] >= case["min_recall"] and metrics["reciprocal_rank"] >= case["min_rr"]
                  and not set(returned).intersection(case.get("forbidden", [])) and grounded
                  and hits.diagnostics["degraded"] == bool(case.get("embedding_failure")))
        retrieval.append({"id": case["id"], "returned": returned, **metrics, "source_grounded": grounded,
                          "degraded": hits.diagnostics["degraded"], "passed": passed})
    grounding = []
    for case in corpus["grounding"]:
        repaired = repair_enrichment(case["prediction"], text=case["source"], title="", source_type="message")
        repaired, _ = ground_card_suffixes(repaired, source_text=case["source"])
        grounding.append({"id": case["id"], "passed": repaired == case["expected"], "actual": repaired})
    return {"corpus_version": corpus["version"], "corpus_sha256": hashlib.sha256(raw).hexdigest(),
            "mode": "configured-embedding-provider" if embed_provider else "offline-provider-double",
            "retrieval": retrieval, "grounding": grounding,
            "passed": all(row["passed"] for row in retrieval + grounding)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live-config", type=Path, help="embedding config; real providers may spend money")
    parser.add_argument("--allow-paid", action="store_true", help="authorize configured providers after live preflight")
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.unlink(missing_ok=True)
    provider = None
    identity = None
    if args.live_config:
        if not args.allow_paid:
            parser.error("--live-config requires --allow-paid")
        if subprocess.call([sys.executable, str(ROOT / 'scripts/live_preflight.py')]) != 0:
            args.output.write_text(json.dumps({'passed': False, 'error': 'live_preflight_failed'}) + '\n')
            return 1
        from core.config import load_config
        from providers.embed import build_embed_provider
        config = load_config(str(args.live_config))
        provider = build_embed_provider(config)
        identity = {key: config.get('embeddings', {}).get(key) for key in ('provider', 'model')}
    with tempfile.TemporaryDirectory(prefix="rag-quality-") as directory:
        report = run_quality(Path(directory), embed_provider=provider)
    if identity:
        report['embedding_provider'] = identity
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
