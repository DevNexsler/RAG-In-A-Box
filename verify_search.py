"""Diagnostic: full 3-stage search pipeline — vector, BM25, RRF fusion, Qwen3 rerank."""
import sys
import time
from dotenv import load_dotenv
load_dotenv()

from core.config import load_config
from lancedb_store import LanceDBStore
from providers.embed import build_embed_provider
from search_hybrid import hybrid_search, reciprocal_rank_fusion, LlamaCppReranker

config = load_config("config_test.yaml")
store = LanceDBStore(config["index_root"], config.get("lancedb", {}).get("table", "chunks"))
embed = build_embed_provider(config)

# Show what's indexed
doc_ids = store.list_doc_ids()
print("=" * 80)
print(f"INDEXED DOCUMENTS ({len(doc_ids)})")
print("=" * 80)
for d in sorted(doc_ids):
    print(f"  {d}")
print()

QUERY = "kimchi fermentation recipe"
TOP_K = 10

print("=" * 80)
print(f'QUERY: "{QUERY}"')
print("=" * 80)

# Stage 1: Vector search
print("\n--- STAGE 1: Vector Search (Gemini Embeddings) ---")
query_vector = embed.embed_query(QUERY)
vector_hits = store.vector_search(query_vector, top_k=TOP_K)
for i, h in enumerate(vector_hits):
    print(f"  #{i+1:2d}  score={h.score:.4f}  {h.doc_id:<30s} {h.loc}")

# Stage 2: Keyword search
print("\n--- STAGE 2: Keyword Search (BM25/tantivy FTS) ---")
keyword_hits = store.keyword_search(QUERY, top_k=TOP_K)
if keyword_hits:
    for i, h in enumerate(keyword_hits):
        print(f"  #{i+1:2d}  score={h.score:.4f}  {h.doc_id:<30s} {h.loc}")
else:
    print("  (no FTS results)")

# Stage 3: RRF fusion
print("\n--- STAGE 3: Reciprocal Rank Fusion (k=60) ---")
fused = reciprocal_rank_fusion([vector_hits, keyword_hits], k=60)
for i, h in enumerate(fused[:TOP_K]):
    print(f"  #{i+1:2d}  rrf={h.score:.6f}  {h.doc_id:<30s} {h.loc}")

# Stage 4: Qwen3 Reranker
print("\n--- STAGE 4: Qwen3-Reranker-0.6B (llama.cpp server) ---")
reranker = LlamaCppReranker(base_url="http://localhost:8787")
candidates = fused[:TOP_K]

t0 = time.perf_counter()
reranked = reranker.rerank(QUERY, candidates)
rerank_time = time.perf_counter() - t0

for i, h in enumerate(reranked):
    print(f"  #{i+1:2d}  relevance={h.score:.6f}  {h.doc_id:<30s} {h.loc}")
    print(f"          snippet: {h.snippet[:80]}")

print(f"\n  Rerank time: {rerank_time:.2f}s for {len(candidates)} candidates")

# Stage 5: Full pipeline via hybrid_search()
print("\n--- FULL PIPELINE: hybrid_search() with reranker ---")
t0 = time.perf_counter()
final_hits = hybrid_search(
    store, embed, QUERY,
    vector_top_k=TOP_K, keyword_top_k=TOP_K, final_top_k=5, rrf_k=60,
    reranker=reranker,
)
full_time = time.perf_counter() - t0
for i, h in enumerate(final_hits):
    print(f"  #{i+1:2d}  score={h.score:.6f}  {h.doc_id:<30s} {h.loc}")
    print(f"          text: {h.text[:120]}")
    print()
print(f"  Total search time: {full_time:.2f}s")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
