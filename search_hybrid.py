"""Hybrid search: parallel vector + keyword (BM25/FTS), fuse with RRF, optional re-rank.

Architecture:
  1. Embed query → vector search (semantic similarity)
  2. Keyword search (BM25/FTS via tantivy) — runs in parallel with step 1
  3. Reciprocal Rank Fusion (RRF) to merge both ranked lists
  4. Optional cross-encoder re-ranker for maximum precision
  5. Filters (doc_id_prefix, source_type) and final top_k

RRF reference: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet
and individual Rank Learning Methods" (SIGIR 2009).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from core.storage import SearchHit

if TYPE_CHECKING:
    from lancedb_store import LanceDBStore
    from providers.embed.base import EmbedProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Search result with diagnostics
# ---------------------------------------------------------------------------

class SearchResult:
    """Hybrid search result — behaves like a list of SearchHit, plus diagnostics.

    diagnostics dict:
        keyword_search_active: bool — True if BM25/FTS returned results successfully.
        reranker_applied: bool — True if the cross-encoder reranker ran successfully.
        degraded: bool — True if any retrieval stage failed silently.
    """

    __slots__ = ("hits", "diagnostics")

    def __init__(
        self,
        hits: list[SearchHit],
        diagnostics: dict | None = None,
    ) -> None:
        self.hits = hits
        self.diagnostics = diagnostics or {
            "vector_search_active": True,
            "keyword_search_active": True,
            "reranker_applied": False,
            "degraded": False,
        }

    # List-compatible interface so existing callers (for h in result, result[0], etc.) still work.
    def __len__(self) -> int:
        return len(self.hits)

    def __getitem__(self, idx):
        return self.hits[idx]

    def __iter__(self):
        return iter(self.hits)

    def __bool__(self) -> bool:
        return bool(self.hits)

    def __eq__(self, other):
        if isinstance(other, list):
            return self.hits == other
        return NotImplemented

    def __repr__(self) -> str:
        return f"SearchResult(hits={len(self.hits)}, diagnostics={self.diagnostics})"


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    result_lists: list[list[SearchHit]],
    k: int = 60,
) -> list[SearchHit]:
    """Fuse multiple ranked result lists using RRF.

    score(doc) = Σ  1 / (k + rank_i)   for each list i where doc appears.
    k=60 is near-optimal per Cormack et al.

    Deduplicates by (doc_id, loc) — keeps the first occurrence's metadata.
    """
    scores: dict[tuple[str, str], float] = {}
    hits_map: dict[tuple[str, str], SearchHit] = {}

    for result_list in result_lists:
        for rank, hit in enumerate(result_list):
            key = (hit.doc_id, hit.loc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in hits_map:
                hits_map[key] = hit

    sorted_keys = sorted(scores.keys(), key=lambda key: -scores[key])

    fused: list[SearchHit] = []
    for key in sorted_keys:
        hit = hits_map[key]
        hit.score = scores[key]
        fused.append(hit)

    return fused


# ---------------------------------------------------------------------------
# Reranker protocol (optional)
# ---------------------------------------------------------------------------

class Reranker:
    """Base class for re-rankers. Subclass and override `rerank`."""

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        """Re-score and re-sort hits. Return sorted list (best first)."""
        raise NotImplementedError


class DeepInfraReranker(Reranker):
    """Re-rank using Qwen3-Reranker-8B via DeepInfra inference API.

    Always-on (no cold start), batch scoring (one API call), direct relevance scores.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-Reranker-8B",
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        import os

        self.model = model
        self.api_key = api_key or os.environ.get("DEEPINFRA_API_KEY", "")
        self.timeout = timeout
        self.base_url = f"https://api.deepinfra.com/v1/inference/{model}"

        if not self.api_key:
            raise ValueError(
                "DEEPINFRA_API_KEY not set. Set it in .env or pass api_key."
            )

        logger.info("DeepInfraReranker initialized: model=%s", model)

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        """Score all candidates in one batch call, return sorted by relevance."""
        if not hits:
            return hits

        import httpx

        documents = [h.text[:4000] for h in hits]

        try:
            resp = httpx.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "queries": [query],
                    "documents": documents,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"DeepInfra reranker failed: {e}") from e

        scores = data.get("scores", [])
        for i, score in enumerate(scores):
            if i < len(hits):
                hits[i].score = score

        ranked = sorted(hits, key=lambda h: -h.score)
        logger.info(
            "DeepInfraReranker: scored %d candidates, top_score=%.4f",
            len(ranked),
            ranked[0].score if ranked else 0.0,
        )
        return ranked


def build_reranker(config: dict) -> Reranker | None:
    """Factory: build a Reranker from the search.reranker config section.

    Config example:
        search:
          reranker:
            enabled: true
            provider: "deepinfra"
            model: "Qwen/Qwen3-Reranker-8B"
            timeout: 30
    """
    reranker_cfg = config.get("search", {}).get("reranker", {})
    if not reranker_cfg.get("enabled", False):
        return None

    provider = reranker_cfg.get("provider", "deepinfra")

    if provider == "deepinfra":
        return DeepInfraReranker(
            model=reranker_cfg.get("model", "Qwen/Qwen3-Reranker-8B"),
            api_key=reranker_cfg.get("api_key"),
            timeout=reranker_cfg.get("timeout", 30.0),
        )
    else:
        logger.warning("Unknown reranker provider: %s", provider)
        return None


# ---------------------------------------------------------------------------
# Recency boost
# ---------------------------------------------------------------------------

def _apply_recency_boost(
    hits: list[SearchHit],
    half_life_days: float = 90.0,
    weight: float = 0.3,
) -> list[SearchHit]:
    """Apply a time-decay multiplier to RRF scores, then re-sort.

    boost = 1.0 + weight * decay, where decay = 1 / (1 + age_days / half_life).
    Recent docs get up to +weight boost; old docs approach 1.0 (no penalty).
    """
    now = time.time()
    for hit in hits:
        if hit.mtime and hit.mtime > 0:
            age_days = max((now - hit.mtime) / 86400.0, 0.0)
            decay = 1.0 / (1.0 + age_days / half_life_days)
            hit.score *= (1.0 + weight * decay)
    hits.sort(key=lambda h: -h.score)
    return hits


# ---------------------------------------------------------------------------
# Main hybrid search
# ---------------------------------------------------------------------------

def hybrid_search(
    store: "LanceDBStore",
    embed_provider: "EmbedProvider",
    query: str,
    vector_top_k: int = 50,
    keyword_top_k: int = 50,
    final_top_k: int = 10,
    rrf_k: int = 60,
    doc_id_prefix: str | None = None,
    source_type: str | None = None,
    tags: str | None = None,
    status: str | None = None,
    folder: str | None = None,
    reranker: Reranker | None = None,
    prefer_recent: bool = False,
    recency_half_life_days: float = 90.0,
    recency_weight: float = 0.3,
    metadata_filters: dict[str, str] | None = None,
    enr_doc_type: str | None = None,
    enr_topics: str | None = None,
) -> SearchResult:
    """Run vector + keyword search in parallel, fuse with RRF, optionally re-rank.

    Filters (source_type, folder, tags, enr_doc_type, enr_topics, etc.) are
    applied as LanceDB pre-filters *before* ANN/FTS scoring, so the full
    top_k results match the filter criteria.

    Returns a SearchResult (list-compatible) with a diagnostics dict:
        keyword_search_active: True if BM25/FTS returned results successfully.
        reranker_applied: True if the cross-encoder reranker ran successfully.
        degraded: True if any retrieval stage failed silently.
    """
    if not query.strip():
        return SearchResult(hits=[])

    diagnostics = {
        "vector_search_active": True,
        "keyword_search_active": True,
        "reranker_applied": False,
        "degraded": False,
    }

    # 0. Build WHERE clause for pre-filtering (applied inside LanceDB before scoring)
    where = store._build_where_clause(
        doc_id_prefix=doc_id_prefix,
        source_type=source_type,
        status=status,
        folder=folder,
        tags=tags,
        enr_doc_type=enr_doc_type,
        enr_topics=enr_topics,
        metadata_filters=metadata_filters,
    )

    # 1. Embed query
    query_vector = embed_provider.embed_query(query)

    # 2. Parallel retrieval: vector (semantic) + keyword (BM25/FTS)
    with ThreadPoolExecutor(max_workers=2) as executor:
        vec_future = executor.submit(store.vector_search, query_vector, vector_top_k, where)
        kw_future = executor.submit(store.keyword_search, query, keyword_top_k, where)
        try:
            vector_hits = vec_future.result()
        except Exception as e:
            logger.warning("Vector search failed (degraded to keyword-only): %s", e)
            vector_hits = []
            diagnostics["vector_search_active"] = False
        try:
            keyword_hits = kw_future.result()
        except Exception as e:
            logger.warning("Keyword/FTS search failed (degraded to vector-only): %s", e)
            keyword_hits = []
            diagnostics["keyword_search_active"] = False

    logger.info(
        "Retrieval: %d vector hits, %d keyword hits (fts_ok=%s, where=%s)",
        len(vector_hits),
        len(keyword_hits),
        diagnostics["keyword_search_active"],
        where,
    )

    # 3. Reciprocal Rank Fusion
    fused = reciprocal_rank_fusion([vector_hits, keyword_hits], k=rrf_k)

    # 3b. Optional recency boost (applied after fusion so boost is uniform)
    if prefer_recent:
        fused = _apply_recency_boost(fused, recency_half_life_days, recency_weight)

    # 4. Filters are now applied as pre-filters in LanceDB (step 0), no post-filtering needed.

    # 5. Optional cross-encoder re-rank (on top N candidates for efficiency)
    if reranker is not None:
        rerank_pool = fused[: final_top_k * 6]
        try:
            fused = reranker.rerank(query, rerank_pool)
            diagnostics["reranker_applied"] = True
        except Exception as e:
            logger.warning("Reranker failed (degraded to RRF order): %s", e)

    # 6. Compute degraded flag
    diagnostics["degraded"] = (
        not diagnostics["vector_search_active"]
        or not diagnostics["keyword_search_active"]
        or (reranker is not None and not diagnostics["reranker_applied"])
    )

    # 7. Final top_k
    return SearchResult(hits=fused[:final_top_k], diagnostics=diagnostics)
