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


class CrossEncoderReranker(Reranker):
    """Re-rank using a cross-encoder model from sentence-transformers.

    Default model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB, fast).
    Install: pip install sentence-transformers
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "Cross-encoder re-ranking requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            )
        self._model = CrossEncoder(model_name)
        logger.info("CrossEncoderReranker loaded: %s", model_name)

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        if not hits:
            return hits
        pairs = [(query, h.text) for h in hits]
        scores = self._model.predict(pairs)
        for hit, score in zip(hits, scores):
            hit.score = float(score)
        return sorted(hits, key=lambda h: -h.score)


class LlamaCppReranker(Reranker):
    """Re-rank using Qwen3-Reranker via llama.cpp server's native /v1/rerank endpoint.

    The server is auto-managed: started on first use, shut down after idle timeout.
    If the server cannot be started, reranking blocks and retries rather than skipping.

    Setup:
        brew install llama.cpp
        # Model GGUF must be converted with llama.cpp's convert_hf_to_gguf.py
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8787",
        model_name: str = "qwen3-reranker",
        timeout: float = 120.0,
        model_path: str = "",
        idle_timeout: float = 300.0,
        heartbeat_dir: str = "",
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.model_path = model_path
        self.idle_timeout = idle_timeout
        self._server_manager = None

        if model_path:
            from llama_server import LlamaServerManager, DEFAULT_HEARTBEAT_DIR
            parsed_port = int(base_url.rstrip("/").rsplit(":", 1)[-1])
            self._server_manager = LlamaServerManager.get_instance(
                name="reranker",
                model_path=model_path,
                port=parsed_port,
                server_flags=["--reranking", "-c", "4096", "-b", "4096", "-ub", "4096"],
                idle_timeout=idle_timeout,
                heartbeat_dir=heartbeat_dir or DEFAULT_HEARTBEAT_DIR,
            )

        logger.info(
            "LlamaCppReranker initialized: %s/v1/rerank (model=%s, auto_manage=%s)",
            self.base_url, model_name, bool(self._server_manager),
        )

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        """Score all candidates in one batch call, return sorted by relevance.

        If the server is not running and a model_path was configured, the server
        is started automatically (blocking until ready). The server shuts down
        after idle_timeout seconds of inactivity.

        Raises RuntimeError on failure so the caller can track degradation.
        """
        if not hits:
            return hits

        import httpx

        if self._server_manager:
            if not self._server_manager.ensure_running():
                raise RuntimeError("Could not start llama-server for reranking")

        documents = [h.text[:4000] for h in hits]

        try:
            resp = httpx.post(
                f"{self.base_url}/v1/rerank",
                json={
                    "model": self.model_name,
                    "query": query,
                    "documents": documents,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            if self._server_manager:
                logger.warning(
                    "Rerank call failed, attempting server restart: %s", e,
                )
                if self._server_manager.ensure_running():
                    try:
                        resp = httpx.post(
                            f"{self.base_url}/v1/rerank",
                            json={
                                "model": self.model_name,
                                "query": query,
                                "documents": documents,
                            },
                            timeout=self.timeout,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                    except Exception as e2:
                        raise RuntimeError(f"Rerank failed after server restart: {e2}") from e2
                else:
                    raise RuntimeError("Reranker server restart failed") from e
            else:
                raise RuntimeError(f"Reranker call failed: {e}") from e

        scores_by_idx: dict[int, float] = {}
        for result in data.get("results", []):
            scores_by_idx[result["index"]] = result["relevance_score"]

        for i, hit in enumerate(hits):
            hit.score = scores_by_idx.get(i, 0.0)

        ranked = sorted(hits, key=lambda h: -h.score)

        if self._server_manager:
            self._server_manager.touch()

        logger.info(
            "LlamaCppReranker: scored %d candidates, top_score=%.4f",
            len(ranked),
            ranked[0].score if ranked else 0.0,
        )
        return ranked


class BasetenReranker(Reranker):
    """Re-rank using Qwen3-Reranker-8B via Baseten vLLM.

    Uses /v1/chat/completions with logprobs to compute P(yes) relevance
    scores. Each (query, doc) pair is scored individually via the
    Qwen3-Reranker chat template with thinking disabled.
    """

    _RETRY_BACKOFF = (10.0, 20.0, 30.0)
    _MAX_RETRIES = 3

    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
    ):
        import os

        self.model_id = model_id
        self.api_key = api_key or os.environ.get("BASETEN_API_KEY", "")
        self.timeout = timeout
        self.instruction = instruction
        self.base_url = (
            f"https://model-{model_id}.api.baseten.co"
            f"/environments/production/sync"
        )

        if not self.api_key:
            raise ValueError(
                "BASETEN_API_KEY not set. Set it in .env or pass api_key."
            )

        self._ready = False
        logger.info("BasetenReranker initialized: model_id=%s", model_id)

    def _ensure_ready(self) -> None:
        """Wake the Baseten deployment and wait until it responds.

        Sends a lightweight /v1/models probe every 10s for up to 120s.
        This absorbs scale-to-zero cold start time *before* scoring,
        so actual rerank requests don't timeout mid-flight.
        """
        if self._ready:
            return

        import httpx
        import time

        headers = {"Authorization": f"Api-Key {self.api_key}"}
        max_wait = 120.0
        poll_interval = 10.0
        start = time.monotonic()

        logger.info("Waiting for Baseten reranker to become ready...")
        while time.monotonic() - start < max_wait:
            try:
                resp = httpx.get(
                    f"{self.base_url}/v1/models",
                    headers=headers,
                    timeout=15.0,
                )
                if resp.status_code == 200:
                    elapsed = time.monotonic() - start
                    self._ready = True
                    logger.info("Baseten reranker ready (%.1fs)", elapsed)
                    return
            except Exception:
                pass
            time.sleep(poll_interval)

        logger.warning(
            "Baseten reranker not ready after %.0fs — proceeding anyway",
            time.monotonic() - start,
        )

    def _score_hit(self, query: str, hit: SearchHit, client: "httpx.Client") -> float:
        """Score a single (query, doc) pair via vLLM logprobs."""
        import math

        prompt = (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {hit.text[:4000]}"
        )
        resp = client.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": "qwen3-reranker",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        resp.raise_for_status()
        data = resp.json()

        choices = data.get("choices", [])
        if not choices:
            return 0.0

        logprobs_data = choices[0].get("logprobs", {})
        content = logprobs_data.get("content", []) if logprobs_data else []
        if not content:
            return 0.0

        top_lps = content[0].get("top_logprobs", [])
        yes_lp = next((lp["logprob"] for lp in top_lps if lp["token"].lower() == "yes"), None)
        no_lp = next((lp["logprob"] for lp in top_lps if lp["token"].lower() == "no"), None)

        if yes_lp is not None and no_lp is not None:
            max_lp = max(yes_lp, no_lp)
            yes_exp = math.exp(yes_lp - max_lp)
            no_exp = math.exp(no_lp - max_lp)
            return yes_exp / (yes_exp + no_exp)
        if yes_lp is not None:
            return math.exp(yes_lp)
        return 0.0

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        """Score candidates via vLLM logprobs, return sorted by relevance.

        On first call, waits for the deployment to wake up (scale-to-zero).
        Then retries with backoff on transient failures.
        """
        if not hits:
            return hits

        self._ensure_ready()

        import httpx
        import time

        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
        }

        last_exc = None
        for attempt in range(self._MAX_RETRIES + 1):
            try:
                with httpx.Client(headers=headers, timeout=self.timeout) as client:
                    for hit in hits:
                        hit.score = self._score_hit(query, hit, client)

                ranked = sorted(hits, key=lambda h: -h.score)
                logger.info(
                    "BasetenReranker: scored %d candidates, top_score=%.4f",
                    len(ranked),
                    ranked[0].score if ranked else 0.0,
                )
                return ranked

            except Exception as e:
                last_exc = e
                if attempt < self._MAX_RETRIES:
                    backoff = self._RETRY_BACKOFF[min(attempt, len(self._RETRY_BACKOFF) - 1)]
                    logger.warning(
                        "rerank() attempt %d/%d failed (%s: %s), retrying in %.0fs...",
                        attempt + 1, self._MAX_RETRIES + 1,
                        type(e).__name__, e, backoff,
                    )
                    time.sleep(backoff)
                else:
                    raise RuntimeError(
                        f"Baseten reranker failed after {attempt + 1} attempts: {e}"
                    ) from e

        raise RuntimeError(f"Baseten reranker failed: {last_exc}") from last_exc


def build_reranker(config: dict) -> Reranker | None:
    """Factory: build a Reranker from the search.reranker config section.

    Config example (baseten — cloud):
        search:
          reranker:
            enabled: true
            provider: "baseten"
            model_id: "wnppr2y3"

    Config example (llama_cpp — local, with auto-managed server):
        search:
          reranker:
            enabled: true
            provider: "llama_cpp"
            base_url: "http://localhost:8787"
            model_path: "models/qwen3-reranker-4b-q8_0.gguf"
            idle_timeout: 300

    Config example (cross_encoder):
        search:
          reranker:
            enabled: true
            provider: "cross_encoder"
            model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    """
    reranker_cfg = config.get("search", {}).get("reranker", {})
    if not reranker_cfg.get("enabled", False):
        return None

    provider = reranker_cfg.get("provider", "llama_cpp")

    if provider == "baseten":
        return BasetenReranker(
            model_id=reranker_cfg.get("model_id", ""),
            api_key=reranker_cfg.get("api_key"),
            timeout=reranker_cfg.get("timeout", 60.0),
        )
    elif provider == "llama_cpp":
        return LlamaCppReranker(
            base_url=reranker_cfg.get("base_url", "http://localhost:8787"),
            model_name=reranker_cfg.get("model", "qwen3-reranker"),
            timeout=reranker_cfg.get("timeout", 120.0),
            model_path=reranker_cfg.get("model_path", ""),
            idle_timeout=reranker_cfg.get("idle_timeout", 1800.0),
            heartbeat_dir=config.get("server", {}).get("heartbeat_dir", ""),
        )
    elif provider == "cross_encoder":
        return CrossEncoderReranker(
            model_name=reranker_cfg.get(
                "model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
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
