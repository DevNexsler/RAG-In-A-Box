"""Hybrid search: parallel vector + keyword (BM25/FTS), fuse with RRF, optional re-rank.

Architecture:
  1. Embed query → vector search (semantic similarity)
  2. Keyword search (BM25/FTS via tantivy) — runs in parallel with step 1
  3. Reciprocal Rank Fusion (RRF) to merge both ranked lists
  4. Length normalization (prevents long chunks from dominating)
  5. Importance weighting (boosts high-priority documents)
  6. Optional recency boost + time decay with floor
  7. Optional cross-encoder re-rank with cosine fallback on failure
  8. MMR diversity (removes near-duplicate chunks)
  9. Minimum score threshold (discards low-relevance noise)
  10. Final top_k

RRF reference: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet
and individual Rank Learning Methods" (SIGIR 2009).

Enhancements inspired by memory-lancedb-pro (win4r/memory-lancedb-pro):
  - Length normalization: log-based penalty for overly long chunks
  - Importance weighting: score *= (1 - w + w * importance) based on metadata field
  - MMR diversity: conservative cosine deduplication (threshold 0.95)
  - Cross-encoder blend: 60/40 reranker/original score preservation
  - Cosine fallback: lightweight rerank when cross-encoder fails
  - Time decay floor: old docs never lose more than 50% relevance
  - Minimum score threshold: filters noise below configurable cutoff
"""

from __future__ import annotations

import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from core.storage import SearchHit
from core.tracing import get_tracer

if TYPE_CHECKING:
    from lancedb_store import LanceDBStore
    from providers.embed.base import EmbedProvider

# Lazy tracer (resolves provider per call); spans are no-ops when tracing is off.
_tracer = get_tracer("pipeline")

logger = logging.getLogger(__name__)

_TRANSIENT_FTS_ERROR_MARKERS = (
    "Added column's length must match table's length",
)


def _is_transient_fts_error(error: Exception) -> bool:
    """Detect known transient LanceDB FTS errors during index rebuild windows."""
    text = str(error)
    return any(marker in text for marker in _TRANSIENT_FTS_ERROR_MARKERS)


def _retry_keyword_search(
    store: "LanceDBStore",
    query: str,
    top_k: int,
    where: str | None,
    error: Exception,
) -> tuple[list[SearchHit], Exception | None]:
    """Retry keyword search with backoff for transient FTS rebuild races."""
    retry_delays = [0.25, 0.5] if _is_transient_fts_error(error) else [0.0]
    last_error: Exception = error

    for delay in retry_delays:
        if delay > 0:
            time.sleep(delay)
        try:
            return store.keyword_search(query, top_k, where), None
        except Exception as exc:  # pragma: no cover - exercised via caller behavior
            last_error = exc

    return [], last_error


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
# Length normalization
# ---------------------------------------------------------------------------

def _apply_length_normalization(
    hits: list[SearchHit],
    anchor: int = 800,
) -> list[SearchHit]:
    """Penalize long chunks that score high due to keyword density.

    Formula: score *= 1 / (1 + 0.5 * log2(len / anchor))
    Chunks at or below `anchor` chars are unaffected (factor >= 1.0 capped to 1.0).
    Very long chunks (e.g., 3200 chars at anchor=800) get ~0.67x.

    Inspired by memory-lancedb-pro's length normalization approach.
    """
    for hit in hits:
        text_len = len(hit.text) if hit.text else 0
        if text_len > anchor:
            factor = 1.0 / (1.0 + 0.5 * math.log2(text_len / anchor))
            hit.score *= factor
    hits.sort(key=lambda h: -h.score)
    return hits


# ---------------------------------------------------------------------------
# Importance weighting
# ---------------------------------------------------------------------------

def _apply_importance_weighting(
    hits: list[SearchHit],
    field: str = "enr_importance",
    weight: float = 0.3,
) -> list[SearchHit]:
    """Boost documents with higher importance/priority metadata.

    Reads a numeric metadata field (0.0–1.0) from each hit and scales the score:
        score *= (1 - weight + weight * importance)

    With default weight=0.3:
        importance=1.0 → score *= 1.0 (no change)
        importance=0.5 → score *= 0.85
        importance=0.0 → score *= 0.7

    The field is looked up on the hit object first (e.g., frontmatter-promoted
    columns like ``priority``), then in ``extra_metadata``. If the field is
    missing or non-numeric, importance defaults to 0.5 (neutral).

    Inspired by memory-lancedb-pro's importance weighting.
    """
    for hit in hits:
        # Try direct attribute, then extra_metadata
        raw = getattr(hit, field, None)
        if not raw:  # None or empty string
            raw = (hit.extra_metadata or {}).get(field)

        # Parse to float, default to 0.5 (neutral) if missing/invalid
        try:
            importance = float(raw) if raw else 0.5
        except (TypeError, ValueError):
            importance = 0.5

        # Guard against NaN/Inf, then clamp to [0, 1]
        if math.isnan(importance) or math.isinf(importance):
            importance = 0.5
        importance = max(0.0, min(1.0, importance))

        factor = (1.0 - weight) + weight * importance
        hit.score *= factor

    hits.sort(key=lambda h: -h.score)
    return hits


# ---------------------------------------------------------------------------
# Minimum score threshold
# ---------------------------------------------------------------------------

def _apply_min_score_threshold(
    hits: list[SearchHit],
    threshold: float = 0.0,
) -> list[SearchHit]:
    """Discard results below a minimum score threshold.

    Set threshold=0.0 (default) to disable — all results pass through.
    Typical production values: 0.01–0.05 for RRF scores.

    Inspired by memory-lancedb-pro's noise filtering.
    """
    if threshold <= 0.0:
        return hits
    return [h for h in hits if h.score >= threshold]


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

    Uses 60/40 blending: final_score = 0.6 * reranker_score + 0.4 * original_score.
    This preserves retrieval-stage signal and prevents the cross-encoder from
    completely overriding the fusion stage.
    """

    BLEND_RERANKER = 0.6
    BLEND_ORIGINAL = 0.4

    def __init__(
        self,
        model: str = "Qwen/Qwen3-Reranker-8B",
        api_key: str | None = None,
        timeout: float = 30.0,
        base_url: str | None = None,
    ):
        import os

        self.model = model
        self.api_key = api_key or os.environ.get("DEEPINFRA_API_KEY", "")
        self.timeout = timeout
        self._base_url = (base_url or "https://api.deepinfra.com").rstrip("/")
        self.base_url = f"{self._base_url}/v1/inference/{model}"

        if not self.api_key:
            raise ValueError(
                "DEEPINFRA_API_KEY not set. Set it in .env or pass api_key."
            )

        logger.info("DeepInfraReranker initialized: model=%s", model)

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        """Score all candidates in one batch call, blend with original scores, return sorted."""
        if not hits:
            return hits

        import httpx

        # Preserve original scores for blending
        original_scores = {(h.doc_id, h.loc): h.score for h in hits}

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

        # Normalize reranker scores to [0, 1] for blending
        raw_scores = data.get("scores", [])
        if raw_scores:
            max_score = max(raw_scores)
            min_score = min(raw_scores)
            score_range = max_score - min_score
            if score_range > 0:
                norm_scores = [(s - min_score) / score_range for s in raw_scores]
            else:
                norm_scores = [1.0] * len(raw_scores)
        else:
            norm_scores = []

        # Normalize original scores to [0, 1] for blending
        orig_vals = [original_scores.get((h.doc_id, h.loc), 0.0) for h in hits]
        if orig_vals:
            max_orig = max(orig_vals)
            min_orig = min(orig_vals)
            orig_range = max_orig - min_orig
            if orig_range > 0:
                norm_orig = [(v - min_orig) / orig_range for v in orig_vals]
            else:
                norm_orig = [1.0] * len(orig_vals)
        else:
            norm_orig = []

        # Blend: 60% reranker + 40% original
        for i in range(len(hits)):
            reranker_s = norm_scores[i] if i < len(norm_scores) else 0.0
            original_s = norm_orig[i] if i < len(norm_orig) else 0.0
            hits[i].score = self.BLEND_RERANKER * reranker_s + self.BLEND_ORIGINAL * original_s

        ranked = sorted(hits, key=lambda h: -h.score)
        logger.info(
            "DeepInfraReranker: scored %d candidates (60/40 blend), top_score=%.4f",
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
            base_url=reranker_cfg.get("base_url"),
        )
    else:
        logger.warning("Unknown reranker provider: %s", provider)
        return None


# ---------------------------------------------------------------------------
# Cosine similarity fallback reranker
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _cosine_fallback_rerank(
    query_vector: list[float],
    hits: list[SearchHit],
    store: "LanceDBStore",
    blend_cosine: float = 0.3,
    blend_original: float = 0.7,
) -> list[SearchHit]:
    """Lightweight rerank using cosine similarity when cross-encoder fails.

    Retrieves stored embeddings and computes cosine similarity with the query
    vector. Blends 70% original score + 30% cosine similarity.

    Falls back silently to original order if embeddings can't be retrieved.
    """
    try:
        for hit in hits:
            chunk_uid = f"{hit.doc_id}::{hit.loc}"
            stored_vec = store.get_vector(chunk_uid)
            if stored_vec is not None:
                cos_sim = _cosine_similarity(query_vector, stored_vec)
                hit.score = blend_original * hit.score + blend_cosine * cos_sim
        hits.sort(key=lambda h: -h.score)
        logger.info("Cosine fallback rerank applied to %d hits", len(hits))
    except Exception as e:
        logger.warning("Cosine fallback rerank failed: %s", e)
    return hits


# ---------------------------------------------------------------------------
# Recency boost + time decay
# ---------------------------------------------------------------------------

def _apply_recency_boost(
    hits: list[SearchHit],
    half_life_days: float = 90.0,
    weight: float = 0.3,
) -> list[SearchHit]:
    """Apply recency boost (additive) and time decay (multiplicative with floor).

    Two mechanisms (inspired by memory-lancedb-pro):

    1. Recency boost (additive): small bonus for recent content.
       boost = weight * exp(-age_days / half_life)
       Default: up to +0.3 for very recent docs, decaying to ~0 over months.

    2. Time decay (multiplicative, floor at 0.5): old content gradually loses
       relevance but never more than 50%.
       factor = 0.5 + 0.5 * exp(-age_days / decay_half_life)
       decay_half_life = half_life_days (same parameter, different mechanism).

    Combined: score = score * time_decay_factor + recency_boost
    """
    now = time.time()
    for hit in hits:
        if hit.mtime and hit.mtime > 0:
            age_days = max((now - hit.mtime) / 86400.0, 0.0)

            # Time decay (multiplicative, floor at 0.5)
            decay_factor = 0.5 + 0.5 * math.exp(-age_days / half_life_days)
            hit.score *= decay_factor

            # Recency boost (additive)
            recency_bonus = weight * math.exp(-age_days / half_life_days)
            hit.score += recency_bonus

    hits.sort(key=lambda h: -h.score)
    return hits


# ---------------------------------------------------------------------------
# MMR diversity
# ---------------------------------------------------------------------------

_MEDIA_INTENT_TERMS: dict[str, str] = {
    "video": "video", "videos": "video", "walkthrough": "video",
    "clip": "video", "clips": "video", "footage": "video",
    "photo": "img", "photos": "img", "pic": "img", "pics": "img",
    "picture": "img", "pictures": "img", "image": "img", "images": "img",
    "screenshot": "img", "screenshots": "img",
    "audio": "audio", "voicemail": "audio", "recording": "audio",
}


def _media_intent_types(query: str) -> set[str]:
    """source_types a query explicitly asks for ("video walkthrough" -> {"video"})."""
    if not query:
        return set()
    return {
        _MEDIA_INTENT_TERMS[token]
        for token in re.findall(r"[a-z]+", query.lower())
        if token in _MEDIA_INTENT_TERMS
    }


def _apply_media_intent_boost(
    hits: list[SearchHit], query: str, weight: float = 0.35
) -> list[SearchHit]:
    """Lift attachments when the query explicitly names the medium it wants.

    An attachment carries its neighbour-conversation context (e.g. the address
    from the next message) inside a multi-KB visual describe, so that address is
    a tiny fraction of the chunk. A query like "163 Washington video walkthrough"
    is therefore dominated by the *message* containing that address — it matches
    at ~1.0 because it literally is that text — and every top slot goes to
    messages. Measured on prod: the video sat in the candidate pool at rank 22
    (a sibling at rank 5) yet never survived the final top-10 cut.

    When the query names a medium, that medium is the thing being asked for, so
    boost it enough to survive selection. Queries that name no medium are
    returned untouched, so ordinary search is unaffected.
    """
    if not hits or not query or weight <= 0:
        return hits
    wanted = _media_intent_types(query)
    if not wanted:
        return hits
    for hit in hits:
        if (hit.source_type or "") in wanted:
            hit.score *= 1.0 + weight
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


def _ensure_media_intent_slots(
    hits: list[SearchHit],
    pool: list[SearchHit],
    query: str,
    min_slots: int = 2,
) -> list[SearchHit]:
    """Guarantee a query that names a medium returns some of that medium.

    Score boosting cannot win this contest: an attachment carries its
    conversation context (e.g. the property address) inside a multi-KB visual
    describe, while the messages quoting that same address ARE that text and
    re-rank at ~1.0. Measured on prod: filtered to videos the right clip ranks
    #1 at 1.205, but in a mixed pool every one of the ten slots goes to
    messages, so the attachment is never returned.

    When the query explicitly asks for a video/photo/recording, reserve a couple
    of slots for the best-scoring candidates of that type. Queries that name no
    medium are untouched.
    """
    if not hits or not query or min_slots <= 0:
        return hits
    wanted = _media_intent_types(query)
    if not wanted:
        return hits
    present = [h for h in hits if (h.source_type or "") in wanted]
    if len(present) >= min_slots:
        return hits
    seen = {(h.doc_id, h.loc) for h in hits}
    extras: list[SearchHit] = []
    for hit in pool:  # pool is score-ordered
        if len(present) + len(extras) >= min_slots:
            break
        if (hit.source_type or "") in wanted and (hit.doc_id, hit.loc) not in seen:
            extras.append(hit)
            seen.add((hit.doc_id, hit.loc))
    if not extras:
        return hits
    # Make room by dropping the weakest NON-matching hits (walking up from the
    # tail) so hits that already satisfy the quota are never evicted.
    room = len(extras)
    dropped: set[int] = set()
    for idx in range(len(hits) - 1, -1, -1):
        if room <= 0:
            break
        if (hits[idx].source_type or "") not in wanted:
            dropped.add(idx)
            room -= 1
    keep = [hit for idx, hit in enumerate(hits) if idx not in dropped]
    # Rescale injected scores onto the result set's scale. Recall-pass hits carry
    # raw vector/keyword scores (measured: 18.8 next to a fused 1.0), so anything
    # that re-sorts by score would rank a tail-injected hit first. Map them just
    # below the weakest kept hit, preserving their order relative to each other.
    if keep:
        floor = min(hit.score for hit in keep)
        top = max((hit.score for hit in extras), default=0.0)
        if top > 0:
            for extra in extras:
                extra.score = floor * 0.99 * (extra.score / top)
    return keep + extras


def _apply_mmr_diversity(
    hits: list[SearchHit],
    store: "LanceDBStore",
    similarity_threshold: float = 0.95,
    protected_top_k: int = 3,
    pool_limit: int | None = None,
    return_deferred: bool = False,
) -> list[SearchHit] | tuple[list[SearchHit], list[dict]]:
    """Remove near-duplicate chunks using Maximal Marginal Relevance.

    Greedy selection after a protected top band: for each later candidate,
    compute cosine similarity against selected results. If similarity is above
    threshold, defer it (append at end). This reduces duplicate spam without
    hiding the strongest few matches.

    Falls back silently to original order if vectors can't be retrieved.

    Inspired by memory-lancedb-pro's MMR diversity approach.
    """
    if len(hits) <= 1:
        return (hits, []) if return_deferred else hits

    def _chunk_uid(hit: SearchHit) -> str:
        return f"{hit.doc_id}::{hit.loc}"

    def _deferred_record(hit: SearchHit, similar_to: SearchHit, similarity: float) -> dict:
        record_uuid = _chunk_uid(hit)
        return {
            "uuid": record_uuid,
            "record_uuid": record_uuid,
            "doc_id": hit.doc_id,
            "loc": hit.loc,
            "similar_to_uuid": _chunk_uid(similar_to),
            "similarity": similarity,
            "score": hit.score,
            "title": hit.title,
            "rel_path": hit.rel_path,
        }

    try:
        active_hits = hits[:pool_limit] if pool_limit else list(hits)
        tail_hits = hits[len(active_hits):]

        vectors: dict[str, list[float]] = {}
        missing_uids: list[str] = []
        for hit in active_hits:
            chunk_uid = _chunk_uid(hit)
            vec = getattr(hit, "vector", None)
            if vec is not None:
                vectors[chunk_uid] = vec
            else:
                missing_uids.append(chunk_uid)

        if missing_uids and hasattr(store, "get_vectors"):
            vectors.update(store.get_vectors(missing_uids))

        if not vectors:
            return (hits, []) if return_deferred else hits

        protected_count = max(0, min(protected_top_k, len(active_hits)))
        selected: list[SearchHit] = list(active_hits[:protected_count])
        deferred: list[SearchHit] = []
        deferred_records: list[dict] = []

        for hit in active_hits[protected_count:]:
            chunk_uid = _chunk_uid(hit)
            hit_vec = vectors.get(chunk_uid)

            if hit_vec is None:
                selected.append(hit)
                continue

            too_similar = False
            similar_to: SearchHit | None = None
            similarity = 0.0
            for sel_hit in selected:
                sel_uid = _chunk_uid(sel_hit)
                sel_vec = vectors.get(sel_uid)
                if sel_vec is not None:
                    sim = _cosine_similarity(hit_vec, sel_vec)
                    if sim > similarity_threshold:
                        too_similar = True
                        similar_to = sel_hit
                        similarity = sim
                        break

            if too_similar:
                deferred.append(hit)
                if similar_to is not None:
                    deferred_records.append(_deferred_record(hit, similar_to, similarity))
            else:
                selected.append(hit)

        result = selected + deferred + tail_hits
        if deferred:
            logger.info(
                "MMR diversity: %d selected, %d deferred (threshold=%.2f)",
                len(selected), len(deferred), similarity_threshold,
            )
        return (result, deferred_records) if return_deferred else result

    except Exception as e:
        logger.warning("MMR diversity failed (using original order): %s", e)
        return (hits, []) if return_deferred else hits


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
    source_name: str | None = None,
    tags: str | None = None,
    status: str | None = None,
    folder: str | None = None,
    reranker: Reranker | None = None,
    prefer_recent: bool = False,
    recency_half_life_days: float = 90.0,
    recency_weight: float = 0.3,
    metadata_filters: dict[str, str] | None = None,
    filter_ast: dict | None = None,
    enr_doc_type: str | None = None,
    enr_topics: str | None = None,
    importance_field: str = "enr_importance",
    importance_weight: float = 0.3,
    min_score_threshold: float = 0.0,
    media_intent_weight: float = 0.35,
    media_intent_slots: int = 2,
) -> SearchResult:
    """Run vector + keyword search in parallel, fuse with RRF, optionally re-rank.

    Pipeline:
      1. Pre-filter (SQL WHERE on LanceDB)
      2. Parallel vector + keyword retrieval
      3. RRF fusion
      4. Length normalization
      5. Importance weighting (boosts docs with high importance/priority metadata)
      6. Optional recency boost + time decay (with floor)
      7. Optional cross-encoder re-rank (60/40 blend, cosine fallback)
      8. MMR diversity (removes near-duplicate chunks)
      9. Minimum score threshold (discards noise)
      10. Final top_k

    Returns a SearchResult (list-compatible) with a diagnostics dict.
    """
    with _tracer.start_as_current_span(
        "search.hybrid", attributes={"top_k": final_top_k}
    ):
        search_started = time.perf_counter()
        timing_ms = {
            "total": 0.0,
            "embed": 0.0,
            "vector": 0.0,
            "keyword": 0.0,
            "fusion": 0.0,
            "rerank": 0.0,
            "mmr": 0.0,
        }
        candidate_counts = {
            "vector": 0,
            "keyword": 0,
            "fused": 0,
            "returned": 0,
        }
        if not query.strip():
            timing_ms["total"] = round((time.perf_counter() - search_started) * 1000, 3)
            diagnostics = {
                "vector_search_active": True,
                "keyword_search_active": True,
                "reranker_applied": False,
                "degraded": False,
                "mmr_deferred_count": 0,
                "similar_deferred_results": [],
                "timing_ms": timing_ms,
                "candidate_counts": candidate_counts,
            }
            return SearchResult(hits=[], diagnostics=diagnostics)

        diagnostics = {
            "vector_search_active": True,
            "keyword_search_active": True,
            "reranker_applied": False,
            "degraded": False,
            "mmr_deferred_count": 0,
            "similar_deferred_results": [],
            "timing_ms": timing_ms,
            "candidate_counts": candidate_counts,
        }

        # 0. Build WHERE clause for pre-filtering (applied inside LanceDB before scoring)
        _where_kwargs = dict(
            doc_id_prefix=doc_id_prefix,
            source_type=source_type,
            source_name=source_name,
            status=status,
            folder=folder,
            tags=tags,
            enr_doc_type=enr_doc_type,
            enr_topics=enr_topics,
            metadata_filters=metadata_filters,
            filter_ast=filter_ast,
        )
        where = store._build_where_clause(**_where_kwargs)

        # 1. Embed query. If this fails, keep FTS/BM25 available instead of aborting.
        query_vector: list[float] | None = None
        vector_error: Exception | None = None
        keyword_error: Exception | None = None
        vector_hits: list[SearchHit] = []
        keyword_hits: list[SearchHit] = []

        try:
            stage_started = time.perf_counter()
            query_vector = embed_provider.embed_query(query)
            timing_ms["embed"] = round((time.perf_counter() - stage_started) * 1000, 3)
        except Exception as e:
            vector_error = e
            diagnostics["vector_search_active"] = False
            logger.warning("Query embedding failed (degraded to keyword-only): %s", e)
            try:
                keyword_hits = store.keyword_search(query, keyword_top_k, where)
            except Exception as kw_exc:
                keyword_error = kw_exc
        else:
            # 2. Parallel retrieval: vector (semantic) + keyword (BM25/FTS)
            def _timed_vector_search():
                started = time.perf_counter()
                return store.vector_search(query_vector, vector_top_k, where), round(
                    (time.perf_counter() - started) * 1000, 3
                )

            def _timed_keyword_search():
                started = time.perf_counter()
                return store.keyword_search(query, keyword_top_k, where), round(
                    (time.perf_counter() - started) * 1000, 3
                )

            with ThreadPoolExecutor(max_workers=2) as executor:
                vec_future = executor.submit(_timed_vector_search)
                kw_future = executor.submit(_timed_keyword_search)
                try:
                    vector_hits, timing_ms["vector"] = vec_future.result()
                except Exception as e:
                    vector_error = e
                try:
                    keyword_hits, timing_ms["keyword"] = kw_future.result()
                except Exception as e:
                    keyword_error = e

        # LanceDB can intermittently fail one side of concurrent retrieval even
        # though the underlying index is healthy. Retry once serially before
        # declaring a degraded search path.
        if query_vector is not None and vector_error is not None:
            try:
                stage_started = time.perf_counter()
                vector_hits = store.vector_search(query_vector, vector_top_k, where)
                timing_ms["vector"] += round((time.perf_counter() - stage_started) * 1000, 3)
                logger.info("Vector search recovered on retry after transient failure: %s", vector_error)
            except Exception as retry_error:
                logger.warning("Vector search failed (degraded to keyword-only): %s", retry_error)
                diagnostics["vector_search_active"] = False

        if keyword_error is not None:
            stage_started = time.perf_counter()
            keyword_hits, retry_error = _retry_keyword_search(
                store,
                query,
                keyword_top_k,
                where,
                keyword_error,
            )
            timing_ms["keyword"] += round((time.perf_counter() - stage_started) * 1000, 3)
            if retry_error is None:
                logger.info("Keyword/FTS search recovered on retry after transient failure: %s", keyword_error)
            else:
                logger.warning("Keyword/FTS search failed (degraded to vector-only): %s", retry_error)
                diagnostics["keyword_search_active"] = False

        candidate_counts["vector"] = len(vector_hits)
        candidate_counts["keyword"] = len(keyword_hits)

        logger.info(
            "Retrieval: %d vector hits, %d keyword hits (fts_ok=%s, where=%s)",
            len(vector_hits),
            len(keyword_hits),
            diagnostics["keyword_search_active"],
            where,
        )

        # 3. Reciprocal Rank Fusion
        stage_started = time.perf_counter()
        fused = reciprocal_rank_fusion([vector_hits, keyword_hits], k=rrf_k)

        # 4. Length normalization (prevents long keyword-rich chunks from dominating)
        fused = _apply_length_normalization(fused)

        # 5. Importance weighting (boost high-priority docs via metadata field)
        fused = _apply_importance_weighting(fused, field=importance_field, weight=importance_weight)

        # 6. Optional recency boost + time decay (applied after fusion so boost is uniform)
        if prefer_recent:
            fused = _apply_recency_boost(fused, recency_half_life_days, recency_weight)
        timing_ms["fusion"] = round((time.perf_counter() - stage_started) * 1000, 3)
        candidate_counts["fused"] = len(fused)

        # 6a. Media-intent RECALL. The source_type filter is pushed down into
        #     retrieval, so filtering changes what is fetched, not just what is
        #     kept: unfiltered, the top-50 vector + top-50 keyword rows can be
        #     entirely messages and no video/audio is retrieved at all. Measured
        #     on prod — three attachments that rank #1 *within their own medium*
        #     (1.127 / 1.166 / 0.871) were absent from the unfiltered candidate
        #     set entirely, so the quota below had nothing to inject.
        #
        #     When the query names a medium and nothing of that type was
        #     retrieved, run one targeted retrieval per missing type and add the
        #     results to the pool. Never runs when the caller already filtered by
        #     source_type, nor when that medium is already represented.
        media_recall: list[SearchHit] = []
        _wanted_types = _media_intent_types(query) if media_intent_slots > 0 else set()
        if _wanted_types and not source_type:
            _missing = {
                st for st in _wanted_types
                if not any((h.source_type or "") == st for h in fused)
            }
            for _st in sorted(_missing):
                try:
                    _kw = dict(_where_kwargs)
                    _kw["source_type"] = _st
                    _media_where = store._build_where_clause(**_kw)
                    _limit = max(media_intent_slots * 3, 5)
                    if query_vector is not None:
                        media_recall.extend(
                            store.vector_search(query_vector, _limit, _media_where)
                        )
                    media_recall.extend(
                        store.keyword_search(query, _limit, _media_where)
                    )
                except Exception as exc:
                    logger.warning(
                        "Media-intent recall for source_type=%s failed: %s", _st, exc
                    )
            if media_recall:
                _seen_keys = {(h.doc_id, h.loc) for h in fused}
                _deduped: list[SearchHit] = []
                for _hit in sorted(media_recall, key=lambda h: -h.score):
                    _key = (_hit.doc_id, _hit.loc)
                    if _key not in _seen_keys:
                        _seen_keys.add(_key)
                        _deduped.append(_hit)
                media_recall = _deduped
                diagnostics["media_intent_recall"] = len(media_recall)

        # 6b. Media-intent boost — applied BEFORE the re-rank pool is truncated.
        #     The re-ranker only sees fused[:final_top_k*6], so an attachment
        #     that starts below that cut is discarded before any later boost can
        #     help it. Measured: for a busy address ("163 Washington") dozens of
        #     messages outrank the video in fusion, so it never entered the pool
        #     and a post-rerank boost was a no-op.
        fused = _apply_media_intent_boost(fused, query, weight=media_intent_weight)

        # Snapshot the FULL fused candidate set for the media-intent quota
        # (step 11b). Everything downstream narrows: the re-rank pool is
        # fused[:top_k*6], MMR keeps ~top_k*2, then the score threshold prunes.
        # Measured on prod: for "163 Washington video walkthrough" the right
        # video sits below fused rank 60, so a snapshot taken any later contains
        # no video at all and the quota silently has nothing to inject.
        media_intent_pool = list(fused) + media_recall

        # 7. Optional cross-encoder re-rank (on top N candidates for efficiency)
        #    On failure: cosine similarity fallback instead of giving up entirely.
        if reranker is not None:
            rerank_pool = fused[: final_top_k * 6]
            try:
                stage_started = time.perf_counter()
                fused = reranker.rerank(query, rerank_pool)
                timing_ms["rerank"] = round((time.perf_counter() - stage_started) * 1000, 3)
                diagnostics["reranker_applied"] = True
            except Exception as e:
                logger.warning("Reranker failed, falling back to cosine rerank: %s", e)
                if query_vector is not None:
                    stage_started = time.perf_counter()
                    fused = _cosine_fallback_rerank(query_vector, rerank_pool, store)
                    timing_ms["rerank"] = round((time.perf_counter() - stage_started) * 1000, 3)

        # 7b. Media-intent boost — applied AFTER the re-ranker so it survives
        #     into the final cut. The re-ranker scores textual relevance and
        #     consistently prefers the message that quotes an attachment's
        #     context over the attachment itself, which is exactly the failure
        #     this corrects.
        fused = _apply_media_intent_boost(fused, query, weight=media_intent_weight)

        # 8. MMR diversity (remove near-duplicate chunks)
        stage_started = time.perf_counter()
        fused, deferred_results = _apply_mmr_diversity(
            fused,
            store,
            protected_top_k=3,
            pool_limit=max(final_top_k * 2, 3),
            return_deferred=True,
        )

        # 9. Minimum score threshold (discard noise)
        if min_score_threshold > 0.0:
            deferred_results = [
                record for record in deferred_results
                if float(record.get("score", 0.0)) >= min_score_threshold
            ]
        fused = _apply_min_score_threshold(fused, threshold=min_score_threshold)
        diagnostics["mmr_deferred_count"] = len(deferred_results)
        diagnostics["similar_deferred_results"] = deferred_results
        timing_ms["mmr"] = round((time.perf_counter() - stage_started) * 1000, 3)

        # 10. Compute degraded flag
        diagnostics["degraded"] = (
            not diagnostics["vector_search_active"]
            or not diagnostics["keyword_search_active"]
            or (reranker is not None and not diagnostics["reranker_applied"])
        )

        # 11. Final top_k
        hits = fused[:final_top_k]

        # 11b. Media-intent quota — a query that names a medium must return some
        #      of that medium even when messages sweep every slot on score.
        if media_intent_slots > 0:
            hits = _ensure_media_intent_slots(
                hits, media_intent_pool, query, min_slots=media_intent_slots
            )
        candidate_counts["returned"] = len(hits)
        timing_ms["total"] = round((time.perf_counter() - search_started) * 1000, 3)
        return SearchResult(hits=hits, diagnostics=diagnostics)
