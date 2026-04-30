"""Tests for hybrid search: parallel vector + keyword (BM25/FTS), RRF fusion, re-ranker."""

import os
import tempfile

import pytest

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from lancedb_store import LanceDBStore
from search_hybrid import hybrid_search, reciprocal_rank_fusion, Reranker, SearchResult, _apply_recency_boost
from core.storage import SearchHit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockEmbedProvider:
    """Returns a fixed vector for any query. Tests search logic without API calls."""

    def __init__(self, vector: list[float]):
        self._vector = vector

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vector for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vector


def _make_node(doc_id, loc, text, vector, source_type="md", **extra_meta):
    meta = {
        "doc_id": doc_id,
        "source_type": source_type,
        "loc": loc,
        "snippet": text[:200],
        "mtime": 1.0,
        "size": len(text),
    }
    meta.update(extra_meta)
    node = TextNode(
        text=text,
        id_=f"{doc_id}::{loc}",
        embedding=vector,
        metadata=meta,
    )
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
    return node


def _build_store_with_fts(tmpdir, nodes):
    """Create a LanceDBStore, upsert nodes, and build the FTS index."""
    store = LanceDBStore(tmpdir, "test_chunks")
    store.upsert_nodes(nodes)
    store.create_fts_index()
    return store


# ---------------------------------------------------------------------------
# RRF unit tests
# ---------------------------------------------------------------------------

def test_rrf_single_list():
    """RRF with a single list should preserve order."""
    hits = [
        SearchHit(doc_id="a.md", loc="c:0", snippet="a", text="alpha", score=1.0),
        SearchHit(doc_id="b.md", loc="c:0", snippet="b", text="beta", score=0.5),
    ]
    fused = reciprocal_rank_fusion([hits], k=60)
    assert len(fused) == 2
    assert fused[0].doc_id == "a.md"
    assert fused[1].doc_id == "b.md"
    assert fused[0].score > fused[1].score


def test_rrf_two_lists_overlap():
    """Doc appearing in both lists should get a higher RRF score."""
    list_a = [
        SearchHit(doc_id="a.md", loc="c:0", snippet="a", text="alpha", score=1.0),
        SearchHit(doc_id="b.md", loc="c:0", snippet="b", text="beta", score=0.5),
    ]
    list_b = [
        SearchHit(doc_id="a.md", loc="c:0", snippet="a", text="alpha", score=1.0),
        SearchHit(doc_id="c.md", loc="c:0", snippet="c", text="gamma", score=0.5),
    ]
    fused = reciprocal_rank_fusion([list_a, list_b], k=60)
    # "a.md" appears in both → highest RRF score
    assert fused[0].doc_id == "a.md"
    # Should have 3 unique docs
    assert len(fused) == 3


def test_rrf_disjoint_lists():
    """Disjoint lists: all docs should appear, tied RRF for same rank."""
    list_a = [SearchHit(doc_id="a.md", loc="c:0", snippet="a", text="a", score=1.0)]
    list_b = [SearchHit(doc_id="b.md", loc="c:0", snippet="b", text="b", score=1.0)]
    fused = reciprocal_rank_fusion([list_a, list_b], k=60)
    assert len(fused) == 2
    # Both at rank 0 in their lists → same RRF score
    assert fused[0].score == fused[1].score


def test_rrf_empty_lists():
    """RRF with empty lists should return empty."""
    fused = reciprocal_rank_fusion([[], []])
    assert fused == []


# ---------------------------------------------------------------------------
# Hybrid search end-to-end (with real LanceDBStore + FTS)
# ---------------------------------------------------------------------------

def test_hybrid_returns_results():
    """Basic hybrid search should return relevant results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vec = [1.0] + [0.0] * 767
        nodes = [
            _make_node("a.md", "c:0", "roof insurance claim damage", vec),
            _make_node("b.md", "c:0", "kimchi recipe fermentation", [0.0] + [1.0] + [0.0] * 766),
        ]
        store = _build_store_with_fts(tmpdir, nodes)
        embed = MockEmbedProvider(vec)

        hits = hybrid_search(store, embed, "roof claim", vector_top_k=10, final_top_k=5)
        assert len(hits) >= 1
        assert hits[0].doc_id == "a.md"


def test_hybrid_keyword_boost():
    """A doc matching keywords but not closest vector should still rank high via RRF."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Both docs have the same vector (so vector search ranks them equally)
        vec = [1.0] + [0.0] * 767
        nodes = [
            _make_node("no_keyword.md", "c:0", "generic text about nothing specific", vec),
            _make_node("has_keyword.md", "c:0", "specific roof insurance claim damage report", vec),
        ]
        store = _build_store_with_fts(tmpdir, nodes)
        embed = MockEmbedProvider(vec)

        hits = hybrid_search(store, embed, "roof insurance claim", vector_top_k=10, final_top_k=5)
        assert len(hits) >= 1
        # "has_keyword.md" should rank higher because FTS matches keywords
        keyword_hit = [h for h in hits if h.doc_id == "has_keyword.md"]
        assert len(keyword_hit) == 1


def test_hybrid_empty_query():
    """Empty query should return empty results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        embed = MockEmbedProvider([0.0] * 768)
        hits = hybrid_search(store, embed, "", vector_top_k=10, final_top_k=5)
        assert hits == []


def test_hybrid_doc_id_prefix_filter():
    """doc_id_prefix filter should restrict results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vec = [1.0] + [0.0] * 767
        nodes = [
            _make_node("claims/a.md", "c:0", "roof damage report", vec),
            _make_node("recipes/b.md", "c:0", "roof shingle cake topping", vec),
        ]
        store = _build_store_with_fts(tmpdir, nodes)
        embed = MockEmbedProvider(vec)

        hits = hybrid_search(
            store, embed, "roof", vector_top_k=10, final_top_k=5, doc_id_prefix="claims/"
        )
        doc_ids = [h.doc_id for h in hits]
        assert all(d.startswith("claims/") for d in doc_ids)


def test_hybrid_source_type_filter():
    """source_type filter should restrict results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vec = [1.0] + [0.0] * 767
        nodes = [
            _make_node("a.md", "c:0", "markdown document about cats", vec, source_type="md"),
            _make_node("b.pdf", "p:1:c:0", "pdf document about cats", vec, source_type="pdf"),
        ]
        store = _build_store_with_fts(tmpdir, nodes)
        embed = MockEmbedProvider(vec)

        hits = hybrid_search(
            store, embed, "cats", vector_top_k=10, final_top_k=5, source_type="pdf"
        )
        assert all(h.source_type == "pdf" for h in hits)
        assert len(hits) == 1


# ---------------------------------------------------------------------------
# FTS-specific tests
# ---------------------------------------------------------------------------

def test_keyword_search_returns_hits():
    """keyword_search should find docs by text content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vec = [0.0] * 768
        nodes = [
            _make_node("a.md", "c:0", "kimchi fermentation recipe traditional", vec),
            _make_node("b.md", "c:0", "insurance claim roof damage report", vec),
        ]
        store = _build_store_with_fts(tmpdir, nodes)
        hits = store.keyword_search("kimchi fermentation", top_k=5)
        assert len(hits) >= 1
        assert any(h.doc_id == "a.md" for h in hits)


def test_keyword_search_no_fts_index():
    """keyword_search without FTS index should raise (hybrid_search handles degradation)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vec = [0.0] * 768
        nodes = [_make_node("a.md", "c:0", "some text", vec)]
        store = LanceDBStore(tmpdir, "test_chunks")
        store.upsert_nodes(nodes)
        # Don't create FTS index — keyword_search should raise
        with pytest.raises(Exception):
            store.keyword_search("some text", top_k=5)


def test_keyword_search_empty_query():
    """Empty query should return empty results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        hits = store.keyword_search("", top_k=5)
        assert hits == []


# ---------------------------------------------------------------------------
# Reranker tests
# ---------------------------------------------------------------------------

class MockReranker(Reranker):
    """Mock reranker that reverses the hit order (for testing)."""

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        for i, h in enumerate(reversed(hits)):
            h.score = 1.0 - i * 0.1
        return list(reversed(hits))


def test_hybrid_with_reranker():
    """Hybrid search with a reranker should apply re-ranking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vec = [1.0] + [0.0] * 767
        nodes = [
            _make_node("a.md", "c:0", "first document content", vec),
            _make_node("b.md", "c:0", "second document content", vec),
        ]
        store = _build_store_with_fts(tmpdir, nodes)
        embed = MockEmbedProvider(vec)

        reranker = MockReranker()
        hits = hybrid_search(
            store, embed, "document content",
            vector_top_k=10, final_top_k=5, reranker=reranker,
        )
        assert len(hits) >= 1


# ---------------------------------------------------------------------------
# Recency boost tests
# ---------------------------------------------------------------------------

def test_recency_boost_recent_wins():
    """A recent doc should score higher than an old doc after recency boost."""
    import time
    now = time.time()
    old = SearchHit(doc_id="old.md", loc="c:0", snippet="x", text="x", score=1.0, mtime=now - 365 * 86400)
    new = SearchHit(doc_id="new.md", loc="c:0", snippet="x", text="x", score=1.0, mtime=now - 1 * 86400)
    # Both start with equal score; after boost, new should win
    boosted = _apply_recency_boost([old, new], half_life_days=90, weight=0.3)
    assert boosted[0].doc_id == "new.md"
    assert boosted[0].score > boosted[1].score


def test_recency_boost_preserves_relevance():
    """A highly relevant old doc should still beat a less relevant new doc."""
    import time
    now = time.time()
    old_relevant = SearchHit(doc_id="old.md", loc="c:0", snippet="x", text="x", score=2.0, mtime=now - 365 * 86400)
    new_weak = SearchHit(doc_id="new.md", loc="c:0", snippet="x", text="x", score=0.5, mtime=now - 1 * 86400)
    boosted = _apply_recency_boost([old_relevant, new_weak], half_life_days=90, weight=0.3)
    assert boosted[0].doc_id == "old.md"


def test_recency_boost_zero_mtime_no_crash():
    """Hits with mtime=0 (missing) should not crash."""
    hit = SearchHit(doc_id="a.md", loc="c:0", snippet="x", text="x", score=1.0, mtime=0.0)
    boosted = _apply_recency_boost([hit], half_life_days=90, weight=0.3)
    assert len(boosted) == 1
    assert boosted[0].score == 1.0  # no boost applied (mtime is 0)


def test_hybrid_with_prefer_recent():
    """hybrid_search with prefer_recent=True should boost recent docs."""
    import time
    with tempfile.TemporaryDirectory() as tmpdir:
        vec = [1.0] + [0.0] * 767
        now = time.time()
        old_node = _make_node("old.md", "c:0", "matching search content", vec)
        old_node.metadata["mtime"] = now - 365 * 86400
        new_node = _make_node("new.md", "c:0", "matching search content", vec)
        new_node.metadata["mtime"] = now - 1 * 86400

        store = _build_store_with_fts(tmpdir, [old_node, new_node])
        embed = MockEmbedProvider(vec)

        result = hybrid_search(
            store, embed, "matching search content",
            vector_top_k=10, final_top_k=5, prefer_recent=True,
        )
        assert len(result) == 2
        assert result[0].doc_id == "new.md"


# ---------------------------------------------------------------------------
# Diagnostics tests
# ---------------------------------------------------------------------------

def test_diagnostics_normal_search():
    """Normal hybrid search should report all-healthy diagnostics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vec = [1.0] + [0.0] * 767
        nodes = [_make_node("a.md", "c:0", "roof insurance claim damage", vec)]
        store = _build_store_with_fts(tmpdir, nodes)
        embed = MockEmbedProvider(vec)

        result = hybrid_search(store, embed, "roof claim", vector_top_k=10, final_top_k=5)
        assert isinstance(result, SearchResult)
        assert result.diagnostics["vector_search_active"] is True
        assert result.diagnostics["keyword_search_active"] is True
        assert result.diagnostics["reranker_applied"] is False  # no reranker configured
        assert result.diagnostics["degraded"] is False


def test_diagnostics_keyword_failure():
    """When FTS index is missing, keyword_search_active should be False and degraded True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vec = [1.0] + [0.0] * 767
        nodes = [_make_node("a.md", "c:0", "roof insurance claim damage", vec)]
        store = LanceDBStore(tmpdir, "test_chunks")
        store.upsert_nodes(nodes)
        # Don't create FTS index — keyword_search will raise, hybrid_search handles it
        embed = MockEmbedProvider(vec)

        result = hybrid_search(store, embed, "roof claim", vector_top_k=10, final_top_k=5)
        assert isinstance(result, SearchResult)
        assert result.diagnostics["keyword_search_active"] is False
        assert result.diagnostics["degraded"] is True
        # Should still return vector-only results
        assert len(result) >= 1


def test_diagnostics_keyword_failure_recovers_on_retry(monkeypatch):
    """Transient keyword search failures should retry once before degrading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vec = [1.0] + [0.0] * 767
        nodes = [_make_node("a.md", "c:0", "roof insurance claim damage", vec)]
        store = _build_store_with_fts(tmpdir, nodes)
        embed = MockEmbedProvider(vec)

        original_keyword_search = store.keyword_search
        calls = {"count": 0}

        def flaky_keyword_search(query: str, top_k: int = 50, where: str | None = None):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError(
                    "Added column's length must match table's length. Expected length 49 but got length 50"
                )
            return original_keyword_search(query, top_k=top_k, where=where)

        monkeypatch.setattr(store, "keyword_search", flaky_keyword_search)

        result = hybrid_search(store, embed, "roof claim", vector_top_k=10, final_top_k=5)
        assert isinstance(result, SearchResult)
        assert result.diagnostics["keyword_search_active"] is True
        assert result.diagnostics["degraded"] is False
        assert calls["count"] == 2
        assert len(result) >= 1


def test_diagnostics_keyword_failure_recovers_after_rebuild_window(monkeypatch):
    """Repeated transient FTS shape errors should back off and retry until rebuild finishes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vec = [1.0] + [0.0] * 767
        nodes = [_make_node("a.md", "c:0", "roof insurance claim damage", vec)]
        store = _build_store_with_fts(tmpdir, nodes)
        embed = MockEmbedProvider(vec)

        original_keyword_search = store.keyword_search
        calls = {"count": 0}
        sleep_calls: list[float] = []

        def flaky_keyword_search(query: str, top_k: int = 50, where: str | None = None):
            calls["count"] += 1
            if calls["count"] < 3:
                raise RuntimeError(
                    "Added column's length must match table's length. Expected length 49 but got length 50"
                )
            return original_keyword_search(query, top_k=top_k, where=where)

        monkeypatch.setattr(store, "keyword_search", flaky_keyword_search)
        monkeypatch.setattr(
            hybrid_search.__globals__["time"],
            "sleep",
            lambda seconds: sleep_calls.append(seconds),
        )

        result = hybrid_search(store, embed, "roof claim", vector_top_k=10, final_top_k=5)
        assert isinstance(result, SearchResult)
        assert result.diagnostics["keyword_search_active"] is True
        assert result.diagnostics["degraded"] is False
        assert calls["count"] == 3
        assert sleep_calls == [0.25, 0.5]
        assert len(result) >= 1


class FailingReranker(Reranker):
    """Reranker that always raises (simulates server down)."""
    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        raise RuntimeError("Reranker server unavailable")


def test_diagnostics_reranker_failure():
    """When reranker fails, reranker_applied should be False and degraded True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vec = [1.0] + [0.0] * 767
        nodes = [_make_node("a.md", "c:0", "roof insurance claim damage", vec)]
        store = _build_store_with_fts(tmpdir, nodes)
        embed = MockEmbedProvider(vec)

        result = hybrid_search(
            store, embed, "roof claim",
            vector_top_k=10, final_top_k=5, reranker=FailingReranker(),
        )
        assert isinstance(result, SearchResult)
        assert result.diagnostics["keyword_search_active"] is True
        assert result.diagnostics["reranker_applied"] is False
        assert result.diagnostics["degraded"] is True
        # Should still return RRF-fused results
        assert len(result) >= 1


def test_diagnostics_reranker_success():
    """When reranker succeeds, reranker_applied should be True and degraded False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vec = [1.0] + [0.0] * 767
        nodes = [_make_node("a.md", "c:0", "roof insurance claim damage", vec)]
        store = _build_store_with_fts(tmpdir, nodes)
        embed = MockEmbedProvider(vec)

        result = hybrid_search(
            store, embed, "roof claim",
            vector_top_k=10, final_top_k=5, reranker=MockReranker(),
        )
        assert isinstance(result, SearchResult)
        assert result.diagnostics["reranker_applied"] is True
        assert result.diagnostics["degraded"] is False


def test_diagnostics_empty_query():
    """Empty query should return SearchResult with empty hits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        embed = MockEmbedProvider([0.0] * 768)
        result = hybrid_search(store, embed, "", vector_top_k=10, final_top_k=5)
        assert isinstance(result, SearchResult)
        assert result == []
        assert len(result) == 0


def test_diagnostics_both_fail():
    """When both keyword and reranker fail, degraded should be True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vec = [1.0] + [0.0] * 767
        nodes = [_make_node("a.md", "c:0", "roof insurance claim damage", vec)]
        store = LanceDBStore(tmpdir, "test_chunks")
        store.upsert_nodes(nodes)
        # No FTS index, and a failing reranker
        embed = MockEmbedProvider(vec)

        result = hybrid_search(
            store, embed, "roof claim",
            vector_top_k=10, final_top_k=5, reranker=FailingReranker(),
        )
        assert result.diagnostics["vector_search_active"] is True
        assert result.diagnostics["keyword_search_active"] is False
        assert result.diagnostics["reranker_applied"] is False
        assert result.diagnostics["degraded"] is True


# -----------------------------------------------------------------------
# Live integration tests — real OpenRouter embedding + DeepInfra reranker
# -----------------------------------------------------------------------

_has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
_has_deepinfra = bool(os.environ.get("DEEPINFRA_API_KEY"))


@pytest.mark.live
@pytest.mark.skipif(not _has_openrouter, reason="OPENROUTER_API_KEY not set")
class TestOpenRouterEmbeddingLive:
    """Live tests for OpenRouter embedding (Qwen3-Embedding-8B)."""

    def test_embed_produces_vectors(self):
        """Real embedding returns non-zero vectors."""
        from providers.embed.openrouter_embed import OpenRouterEmbedProvider

        provider = OpenRouterEmbedProvider(model="qwen/qwen3-embedding-8b")
        vectors = provider.embed_texts(["hello world"])
        assert len(vectors) == 1
        assert len(vectors[0]) > 0
        assert any(v != 0.0 for v in vectors[0])

    def test_embed_query_vs_document_differ(self):
        """Query embedding (with instruction prefix) should differ from document embedding."""
        from providers.embed.openrouter_embed import OpenRouterEmbedProvider

        provider = OpenRouterEmbedProvider(model="qwen/qwen3-embedding-8b")
        doc_vec = provider.embed_texts(["kimchi recipe"])[0]
        query_vec = provider.embed_query("kimchi recipe")
        assert doc_vec != query_vec


@pytest.mark.live
@pytest.mark.skipif(not _has_deepinfra, reason="DEEPINFRA_API_KEY not set")
class TestDeepInfraRerankerLive:
    """Live tests for DeepInfra-hosted Qwen3-Reranker."""

    def test_reranker_scores_and_sorts(self):
        """Real reranker should assign higher scores to relevant docs."""
        from core.config import load_config
        from search_hybrid import build_reranker

        config = load_config()
        reranker = build_reranker(config)
        assert reranker is not None, "Reranker not configured"

        hits = [
            SearchHit(
                doc_id="relevant.md", loc="c:0", snippet="kimchi recipe",
                text="Traditional Korean kimchi fermentation recipe with napa cabbage",
                score=0.5,
            ),
            SearchHit(
                doc_id="irrelevant.md", loc="c:0", snippet="tax forms",
                text="IRS Form 1040 instructions for federal income tax filing",
                score=0.5,
            ),
        ]
        reranked = reranker.rerank("kimchi fermentation recipe", hits)
        assert len(reranked) == 2
        assert reranked[0].doc_id == "relevant.md", (
            f"Expected relevant.md first, got {reranked[0].doc_id}"
        )
        assert reranked[0].score > reranked[1].score


# ---------------------------------------------------------------------------
# Pre-filtered search E2E tests
# ---------------------------------------------------------------------------

def test_prefilter_source_type():
    """Pre-filter by source_type should return full top_k of matching docs only."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vec = [1.0] + [0.0] * 767
        nodes = [
            _make_node("a.md", "c:0", "important document text", vec, source_type="md"),
            _make_node("b.pdf", "p:1:c:0", "important document text pdf", vec, source_type="pdf"),
            _make_node("c.pdf", "p:1:c:0", "important document text pdf two", vec, source_type="pdf"),
        ]
        store = _build_store_with_fts(tmpdir, nodes)
        embed = MockEmbedProvider(vec)

        result = hybrid_search(
            store, embed, "important document",
            vector_top_k=10, final_top_k=10, source_type="pdf",
        )
        assert all(h.source_type == "pdf" for h in result)
        assert len(result) == 2


def test_prefilter_enr_doc_type():
    """Pre-filter by enr_doc_type should only return matching enriched docs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vec = [1.0] + [0.0] * 767
        nodes = [
            _make_node("a.md", "c:0", "soil analysis geotechnical", vec,
                       enr_doc_type="Geotechnical Report"),
            _make_node("b.md", "c:0", "quarterly earnings financial", vec,
                       enr_doc_type="Financial Report"),
            _make_node("c.md", "c:0", "soil testing lab results", vec,
                       enr_doc_type="Geotechnical Report"),
        ]
        store = _build_store_with_fts(tmpdir, nodes)
        embed = MockEmbedProvider(vec)

        result = hybrid_search(
            store, embed, "soil analysis",
            vector_top_k=10, final_top_k=10, enr_doc_type="Geotechnical Report",
        )
        assert len(result) >= 1
        for h in result:
            assert "Geotechnical Report" in (h.enr_doc_type or "")


def test_prefilter_enr_topics():
    """Pre-filter by enr_topics should only return matching enriched docs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vec = [1.0] + [0.0] * 767
        nodes = [
            _make_node("a.md", "c:0", "neural network deep learning", vec,
                       enr_topics="machine learning, deep learning"),
            _make_node("b.md", "c:0", "kimchi fermentation recipe", vec,
                       enr_topics="korean cooking, fermentation"),
        ]
        store = _build_store_with_fts(tmpdir, nodes)
        embed = MockEmbedProvider(vec)

        result = hybrid_search(
            store, embed, "deep learning neural",
            vector_top_k=10, final_top_k=10, enr_topics="machine learning",
        )
        assert len(result) >= 1
        for h in result:
            assert "machine learning" in (h.enr_topics or "")


def test_prefilter_combined_source_and_folder():
    """Combined source_type + folder pre-filters should both apply."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vec = [1.0] + [0.0] * 767
        nodes = [
            _make_node("a.md", "c:0", "test content", vec, source_type="md", folder="Archive"),
            _make_node("b.pdf", "p:1:c:0", "test content pdf", vec, source_type="pdf", folder="Archive"),
            _make_node("c.pdf", "p:1:c:0", "test content pdf proj", vec, source_type="pdf", folder="Projects"),
        ]
        store = _build_store_with_fts(tmpdir, nodes)
        embed = MockEmbedProvider(vec)

        result = hybrid_search(
            store, embed, "test content",
            vector_top_k=10, final_top_k=10, source_type="pdf", folder="Archive",
        )
        assert len(result) == 1
        assert result[0].doc_id == "b.pdf"


def test_diagnostics_vector_search_failure():
    """When vector search fails, vector_search_active should be False and degraded True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vec = [1.0] + [0.0] * 767
        nodes = [_make_node("a.md", "c:0", "roof insurance claim damage", vec)]
        store = _build_store_with_fts(tmpdir, nodes)
        embed = MockEmbedProvider(vec)

        # Monkey-patch vector_search to raise
        original = store.vector_search
        def failing_vector_search(*args, **kwargs):
            raise RuntimeError("LanceDB corrupted")
        store.vector_search = failing_vector_search

        result = hybrid_search(store, embed, "roof claim", vector_top_k=10, final_top_k=5)
        assert isinstance(result, SearchResult)
        assert result.diagnostics["vector_search_active"] is False
        assert result.diagnostics["keyword_search_active"] is True
        assert result.diagnostics["degraded"] is True
        # Should still return keyword-only results
        assert len(result) >= 1
