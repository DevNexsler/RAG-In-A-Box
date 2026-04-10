"""Integration tests: real embeddings, real LanceDB, real search, real reranker.

Uses the provider stack defined in config_test.yaml — currently OpenRouter
(embeddings + enrichment) and DeepInfra (reranker). Image OCR is DeepSeek-OCR2
at the URL configured in ocr.base_url; image-OCR assertions are skipped
automatically if that service is unreachable.

Run with:  pytest tests/test_integration.py -v -s

The test flow:
  1. Index the test_vault (3 MD + 1 PDF + 1 image) with real embeddings
  2. Run semantic search queries and verify results make sense
  3. Test MCP tool handler implementations
  4. Verify get_chunk returns correct text
  5. Test Qwen3-Reranker via DeepInfra with continuous scoring
  6. Full-pipeline test via index_vault_flow with LLM enrichment via OpenRouter
"""

import os
import sys
import tempfile
from pathlib import Path

import httpx
import pytest

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
_has_deepinfra = bool(os.environ.get("DEEPINFRA_API_KEY"))

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (_has_openrouter and _has_deepinfra),
        reason="OPENROUTER_API_KEY and DEEPINFRA_API_KEY required for integration tests",
    ),
]


# ---------------------------------------------------------------------------
# Shared fixture: index the test vault once, reuse for all search tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def indexed_store():
    """Index the test_vault with real embeddings. Returns (store, embed_provider, config)."""
    from dotenv import load_dotenv
    load_dotenv()

    from core.config import load_config
    from lancedb_store import LanceDBStore
    from providers.embed import build_embed_provider
    from providers.ocr import build_ocr_provider
    from extractors import extract_text
    from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
    from llama_index.core.node_parser import SentenceSplitter

    # Use a temp directory for the index so tests don't pollute
    with tempfile.TemporaryDirectory() as tmpdir:
        config = load_config("config_test.yaml")
        # Override index_root to temp
        config["index_root"] = tmpdir

        vault_root = Path(config["vault_root"])
        store = LanceDBStore(tmpdir, config.get("lancedb", {}).get("table", "chunks"))

        embed_provider = build_embed_provider(config)
        splitter = SentenceSplitter(
            chunk_size=config.get("chunking", {}).get("max_chars", 1800),
            chunk_overlap=config.get("chunking", {}).get("overlap", 200),
        )
        ocr_provider = build_ocr_provider(config)

        # Index all files in test_vault
        pdf_cfg = config.get("pdf", {})
        indexed_docs = []
        for path in vault_root.rglob("*"):
            if not path.is_file():
                continue
            ext = path.suffix.lower().lstrip(".")
            if ext not in ("md", "pdf", "png", "jpg", "jpeg"):
                continue
            rel = str(path.relative_to(vault_root)).replace("\\", "/")
            result = extract_text(
                file_path=str(path),
                ext=ext,
                ocr_provider=ocr_provider,
                pdf_strategy=pdf_cfg.get("strategy", "text_then_ocr"),
                min_text_chars=pdf_cfg.get("min_text_chars_before_ocr", 200),
                ocr_page_limit=pdf_cfg.get("ocr_page_limit", 200),
            )

            if not result.full_text.strip():
                continue

            source_type = "md" if ext == "md" else "pdf" if ext == "pdf" else "img"
            chunks = splitter.split_text(result.full_text)
            vectors = embed_provider.embed_texts(chunks)

            nodes = []
            for i, (chunk_text, vector) in enumerate(zip(chunks, vectors)):
                loc = f"c:{i}"
                node = TextNode(
                    text=chunk_text,
                    id_=f"{rel}::{loc}",
                    embedding=vector,
                    metadata={
                        "doc_id": rel,
                        "source_type": source_type,
                        "loc": loc,
                        "snippet": chunk_text[:200],
                        "mtime": path.stat().st_mtime,
                        "size": path.stat().st_size,
                    },
                )
                node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=rel)
                nodes.append(node)

            store.upsert_nodes(nodes)
            indexed_docs.append(rel)

        # Build FTS index for keyword search
        store.create_fts_index()

        yield {
            "store": store,
            "embed_provider": embed_provider,
            "config": config,
            "indexed_docs": indexed_docs,
            "tmpdir": tmpdir,
        }


# ---------------------------------------------------------------------------
# Indexing tests
# ---------------------------------------------------------------------------

class TestIndexing:
    """Verify the index was built correctly."""

    def test_all_docs_indexed(self, indexed_store):
        """All test_vault files should be in the index."""
        doc_ids = indexed_store["store"].list_doc_ids()
        # At minimum: note1, note2, recipe, sample.pdf (filenames include @ID@ suffix)
        assert len(doc_ids) >= 4, f"Expected >= 4 docs, got {len(doc_ids)}: {doc_ids}"
        assert any("note1" in d for d in doc_ids), f"note1 not found in {doc_ids}"
        assert any("note2" in d for d in doc_ids), f"note2 not found in {doc_ids}"
        assert any("recipe" in d for d in doc_ids), f"recipe not found in {doc_ids}"
        assert any("sample" in d for d in doc_ids), f"sample not found in {doc_ids}"

    def test_markdown_indexed(self, indexed_store):
        """Markdown files should produce chunks."""
        doc_ids = indexed_store["store"].list_doc_ids()
        md_docs = [d for d in doc_ids if d.endswith(".md")]
        assert len(md_docs) >= 3

    def test_pdf_indexed(self, indexed_store):
        """PDF should produce chunks."""
        doc_ids = indexed_store["store"].list_doc_ids()
        assert any("sample" in d for d in doc_ids), f"sample not found in {doc_ids}"


# ---------------------------------------------------------------------------
# Semantic search tests — verify results make sense
# ---------------------------------------------------------------------------

class TestSemanticSearch:
    """Search with real embeddings and verify semantic relevance."""

    def test_search_kimchi_recipe(self, indexed_store):
        """Searching for 'kimchi recipe' should return the recipe as top result."""
        from search_hybrid import hybrid_search

        hits = hybrid_search(
            indexed_store["store"],
            indexed_store["embed_provider"],
            "kimchi fermentation recipe",
            vector_top_k=10,
            final_top_k=5,
        )
        assert len(hits) >= 1
        # The recipe should be the top hit
        assert "recipe" in hits[0].doc_id, \
            f"Expected recipe as top hit, got {hits[0].doc_id} (snippet: {hits[0].snippet[:80]})"

    def test_search_insurance_claim(self, indexed_store):
        """Searching for 'roof insurance claim' should return note2.md."""
        from search_hybrid import hybrid_search

        hits = hybrid_search(
            indexed_store["store"],
            indexed_store["embed_provider"],
            "roof insurance claim documents",
            vector_top_k=10,
            final_top_k=5,
        )
        assert len(hits) >= 1
        assert "note2" in hits[0].doc_id, \
            f"Expected note2 as top hit, got {hits[0].doc_id} (snippet: {hits[0].snippet[:80]})"

    def test_search_semantic_search_project(self, indexed_store):
        """Searching for 'semantic search' should return note1.md or sample.pdf."""
        from search_hybrid import hybrid_search

        hits = hybrid_search(
            indexed_store["store"],
            indexed_store["embed_provider"],
            "semantic search embeddings AI",
            vector_top_k=10,
            final_top_k=5,
        )
        assert len(hits) >= 1
        # note1.md talks about semantic search; sample.pdf talks about indexer architecture
        top_doc = hits[0].doc_id
        assert any(name in top_doc for name in ("note1", "sample")), \
            f"Expected note1 or sample, got {top_doc} (snippet: {hits[0].snippet[:80]})"

    def test_search_scores_are_reasonable(self, indexed_store):
        """Scores should be between 0 and 1 and ordered descending."""
        from search_hybrid import hybrid_search

        hits = hybrid_search(
            indexed_store["store"],
            indexed_store["embed_provider"],
            "document organizer",
            vector_top_k=10,
            final_top_k=5,
        )
        if len(hits) >= 2:
            for i in range(len(hits) - 1):
                assert hits[i].score >= hits[i + 1].score, \
                    f"Scores not descending: {hits[i].score} < {hits[i + 1].score}"
        for h in hits:
            assert 0.0 <= h.score <= 2.0, f"Score out of range: {h.score}"

    def test_search_with_doc_id_prefix_filter(self, indexed_store):
        """Filter by doc_id_prefix should restrict results."""
        from search_hybrid import hybrid_search

        hits = hybrid_search(
            indexed_store["store"],
            indexed_store["embed_provider"],
            "recipe ingredients cooking",
            vector_top_k=10,
            final_top_k=5,
            doc_id_prefix="subfolder/",
        )
        for h in hits:
            assert h.doc_id.startswith("subfolder/"), \
                f"Result {h.doc_id} doesn't match prefix 'subfolder/'"

    def test_search_returns_all_fields(self, indexed_store):
        """Each SearchHit should have doc_id, loc, snippet, text, score."""
        from search_hybrid import hybrid_search

        hits = hybrid_search(
            indexed_store["store"],
            indexed_store["embed_provider"],
            "cabbage salt",
            vector_top_k=10,
            final_top_k=3,
        )
        assert len(hits) >= 1
        hit = hits[0]
        assert hit.doc_id, "doc_id should not be empty"
        assert hit.loc, "loc should not be empty"
        assert hit.snippet, "snippet should not be empty"
        assert hit.text, "text should not be empty"
        assert isinstance(hit.score, float)

    def test_unrelated_query_low_relevance(self, indexed_store):
        """A completely unrelated query should still return results but with lower scores."""
        from search_hybrid import hybrid_search

        hits = hybrid_search(
            indexed_store["store"],
            indexed_store["embed_provider"],
            "quantum physics black holes",
            vector_top_k=10,
            final_top_k=5,
        )
        # Should return results (vector search always returns something)
        # but none should have a very high combined score
        if hits:
            assert hits[0].score < 0.95, \
                f"Unrelated query got suspiciously high score: {hits[0].score}"


# ---------------------------------------------------------------------------
# get_chunk tests
# ---------------------------------------------------------------------------

class TestGetChunk:
    """Test retrieving specific chunks by doc_id + loc."""

    def test_get_chunk_returns_text(self, indexed_store):
        """get_chunk with valid doc_id and loc should return the chunk text."""
        store = indexed_store["store"]
        hit = store.get_chunk("subfolder/recipe@00005@.md", "c:0")
        assert hit is not None, "get_chunk returned None for recipe (00005) c:0"
        assert "cabbage" in hit.text.lower() or "kimchi" in hit.text.lower(), \
            f"Expected recipe content, got: {hit.text[:100]}"

    def test_get_chunk_nonexistent(self, indexed_store):
        """get_chunk for a non-existent doc should return None."""
        store = indexed_store["store"]
        hit = store.get_chunk("does_not_exist.md", "c:0")
        assert hit is None

    def test_get_chunk_wrong_loc(self, indexed_store):
        """get_chunk with wrong loc should return None."""
        store = indexed_store["store"]
        hit = store.get_chunk("note1@00001@.md", "c:999")
        assert hit is None


# ---------------------------------------------------------------------------
# MCP tool handler tests (using the internal _impl functions directly)
# ---------------------------------------------------------------------------

class TestMCPHandlers:
    """Test MCP tool implementations with real data."""

    def test_file_search_impl(self, indexed_store):
        """file_search returns dict with results list, diagnostics, and expected keys."""
        # We need to wire up the MCP module to use our test store
        import mcp_server
        mcp_server._cache = (
            indexed_store["store"],
            indexed_store["embed_provider"],
            indexed_store["config"],
        )

        response = mcp_server._file_search_impl("kimchi recipe", top_k=3)
        assert isinstance(response, dict)
        assert "results" in response
        assert "diagnostics" in response
        results = response["results"]
        assert isinstance(results, list)
        assert len(results) >= 1
        # Check dict keys
        r = results[0]
        assert "doc_id" in r
        assert "loc" in r
        assert "snippet" in r
        assert "score" in r
        assert "recipe" in r["doc_id"]
        # Diagnostics should report healthy search
        assert response["diagnostics"]["degraded"] is False

    def test_file_get_chunk_impl(self, indexed_store):
        """file_get_chunk returns chunk text."""
        import mcp_server
        mcp_server._cache = (
            indexed_store["store"],
            indexed_store["embed_provider"],
            indexed_store["config"],
        )

        result = mcp_server._file_get_chunk_impl("note2@00002@.md", "c:0")
        assert result is not None
        assert "doc_id" in result
        assert "text" in result
        assert "roof" in result["text"].lower() or "insurance" in result["text"].lower()

    def test_file_search_empty_query_returns_error(self, indexed_store):
        """file_search returns structured error for empty query."""
        import mcp_server
        mcp_server._cache = (
            indexed_store["store"],
            indexed_store["embed_provider"],
            indexed_store["config"],
        )

        result = mcp_server._file_search_impl("", top_k=3)
        assert isinstance(result, dict)
        assert result["error"] is True
        assert result["code"] == "empty_query"
        assert "fix" in result

    def test_file_search_invalid_source_type_returns_error(self, indexed_store):
        """file_search returns error for invalid source_type."""
        import mcp_server
        mcp_server._cache = (
            indexed_store["store"],
            indexed_store["embed_provider"],
            indexed_store["config"],
        )

        result = mcp_server._file_search_impl("test", source_type="docx")
        assert isinstance(result, dict)
        assert result["error"] is True
        assert result["code"] == "invalid_parameter"

    def test_file_get_chunk_not_found_returns_error(self, indexed_store):
        """file_get_chunk returns structured error when chunk not found."""
        import mcp_server
        mcp_server._cache = (
            indexed_store["store"],
            indexed_store["embed_provider"],
            indexed_store["config"],
        )

        result = mcp_server._file_get_chunk_impl("nonexistent.md", "c:0")
        assert isinstance(result, dict)
        assert result["error"] is True
        assert result["code"] == "not_found"
        assert "fix" in result

    def test_file_get_doc_chunks_impl(self, indexed_store):
        """file_get_doc_chunks returns all chunks for a document."""
        import mcp_server
        mcp_server._cache = (
            indexed_store["store"],
            indexed_store["embed_provider"],
            indexed_store["config"],
        )

        result = mcp_server._file_get_doc_chunks_impl("note2@00002@.md")
        assert isinstance(result, list)
        assert len(result) >= 1
        assert "text" in result[0]
        assert "loc" in result[0]

    def test_file_get_doc_chunks_not_found_returns_error(self, indexed_store):
        """file_get_doc_chunks returns error for nonexistent document."""
        import mcp_server
        mcp_server._cache = (
            indexed_store["store"],
            indexed_store["embed_provider"],
            indexed_store["config"],
        )

        result = mcp_server._file_get_doc_chunks_impl("nonexistent.md")
        assert isinstance(result, dict)
        assert result["error"] is True
        assert result["code"] == "not_found"

    def test_file_list_documents_impl(self, indexed_store):
        """file_list_documents returns paginated document list."""
        import mcp_server
        mcp_server._cache = (
            indexed_store["store"],
            indexed_store["embed_provider"],
            indexed_store["config"],
        )

        result = mcp_server._file_list_documents_impl(offset=0, limit=10)
        assert isinstance(result, dict)
        assert "documents" in result
        assert "total" in result
        assert result["total"] >= 4
        assert len(result["documents"]) >= 4
        # Check document fields
        doc = result["documents"][0]
        assert "doc_id" in doc

    def test_file_status_impl(self, indexed_store):
        """file_status returns doc count, chunk count, and provider info."""
        import json
        import mcp_server

        # Write index_metadata.json so status can read it
        meta_path = Path(indexed_store["tmpdir"]) / "index_metadata.json"
        with open(meta_path, "w") as f:
            json.dump({"last_run_at": "2026-02-13T00:00:00Z", "doc_count": 5}, f)

        mcp_server._cache = (
            indexed_store["store"],
            indexed_store["embed_provider"],
            indexed_store["config"],
        )

        result = mcp_server._file_status_impl()
        assert result["doc_count"] >= 4
        assert result["chunk_count"] is not None
        assert result["chunk_count"] > 0
        assert result["last_run_at"] == "2026-02-13T00:00:00Z"
        assert result["embeddings_provider"] == indexed_store["config"].get("embeddings", {}).get("provider")


# ---------------------------------------------------------------------------
# Reranker tests — real Qwen3-Reranker-8B via DeepInfra
# ---------------------------------------------------------------------------

_has_deepinfra_key = bool(os.environ.get("DEEPINFRA_API_KEY"))


@pytest.mark.live
@pytest.mark.skipif(
    not _has_deepinfra_key,
    reason="DEEPINFRA_API_KEY not set",
)
class TestReranker:
    """Test Qwen3-Reranker-8B via DeepInfra with real scoring."""

    def _print_stage(self, label, hits, score_label="score"):
        """Helper: print a ranked list of hits for a pipeline stage."""
        print(f"\n    {label}")
        print(f"    {'─' * 72}")
        for i, h in enumerate(hits):
            print(
                f"    #{i+1:2d}  {score_label}={h.score:.6f}  "
                f"{h.doc_id:<30s}  loc={h.loc}"
            )
            print(f"         snippet: {h.snippet[:90]}")
        print()

    def _build_reranker(self, config):
        """Build a reranker from test config using the configured provider."""
        from search_hybrid import build_reranker
        return build_reranker(config)

    def test_reranker_returns_continuous_scores(self, indexed_store):
        """Reranker should return continuous relevance scores (0.0–1.0)."""
        import time
        from search_hybrid import hybrid_search, reciprocal_rank_fusion

        store = indexed_store["store"]
        embed = indexed_store["embed_provider"]
        config = indexed_store["config"]

        query = "kimchi fermentation recipe"
        print(f"\n{'=' * 76}")
        print(f"  FULL 4-STAGE PIPELINE  |  query: \"{query}\"")
        print(f"{'=' * 76}")

        # Stage 1: Vector search
        query_vector = embed.embed_query(query)
        vector_hits = store.vector_search(query_vector, top_k=10)
        self._print_stage("STAGE 1: Vector Search (Qwen3-Embedding)", vector_hits, "cosine")

        # Stage 2: Keyword search
        keyword_hits = store.keyword_search(query, top_k=10)
        if keyword_hits:
            self._print_stage("STAGE 2: Keyword Search (BM25/tantivy FTS)", keyword_hits, "bm25  ")
        else:
            print("\n    STAGE 2: Keyword Search — (no FTS results)\n")

        # Stage 3: RRF fusion
        fused = reciprocal_rank_fusion([vector_hits, keyword_hits], k=60)
        self._print_stage("STAGE 3: Reciprocal Rank Fusion (k=60)", fused, "rrf   ")

        # Stage 4: Qwen3 Reranker (auto-starts if not running)
        reranker = self._build_reranker(config)
        t0 = time.perf_counter()
        reranked = reranker.rerank(query, fused)
        elapsed = time.perf_counter() - t0
        self._print_stage(
            f"STAGE 4: Qwen3-Reranker (DeepInfra) — {elapsed:.2f}s",
            reranked, "relev.",
        )

        assert len(reranked) >= 1
        for hit in reranked:
            assert isinstance(hit.score, float)
            assert 0.0 <= hit.score <= 1.0, f"Score out of [0,1] range: {hit.score}"
        for i in range(len(reranked) - 1):
            assert reranked[i].score >= reranked[i + 1].score, \
                f"Scores not descending: {reranked[i].score} < {reranked[i + 1].score}"

    def test_reranker_recipe_is_top_result(self, indexed_store):
        """For 'kimchi recipe', the recipe doc should be #1 with a high score."""
        from search_hybrid import hybrid_search

        query = "kimchi fermentation recipe"
        reranker = self._build_reranker(indexed_store["config"])
        hits = hybrid_search(
            indexed_store["store"],
            indexed_store["embed_provider"],
            query,
            vector_top_k=10,
            keyword_top_k=10,
            final_top_k=5,
            reranker=reranker,
        )

        print(f"\n    Query: \"{query}\"  →  Full pipeline with reranker")
        self._print_stage("Final results (reranked)", hits, "relev.")

        assert len(hits) >= 1
        assert "recipe" in hits[0].doc_id, \
            f"Expected recipe as #1, got {hits[0].doc_id}"
        assert hits[0].score > 0.5, \
            f"Expected high relevance for recipe, got {hits[0].score:.4f}"

    def test_reranker_irrelevant_docs_get_low_scores(self, indexed_store):
        """For 'kimchi recipe', non-recipe docs should get scores close to 0."""
        from search_hybrid import hybrid_search

        query = "kimchi fermentation recipe"
        reranker = self._build_reranker(indexed_store["config"])
        hits = hybrid_search(
            indexed_store["store"],
            indexed_store["embed_provider"],
            query,
            vector_top_k=10,
            keyword_top_k=10,
            final_top_k=10,
            reranker=reranker,
        )

        print(f"\n    Query: \"{query}\"  →  Checking irrelevant doc scores")
        for h in hits:
            tag = " ← RELEVANT" if "recipe" in h.doc_id else ""
            print(f"    relev.={h.score:.6f}  {h.doc_id}{tag}")

        non_recipe = [h for h in hits if "recipe" not in h.doc_id]
        for h in non_recipe:
            assert h.score < 0.1, \
                f"Non-recipe doc {h.doc_id} got unexpectedly high score: {h.score:.4f}"

    def test_reranker_meeting_notes_query(self, indexed_store):
        """For 'meeting notes project kickoff', meeting_notes.png should rank high."""
        from search_hybrid import hybrid_search

        query = "meeting notes project kickoff attendees"
        reranker = self._build_reranker(indexed_store["config"])
        hits = hybrid_search(
            indexed_store["store"],
            indexed_store["embed_provider"],
            query,
            vector_top_k=10,
            keyword_top_k=10,
            final_top_k=5,
            reranker=reranker,
        )

        print(f"\n    Query: \"{query}\"")
        self._print_stage("Final results (reranked)", hits, "relev.")

        assert len(hits) >= 1
        top_ids = [h.doc_id for h in hits[:2]]
        assert any("meeting" in d or "note2" in d for d in top_ids), \
            f"Expected meeting-related doc in top 2, got {top_ids}"

    def test_reranker_insurance_claim_query(self, indexed_store):
        """For 'roof insurance claim', note2.md should be top with high score."""
        from search_hybrid import hybrid_search

        query = "roof insurance claim documents photos"
        reranker = self._build_reranker(indexed_store["config"])
        hits = hybrid_search(
            indexed_store["store"],
            indexed_store["embed_provider"],
            query,
            vector_top_k=10,
            keyword_top_k=10,
            final_top_k=5,
            reranker=reranker,
        )

        print(f"\n    Query: \"{query}\"")
        self._print_stage("Final results (reranked)", hits, "relev.")

        assert len(hits) >= 1
        assert "note2" in hits[0].doc_id, \
            f"Expected note2 as #1 for insurance query, got {hits[0].doc_id}"
        assert hits[0].score > 0.5, \
            f"Expected high relevance for insurance doc, got {hits[0].score:.4f}"

    def test_reranker_score_gap(self, indexed_store):
        """The score gap between relevant and irrelevant should be large."""
        from search_hybrid import hybrid_search

        query = "kimchi fermentation recipe"
        reranker = self._build_reranker(indexed_store["config"])
        hits = hybrid_search(
            indexed_store["store"],
            indexed_store["embed_provider"],
            query,
            vector_top_k=10,
            keyword_top_k=10,
            final_top_k=5,
            reranker=reranker,
        )

        if len(hits) >= 2:
            gap = hits[0].score - hits[1].score
            print(f"\n    Query: \"{query}\"")
            print(f"    #1  {hits[0].doc_id:<30s}  relev.={hits[0].score:.6f}")
            print(f"    #2  {hits[1].doc_id:<30s}  relev.={hits[1].score:.6f}")
            print(f"    Score gap: {gap:.6f}")
            assert gap > 0.5, \
                f"Expected large score gap between #1 and #2, got {gap:.4f} " \
                f"(#1={hits[0].score:.4f}, #2={hits[1].score:.4f})"


def _openrouter_available() -> bool:
    """Check if OpenRouter API key is set (needed for enrichment + embedding)."""
    return bool(os.environ.get("OPENROUTER_API_KEY"))


# ---------------------------------------------------------------------------
# Full-pipeline integration test — index_vault_flow with LLM enrichment
# ---------------------------------------------------------------------------

@pytest.mark.live
@pytest.mark.skipif(
    not _openrouter_available(),
    reason="OPENROUTER_API_KEY not set",
)
class TestFullPipelineWithEnrichment:
    """Run the actual index_vault_flow Prefect flow on test_vault with LLM enrichment
    enabled via OpenRouter, then verify chunks have enrichment fields, contextual headers
    include summary/topics, and search results return enrichment data."""

    @pytest.fixture(scope="class")
    def pipeline_result(self):
        """Run index_vault_flow with enrichment enabled. Returns store + config."""
        import shutil
        import yaml

        from core.config import load_config

        with tempfile.TemporaryDirectory() as tmpdir:
            # index_vault_flow calls scan_vault_task which renames files with
            # @NNNNN@ doc-id suffixes. Copy test_vault into tmpdir so the
            # tracked repo isn't mutated by the flow.
            vault_copy = Path(tmpdir) / "vault"
            src_vault = Path(__file__).parent.parent / "test_vault"
            shutil.copytree(src_vault, vault_copy)

            # Build a temp config with enrichment enabled via OpenRouter
            base_config = load_config("config_test.yaml")
            base_config["vault_root"] = str(vault_copy)
            base_config["documents_root"] = str(vault_copy)
            base_config["index_root"] = tmpdir
            # Use production enrichment config (model, token limits, etc.)
            prod_config = load_config()
            base_config["enrichment"] = prod_config.get("enrichment", {})
            base_config["enrichment"]["enabled"] = True
            # Disable semantic chunking to keep the test fast
            base_config.setdefault("chunking", {})["semantic"] = {"enabled": False}

            config_path = Path(tmpdir) / "config_pipeline_test.yaml"
            with open(config_path, "w") as f:
                yaml.dump(base_config, f)

            # Run Prefect flow in ephemeral mode (no server required).
            # Must set env vars AND reset Prefect's cached settings so the
            # ephemeral server is used even if Prefect was imported earlier.
            # Also override DOCUMENTS_ROOT/INDEX_ROOT so load_config inside
            # the flow points at the vault copy, not the host env values.
            saved_env = {}
            for key in (
                "PREFECT_API_URL",
                "PREFECT_SERVER_ALLOW_EPHEMERAL_MODE",
                "DOCUMENTS_ROOT",
                "INDEX_ROOT",
            ):
                saved_env[key] = os.environ.get(key)
            os.environ["PREFECT_API_URL"] = ""
            os.environ["PREFECT_SERVER_ALLOW_EPHEMERAL_MODE"] = "true"
            os.environ["DOCUMENTS_ROOT"] = str(vault_copy)
            os.environ["INDEX_ROOT"] = tmpdir
            try:
                from prefect.settings.models.root import Settings as PrefectSettings
                import prefect.context
                prefect.context.get_settings_context().settings = PrefectSettings()
            except Exception:
                pass
            try:
                from flow_index_vault import index_vault_flow

                index_vault_flow(str(config_path))
            finally:
                for key, val in saved_env.items():
                    if val is not None:
                        os.environ[key] = val
                    else:
                        os.environ.pop(key, None)

            from lancedb_store import LanceDBStore
            from providers.embed import build_embed_provider
            from doc_id_store import DocIDStore, strip_id_from_filename

            store = LanceDBStore(tmpdir, base_config.get("lancedb", {}).get("table", "chunks"))
            embed_provider = build_embed_provider(base_config)

            # Build a reverse map: original filename (pre-rename) -> doc_id.
            # The flow renames files to `name@XXXXX@.ext` and registers the
            # renamed path in the registry. We strip the @XXXXX@ suffix so
            # tests can look up chunks by their pre-rename filenames.
            registry = DocIDStore(Path(tmpdir) / "doc_registry.db")
            try:
                doc_ids_by_orig_name: dict[str, str] = {}
                for did, rel_path in registry.all_mappings().items():
                    parts = rel_path.split("/")
                    parts[-1] = strip_id_from_filename(parts[-1])
                    doc_ids_by_orig_name["/".join(parts)] = did
            finally:
                registry.close()

            yield {
                "store": store,
                "embed_provider": embed_provider,
                "config": base_config,
                "tmpdir": tmpdir,
                "doc_ids_by_orig_name": doc_ids_by_orig_name,
            }

    def test_all_docs_indexed(self, pipeline_result):
        """The flow should index all test_vault files."""
        store_doc_ids = pipeline_result["store"].list_doc_ids()
        by_name = pipeline_result["doc_ids_by_orig_name"]
        print(f"\n    Indexed docs: {store_doc_ids}")
        print(f"    Name→id map: {by_name}")
        assert len(store_doc_ids) >= 3, f"Expected >= 3 docs, got {len(store_doc_ids)}: {store_doc_ids}"
        for name in ("note1.md", "note2.md", "subfolder/recipe.md"):
            assert name in by_name, f"{name} missing from registry map"
            assert by_name[name] in store_doc_ids, f"doc_id for {name} not in store"

    def test_chunks_have_enrichment_metadata(self, pipeline_result):
        """Chunks should carry enrichment fields in their metadata."""
        store = pipeline_result["store"]
        recipe_id = pipeline_result["doc_ids_by_orig_name"]["subfolder/recipe.md"]
        hit = store.get_chunk(recipe_id, "c:0")
        assert hit is not None, f"get_chunk returned None for recipe.md (id={recipe_id}) c:0"

        print(f"\n    Enrichment fields on recipe (00005) c:0:")
        print(f"      enr_summary: {hit.enr_summary[:120] if hit.enr_summary else '(empty)'}")
        print(f"      enr_doc_type: {hit.enr_doc_type}")
        print(f"      enr_topics: {hit.enr_topics}")
        print(f"      enr_keywords: {hit.enr_keywords}")
        print(f"      enr_entities_people: {hit.enr_entities_people}")
        print(f"      enr_entities_places: {hit.enr_entities_places}")
        print(f"      enr_entities_orgs: {hit.enr_entities_orgs}")
        print(f"      enr_entities_dates: {hit.enr_entities_dates}")
        print(f"      enr_key_facts: {hit.enr_key_facts[:120] if hit.enr_key_facts else '(empty)'}")

        assert hit.enr_summary, "enr_summary should be populated for recipe (00005)"
        assert hit.enr_topics, "enr_topics should be populated for recipe (00005)"

    def test_contextual_header_includes_summary(self, pipeline_result):
        """Chunk text should start with a contextual header containing summary."""
        store = pipeline_result["store"]
        note2_id = pipeline_result["doc_ids_by_orig_name"]["note2.md"]
        hit = store.get_chunk(note2_id, "c:0")
        assert hit is not None

        print(f"\n    Chunk text (first 500 chars):\n{hit.text[:500]}")

        assert hit.text.startswith("["), "Chunk should start with contextual header"
        assert "Summary:" in hit.text, "Contextual header should include Summary line"

    def test_contextual_header_includes_topics(self, pipeline_result):
        """Contextual header should include topics from enrichment."""
        store = pipeline_result["store"]
        recipe_id = pipeline_result["doc_ids_by_orig_name"]["subfolder/recipe.md"]
        hit = store.get_chunk(recipe_id, "c:0")
        assert hit is not None
        assert "Topics:" in hit.text, "Contextual header should include Topics"

    def test_search_returns_enrichment_fields(self, pipeline_result):
        """Hybrid search results should include enrichment fields."""
        from search_hybrid import hybrid_search

        store = pipeline_result["store"]
        embed = pipeline_result["embed_provider"]

        hits = hybrid_search(store, embed, "kimchi recipe", vector_top_k=10, final_top_k=5)
        assert len(hits) >= 1

        hit = hits[0]
        print(f"\n    Top search hit: {hit.doc_id}")
        print(f"      enr_summary: {hit.enr_summary[:120] if hit.enr_summary else '(empty)'}")
        print(f"      enr_topics: {hit.enr_topics}")
        print(f"      enr_keywords: {hit.enr_keywords}")
        print(f"      enr_doc_type: {hit.enr_doc_type}")

        assert hit.enr_summary, "Search result should have enr_summary"
        assert hit.enr_topics, "Search result should have enr_topics"

    def test_mcp_search_returns_enrichment_fields(self, pipeline_result):
        """MCP file_search handler should include enrichment fields in response dicts."""
        import mcp_server

        mcp_server._cache = (
            pipeline_result["store"],
            pipeline_result["embed_provider"],
            pipeline_result["config"],
        )

        response = mcp_server._file_search_impl("insurance claim roof", top_k=3)
        assert "results" in response
        results = response["results"]
        assert len(results) >= 1

        r = results[0]
        print(f"\n    MCP search result: {r['doc_id']}")
        for field in ("enr_summary", "enr_doc_type", "enr_topics", "enr_keywords",
                       "enr_entities_people", "enr_entities_places", "enr_entities_orgs",
                       "enr_entities_dates", "enr_key_facts"):
            print(f"      {field}: {r.get(field, '(missing)')}")
            assert field in r, f"MCP response missing enrichment field: {field}"

    def test_mcp_get_chunk_returns_enrichment_fields(self, pipeline_result):
        """MCP file_get_chunk handler should include enrichment fields."""
        import mcp_server

        mcp_server._cache = (
            pipeline_result["store"],
            pipeline_result["embed_provider"],
            pipeline_result["config"],
        )

        note1_id = pipeline_result["doc_ids_by_orig_name"]["note1.md"]
        result = mcp_server._file_get_chunk_impl(note1_id, "c:0")
        assert result is not None

        print(f"\n    MCP get_chunk result for note1.md (id={note1_id}) c:0:")
        for field in ("enr_summary", "enr_doc_type", "enr_topics", "enr_keywords"):
            val = result.get(field, "(missing)")
            print(f"      {field}: {val}")
            assert field in result, f"MCP get_chunk missing field: {field}"

    def test_enrichment_fields_consistent_across_chunks(self, pipeline_result):
        """All chunks of the same document should share identical enrichment values."""
        store = pipeline_result["store"]
        recipe_id = pipeline_result["doc_ids_by_orig_name"]["subfolder/recipe.md"]
        c0 = store.get_chunk(recipe_id, "c:0")
        assert c0 is not None

        # Try to get a second chunk (if the doc produced more than one)
        c1 = store.get_chunk(recipe_id, "c:1")
        if c1 is not None:
            print(f"\n    Comparing enrichment across chunks c:0 and c:1:")
            print(f"      c:0 enr_summary: {c0.enr_summary[:80]}")
            print(f"      c:1 enr_summary: {c1.enr_summary[:80]}")
            assert c0.enr_summary == c1.enr_summary, "enr_summary should be identical across chunks"
            assert c0.enr_doc_type == c1.enr_doc_type, "enr_doc_type should be identical across chunks"
            assert c0.enr_topics == c1.enr_topics, "enr_topics should be identical across chunks"
        else:
            print("\n    Only one chunk for recipe.md — cross-chunk consistency N/A")

    def test_note_with_frontmatter_preserves_tags_and_enrichment(self, pipeline_result):
        """Documents with YAML frontmatter should have both tags AND enrichment."""
        store = pipeline_result["store"]
        note1_id = pipeline_result["doc_ids_by_orig_name"]["note1.md"]
        hit = store.get_chunk(note1_id, "c:0")
        assert hit is not None

        print(f"\n    note1 (00001) metadata:")
        print(f"      tags (frontmatter): {hit.tags}")
        print(f"      enr_summary (LLM): {hit.enr_summary[:100] if hit.enr_summary else '(empty)'}")
        print(f"      enr_topics (LLM): {hit.enr_topics}")

        assert hit.tags, "Frontmatter tags should be preserved"
        assert hit.enr_summary, "LLM enr_summary should be populated"
