"""End-to-end live tests: real cloud services, real LanceDB, full pipeline.

Exercises the full chain: enrich → embed → store → search → rerank.
Requires OPENROUTER_API_KEY and DEEPINFRA_API_KEY. Skipped when keys are missing.

Run with:  pytest tests/test_e2e_live.py -v
"""

import os
import tempfile
import time

import pytest

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
_has_deepinfra = bool(os.environ.get("DEEPINFRA_API_KEY"))


# ---------------------------------------------------------------------------
# End-to-end: enrich → embed → store → search → rerank
# ---------------------------------------------------------------------------

@pytest.mark.live
@pytest.mark.skipif(
    not (_has_openrouter and _has_deepinfra),
    reason="OPENROUTER_API_KEY and/or DEEPINFRA_API_KEY not set",
)
class TestEndToEndLive:
    """Full pipeline test with real cloud services and real LanceDB."""

    # -- Test documents ------------------------------------------------

    DOCS = {
        "recipe.md": {
            "text": (
                "# Traditional Korean Kimchi Recipe\n\n"
                "## Ingredients\n"
                "- 1 large napa cabbage (about 2 kg)\n"
                "- 1/2 cup Korean sea salt\n"
                "- 1 cup Korean red pepper flakes (gochugaru)\n"
                "- 1/4 cup fish sauce\n"
                "- 6 cloves garlic, minced\n"
                "- 1 tbsp fresh ginger, grated\n"
                "- 4 green onions, sliced\n\n"
                "## Instructions\n"
                "1. Cut cabbage into quarters and salt thoroughly.\n"
                "2. Let sit for 2 hours, turning every 30 minutes.\n"
                "3. Mix gochugaru, fish sauce, garlic, ginger, and green onions.\n"
                "4. Rinse cabbage and apply paste between leaves.\n"
                "5. Pack into jars and ferment at room temperature for 2-5 days.\n"
                "6. Refrigerate after fermentation reaches desired tanginess.\n\n"
                "Fermentation time depends on ambient temperature. In summer, "
                "kimchi may be ready in 1-2 days. In winter, allow 5-7 days."
            ),
            "source_type": "md",
            "tags": "cooking, korean, fermentation",
            "folder": "Recipes",
            "status": "published",
            "created": "2024-01-15",
            "author": "Grandma Park",
            "keywords": "kimchi, fermentation, korean food",
        },
        "insurance_claim.md": {
            "text": (
                "# Insurance Claim Report #2024-5678\n\n"
                "**Claimant:** John Smith\n"
                "**Date of Loss:** March 15, 2024\n"
                "**Property:** 123 Main Street, Springfield, IL 62701\n"
                "**Adjuster:** Sarah Johnson, ABC Insurance\n\n"
                "## Damage Assessment\n"
                "Severe roof damage from hailstorm on March 15, 2024. "
                "Multiple shingles displaced, two areas of exposed decking. "
                "Water intrusion detected in attic space.\n\n"
                "## Estimated Repair Cost\n"
                "- Roof replacement: $8,500\n"
                "- Attic remediation: $2,000\n"
                "- Interior ceiling repair: $2,000\n"
                "- **Total: $12,500**\n\n"
                "Claim approved for full amount. Payment issued April 1, 2024."
            ),
            "source_type": "md",
            "tags": "insurance, claims, property",
            "folder": "Insurance/Claims",
            "status": "approved",
            "created": "2024-03-20",
            "author": "Sarah Johnson",
            "keywords": "insurance claim, hail damage, roof repair",
        },
        "meeting_notes.md": {
            "text": (
                "# Engineering Team Meeting — 2024-06-15\n\n"
                "**Attendees:** Alice Chen, Bob Park, Carol Davis\n"
                "**Location:** Conference Room B, Building 4\n\n"
                "## Discussion Topics\n"
                "1. Q3 product roadmap review\n"
                "2. Migration from PostgreSQL to CockroachDB\n"
                "3. New CI/CD pipeline using GitHub Actions\n\n"
                "## Action Items\n"
                "- Alice: Draft CockroachDB migration plan by June 22\n"
                "- Bob: Set up GitHub Actions workflows for staging\n"
                "- Carol: Performance benchmarks comparing Postgres vs CockroachDB\n\n"
                "## Decisions\n"
                "- Approved migration to CockroachDB for Q3\n"
                "- Budget approved for additional staging environment"
            ),
            "source_type": "md",
            "tags": "engineering, meetings, Q3",
            "folder": "Work/Meetings",
            "status": "final",
            "created": "2024-06-15",
            "author": "Alice Chen",
            "keywords": "CockroachDB, migration, CI/CD, GitHub Actions",
        },
    }

    @pytest.fixture(scope="class")
    def pipeline_result(self):
        """Enrich all docs, embed chunks, store in LanceDB."""
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

        from core.config import load_config
        from doc_enrichment import enrich_document
        from lancedb_store import LanceDBStore
        from providers.embed import build_embed_provider
        from providers.llm import build_llm_provider

        config = load_config()
        embed_provider = build_embed_provider(config)
        llm_generator = build_llm_provider(config)
        enrichment_cfg = config.get("enrichment", {})
        chunking_cfg = config.get("chunking", {})
        splitter = SentenceSplitter(
            chunk_size=chunking_cfg.get("max_chars", 1800),
            chunk_overlap=chunking_cfg.get("overlap", 200),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store = LanceDBStore(tmpdir, "chunks")

            for doc_id, doc in self.DOCS.items():
                # Enrich via real OpenRouter LLM (using production config)
                enrichment = {}
                if llm_generator:
                    enrichment = enrich_document(
                        doc["text"], doc_id, doc["source_type"], llm_generator,
                        max_input_chars=enrichment_cfg.get("max_input_chars", 20000),
                        max_output_tokens=enrichment_cfg.get("max_output_tokens", 5000),
                    )
                    enrichment.pop("_enrichment_failed", None)

                # Chunk
                chunks_text = splitter.split_text(doc["text"])

                # Embed via real OpenRouter embeddings
                vectors = embed_provider.embed_texts(chunks_text)

                # Build TextNodes (same format as the real pipeline)
                nodes = []
                for i, (text, vec) in enumerate(zip(chunks_text, vectors)):
                    loc = f"c:{i}"
                    meta = {
                        "doc_id": doc_id,
                        "source_type": doc["source_type"],
                        "loc": loc,
                        "snippet": text[:200],
                        "title": doc_id.replace(".md", ""),
                        "tags": doc.get("tags", ""),
                        "folder": doc.get("folder", ""),
                        "status": doc.get("status", ""),
                        "created": doc.get("created", ""),
                        "description": doc.get("description", ""),
                        "author": doc.get("author", ""),
                        "keywords": doc.get("keywords", ""),
                        "custom_meta": "",
                        "section": "",
                        "size": len(doc["text"]),
                        "mtime": time.time(),
                    }
                    meta.update(enrichment)
                    node = TextNode(
                        text=text,
                        id_=f"{doc_id}::{loc}",
                        embedding=vec,
                        metadata=meta,
                    )
                    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
                    nodes.append(node)

                store.upsert_nodes(nodes)

            # Build FTS index
            store.create_fts_index()

            yield {
                "store": store,
                "embed_provider": embed_provider,
                "config": config,
                "tmpdir": tmpdir,
            }

    # -- Search tests ---------------------------------------------------

    def test_search_finds_relevant_doc(self, pipeline_result):
        """Searching for 'kimchi recipe' should return the recipe doc first."""
        from search_hybrid import hybrid_search

        store = pipeline_result["store"]
        embed = pipeline_result["embed_provider"]

        result = hybrid_search(
            store, embed, "Korean kimchi fermentation recipe",
            vector_top_k=10, final_top_k=5,
        )
        assert len(result.hits) > 0, "Expected search results"
        assert result.hits[0].doc_id == "recipe.md", (
            f"Expected recipe.md as top hit, got {result.hits[0].doc_id}"
        )

    def test_search_insurance_claim(self, pipeline_result):
        """Searching for 'roof damage insurance' should return the claim doc."""
        from search_hybrid import hybrid_search

        store = pipeline_result["store"]
        embed = pipeline_result["embed_provider"]

        result = hybrid_search(
            store, embed, "roof damage hail insurance claim cost",
            vector_top_k=10, final_top_k=5,
        )
        assert len(result.hits) > 0
        doc_ids = [h.doc_id for h in result.hits]
        assert "insurance_claim.md" in doc_ids, (
            f"Expected insurance_claim.md in results, got {doc_ids}"
        )

    def test_enrichment_fields_populated(self, pipeline_result):
        """At least one doc should have non-empty enrichment from real LLM."""
        from search_hybrid import hybrid_search

        store = pipeline_result["store"]
        embed = pipeline_result["embed_provider"]

        # Search broadly to get hits from all docs
        result = hybrid_search(
            store, embed, "kimchi recipe insurance meeting",
            vector_top_k=10, final_top_k=10,
        )
        # At least one hit across all docs should have enrichment
        enriched = [h for h in result.hits if h.enr_summary]
        assert enriched, (
            f"Expected at least one hit with enr_summary populated, "
            f"got {len(result.hits)} hits all with empty summary"
        )

    def test_keyword_search_works(self, pipeline_result):
        """FTS keyword search should find docs by specific terms."""
        from search_hybrid import hybrid_search

        store = pipeline_result["store"]
        embed = pipeline_result["embed_provider"]

        result = hybrid_search(
            store, embed, "gochugaru fish sauce",
            vector_top_k=10, final_top_k=5,
        )
        assert len(result.hits) > 0
        assert result.hits[0].doc_id == "recipe.md"

    def test_search_with_reranker(self, pipeline_result):
        """Full search with DeepInfra reranker should return properly scored results."""
        from search_hybrid import hybrid_search, build_reranker

        store = pipeline_result["store"]
        embed = pipeline_result["embed_provider"]
        config = pipeline_result["config"]

        reranker = build_reranker(config)
        if reranker is None:
            pytest.skip("Reranker not configured")

        result = hybrid_search(
            store, embed, "database migration PostgreSQL CockroachDB",
            vector_top_k=10, final_top_k=5,
            reranker=reranker,
        )
        assert len(result.hits) > 0
        if not result.diagnostics.get("reranker_applied"):
            pytest.skip("Reranker unavailable (cold start / deployment inactive)")
        assert result.hits[0].doc_id == "meeting_notes.md", (
            f"Expected meeting_notes.md as top hit after rerank, got {result.hits[0].doc_id}"
        )

    def test_search_diagnostics_complete(self, pipeline_result):
        """Search diagnostics should report all systems operational."""
        from search_hybrid import hybrid_search

        store = pipeline_result["store"]
        embed = pipeline_result["embed_provider"]

        result = hybrid_search(
            store, embed, "test query",
            vector_top_k=10, final_top_k=5,
        )
        diag = result.diagnostics
        assert diag.get("vector_search_active") is True
        assert diag.get("keyword_search_active") is True
        assert diag.get("degraded") is False

    def test_all_docs_searchable(self, pipeline_result):
        """Every indexed doc should be findable by a targeted query."""
        from search_hybrid import hybrid_search

        store = pipeline_result["store"]
        embed = pipeline_result["embed_provider"]

        queries = {
            "recipe.md": "kimchi fermentation napa cabbage",
            "insurance_claim.md": "hailstorm roof damage claim",
            "meeting_notes.md": "engineering team meeting CockroachDB",
        }
        for expected_doc, query in queries.items():
            result = hybrid_search(
                store, embed, query,
                vector_top_k=10, final_top_k=5,
            )
            found_ids = [h.doc_id for h in result.hits]
            assert expected_doc in found_ids, (
                f"Query '{query}' should find {expected_doc}, got {found_ids}"
            )

    # -- Pre-filter tests with real embeddings ------------------------------

    def test_prefilter_by_folder(self, pipeline_result):
        """folder pre-filter with real embeddings should restrict results."""
        from search_hybrid import hybrid_search

        store = pipeline_result["store"]
        embed = pipeline_result["embed_provider"]

        # Search for "kimchi recipe" but restrict to Insurance/Claims folder
        result = hybrid_search(
            store, embed, "kimchi recipe fermentation",
            vector_top_k=10, final_top_k=5, folder="Insurance/Claims",
        )
        for h in result.hits:
            assert h.folder == "Insurance/Claims", (
                f"Expected folder='Insurance/Claims', got folder='{h.folder}' (doc={h.doc_id})"
            )
        # recipe.md is in Recipes folder, should NOT appear
        found_ids = [h.doc_id for h in result.hits]
        assert "recipe.md" not in found_ids

    def test_prefilter_by_tags(self, pipeline_result):
        """tags pre-filter with real embeddings should restrict results."""
        from search_hybrid import hybrid_search

        store = pipeline_result["store"]
        embed = pipeline_result["embed_provider"]

        result = hybrid_search(
            store, embed, "document report meeting recipe",
            vector_top_k=10, final_top_k=10, tags="insurance",
        )
        assert len(result.hits) >= 1
        for h in result.hits:
            assert "insurance" in (h.tags or "").lower(), (
                f"Expected 'insurance' in tags, got tags='{h.tags}' (doc={h.doc_id})"
            )

    def test_prefilter_by_enr_doc_type(self, pipeline_result):
        """enr_doc_type pre-filter with real embeddings + real enrichment.

        Uses only the primary (first) doc_type value to avoid false matches
        from generic secondary types like 'markdown' that appear across docs.
        """
        from search_hybrid import hybrid_search

        store = pipeline_result["store"]
        embed = pipeline_result["embed_provider"]

        # First, find what enr_doc_type the recipe got assigned
        recipe_result = hybrid_search(
            store, embed, "kimchi recipe",
            vector_top_k=10, final_top_k=5,
        )
        recipe_hit = next((h for h in recipe_result.hits if h.doc_id == "recipe.md"), None)
        assert recipe_hit is not None, "recipe.md should be findable"
        recipe_doc_type = recipe_hit.enr_doc_type
        if not recipe_doc_type:
            pytest.skip("recipe.md has no enr_doc_type (enrichment may not have assigned one)")

        # Use only the primary (first) doc_type — avoids generic secondary types
        # like "markdown" that appear across many docs and would match via OR
        primary_type = recipe_doc_type.split(",")[0].strip()

        # Now search with that doc_type as pre-filter
        filtered_result = hybrid_search(
            store, embed, "ingredients cooking fermentation",
            vector_top_k=10, final_top_k=10, enr_doc_type=primary_type,
        )
        assert len(filtered_result.hits) >= 1
        for h in filtered_result.hits:
            assert primary_type.lower() in (h.enr_doc_type or "").lower(), (
                f"Expected enr_doc_type containing '{primary_type}', "
                f"got '{h.enr_doc_type}' (doc={h.doc_id})"
            )

    def test_prefilter_by_enr_topics(self, pipeline_result):
        """enr_topics pre-filter with real embeddings + real enrichment."""
        from search_hybrid import hybrid_search

        store = pipeline_result["store"]
        embed = pipeline_result["embed_provider"]

        # Find what topics the meeting_notes got
        meeting_result = hybrid_search(
            store, embed, "engineering meeting CockroachDB migration",
            vector_top_k=10, final_top_k=5,
        )
        meeting_hit = next((h for h in meeting_result.hits if h.doc_id == "meeting_notes.md"), None)
        assert meeting_hit is not None, "meeting_notes.md should be findable"
        topics = meeting_hit.enr_topics
        if not topics:
            pytest.skip("meeting_notes.md has no enr_topics")

        # Pick the first topic and use it as a filter
        first_topic = topics.split(",")[0].strip()
        filtered_result = hybrid_search(
            store, embed, "team discussion action items",
            vector_top_k=10, final_top_k=10, enr_topics=first_topic,
        )
        assert len(filtered_result.hits) >= 1
        for h in filtered_result.hits:
            assert first_topic.lower() in (h.enr_topics or "").lower(), (
                f"Expected enr_topics containing '{first_topic}', "
                f"got '{h.enr_topics}' (doc={h.doc_id})"
            )

    def test_prefilter_excludes_non_matching(self, pipeline_result):
        """Pre-filter should give full top_k of matching docs, not truncate."""
        from search_hybrid import hybrid_search

        store = pipeline_result["store"]
        embed = pipeline_result["embed_provider"]

        # Without filter: broad query should find all 3 docs
        unfiltered = hybrid_search(
            store, embed, "document report content",
            vector_top_k=10, final_top_k=10,
        )
        all_doc_ids = {h.doc_id for h in unfiltered.hits}
        assert len(all_doc_ids) >= 2, f"Expected multiple docs, got {all_doc_ids}"

        # With folder filter: only Recipes folder
        filtered = hybrid_search(
            store, embed, "document report content",
            vector_top_k=10, final_top_k=10, folder="Recipes",
        )
        filtered_ids = {h.doc_id for h in filtered.hits}
        assert filtered_ids <= {"recipe.md"}, (
            f"Expected only recipe.md in Recipes folder, got {filtered_ids}"
        )


# ---------------------------------------------------------------------------
# Taxonomy: seed import, CRUD, search — all in a temp LanceDB (no prod writes)
# ---------------------------------------------------------------------------
# TestTaxonomyLive removed — SQLite seed databases (tags.db, directory.db)
# deleted; taxonomy is now managed exclusively via MCP tools in LanceDB.
# ---------------------------------------------------------------------------
