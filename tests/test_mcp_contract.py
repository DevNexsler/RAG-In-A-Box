"""MCP output contract tests — verify response shapes match documented API.

No external services needed. Uses mocks and direct function calls."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.storage import SearchHit
import mcp_server


# ---------------------------------------------------------------------------
# _hit_to_dict contract
# ---------------------------------------------------------------------------


def test_hit_to_dict_all_fields_present():
    """Construct a SearchHit with every field populated and verify output dict."""
    hit = SearchHit(
        doc_id="Projects/recipe.md",
        loc="c:0",
        snippet="A Korean recipe for bibimbap...",
        text="Full text of the recipe chunk.",
        score=0.95,
        source_type="md",
        title="Bibimbap Recipe",
        tags="recipe,korean",
        folder="Projects",
        status="active",
        created="2026-01-15",
        mtime=1700000000.0,
        description="Traditional Korean rice dish",
        author="Dan Park",
        keywords="bibimbap,gochujang,korean rice",
        custom_meta='{"source": "cookbook"}',
        enr_summary="A detailed Korean cooking recipe.",
        enr_doc_type="recipe",
        enr_topics="korean cooking, bibimbap",
        enr_keywords="rice, vegetables, gochujang",
        enr_entities_people="Dan Park",
        enr_entities_places="Seoul",
        enr_entities_orgs="",
        enr_entities_dates="2026-01-15",
        enr_key_facts="Traditional Korean dish, mixed rice bowl",
        extra_metadata={"section": "Ingredients", "priority": "high"},
    )
    d = mcp_server._hit_to_dict(hit)

    # Core fields
    assert d["doc_id"] == "Projects/recipe.md"
    assert d["loc"] == "c:0"
    assert d["snippet"] == "A Korean recipe for bibimbap..."
    assert d["score"] == 0.95
    assert d["title"] == "Bibimbap Recipe"
    assert d["folder"] == "Projects"
    assert d["status"] == "active"
    assert d["source_type"] == "md"
    assert d["description"] == "Traditional Korean rice dish"
    assert d["author"] == "Dan Park"
    assert isinstance(d["keywords"], list)
    assert d["keywords"] == ["bibimbap", "gochujang", "korean rice"]
    assert d["custom_meta"] == '{"source": "cookbook"}'

    # tags must be a list (not comma-separated string)
    assert isinstance(d["tags"], list)
    assert d["tags"] == ["recipe", "korean"]

    # Enrichment fields must use enr_ prefix
    assert d["enr_summary"] == "A detailed Korean cooking recipe."
    assert d["enr_doc_type"] == "recipe"
    assert d["enr_topics"] == "korean cooking, bibimbap"
    assert d["enr_keywords"] == "rice, vegetables, gochujang"
    assert d["enr_entities_people"] == "Dan Park"
    assert d["enr_entities_places"] == "Seoul"
    assert d["enr_entities_orgs"] == ""
    assert d["enr_entities_dates"] == "2026-01-15"
    assert d["enr_key_facts"] == "Traditional Korean dish, mixed rice bowl"

    # Unprefixed enrichment names must NOT appear (keywords is a frontmatter field, not enrichment)
    assert "summary" not in d  # only enr_summary
    assert "doc_type" not in d
    assert "topics" not in d

    # Dynamic metadata fields included
    assert d["section"] == "Ingredients"
    assert d["priority"] == "high"

    # text should NOT be included (include_text defaults to False)
    assert "text" not in d


def test_hit_to_dict_empty_enrichment():
    """SearchHit with empty enrichment should still have enr_* keys as empty strings."""
    hit = SearchHit(
        doc_id="a.md", loc="c:0", snippet="test", text="test text",
        score=0.5,
    )
    d = mcp_server._hit_to_dict(hit)

    # All enr_ keys must be present even when empty
    for key in (
        "enr_summary", "enr_doc_type", "enr_topics", "enr_keywords",
        "enr_entities_people", "enr_entities_places", "enr_entities_orgs",
        "enr_entities_dates", "enr_key_facts",
    ):
        assert key in d, f"Missing key: {key}"
        assert d[key] == "", f"Expected empty string for {key}, got {d[key]!r}"

    # tags should be empty list when None
    assert d["tags"] == []


def test_hit_to_dict_with_text():
    """include_text=True should add the text field."""
    hit = SearchHit(
        doc_id="a.md", loc="c:0", snippet="snip", text="full text here",
        score=0.5,
    )
    d = mcp_server._hit_to_dict(hit, include_text=True)
    assert d["text"] == "full text here"


# ---------------------------------------------------------------------------
# Error response structure
# ---------------------------------------------------------------------------


def test_search_error_empty_query():
    """Empty query returns structured error with expected keys."""
    result = mcp_server._file_search_impl("")
    assert isinstance(result, dict)
    assert result["error"] is True
    assert "code" in result
    assert "message" in result
    assert result["code"] == "empty_query"
    # Error response must NOT have results/diagnostics (distinguishes from success)
    assert "results" not in result
    assert "diagnostics" not in result


def test_search_error_invalid_source_type():
    """Unsafe source_type returns structured error."""
    result = mcp_server._file_search_impl("test query", source_type="bad type")
    assert isinstance(result, dict)
    assert result["error"] is True
    assert result["code"] == "invalid_parameter"
    assert "fix" in result


def test_search_accepts_custom_source_type():
    """Custom source types from non-filesystem sources should be filterable."""
    from search_hybrid import SearchResult

    old_cache = mcp_server._cache
    mcp_server._cache = None
    try:
        fake_store = MagicMock()
        fake_embed = MagicMock()
        config = {"search": {"reranker": {"enabled": False}}}
        with patch.object(mcp_server, "_get_deps", return_value=(fake_store, fake_embed, config)):
            with patch.object(mcp_server, "hybrid_search", return_value=SearchResult([])) as hybrid:
                result = mcp_server._file_search_impl("test query", source_type="pg_message")

        assert "error" not in result
        assert result["results"] == []
        assert hybrid.call_args.kwargs["source_type"] == "pg_message"
    finally:
        mcp_server._cache = old_cache


# ---------------------------------------------------------------------------
# _get_deps failure returns structured error
# ---------------------------------------------------------------------------


def test_get_deps_refreshes_cache_when_index_metadata_changes(tmp_path):
    """A completed background index run should refresh the cached LanceDB store."""
    import json
    import os

    meta_path = tmp_path / "index_metadata.json"
    meta_path.write_text(json.dumps({"last_run_at": "2026-04-30T00:00:00Z"}))
    os.utime(meta_path, ns=(1_000_000_000, 1_000_000_000))

    old_cache = mcp_server._cache
    old_signature = getattr(mcp_server, "_cache_index_signature", None)
    old_identity = getattr(mcp_server, "_cache_identity", None)
    mcp_server._cache = None
    mcp_server._cache_index_signature = None
    mcp_server._cache_identity = None
    try:
        config = {"index_root": str(tmp_path)}
        first_store = MagicMock(name="first_store")
        second_store = MagicMock(name="second_store")

        with patch.object(
            mcp_server,
            "_build_store_and_embed",
            side_effect=[
                (first_store, MagicMock(), config),
                (second_store, MagicMock(), config),
            ],
        ) as build:
            assert mcp_server._get_deps()[0] is first_store
            meta_path.write_text(json.dumps({"last_run_at": "2026-04-30T00:01:00Z"}))
            os.utime(meta_path, ns=(2_000_000_000, 2_000_000_000))

            assert mcp_server._get_deps()[0] is second_store
            assert build.call_count == 2
    finally:
        mcp_server._cache = old_cache
        mcp_server._cache_index_signature = old_signature
        mcp_server._cache_identity = old_identity


def test_get_deps_failure_returns_structured_error():
    """When _build_store_and_embed raises, _file_search_impl returns structured error."""
    old_cache = mcp_server._cache
    mcp_server._cache = None
    try:
        with patch.object(
            mcp_server, "_build_store_and_embed",
            side_effect=RuntimeError("Service connection refused"),
        ):
            result = mcp_server._file_search_impl("test query")
        assert isinstance(result, dict)
        assert result["error"] is True
        assert result["code"] == "service_unavailable"
        assert "initialize" in result["message"] or "connection" in result["message"].lower()
        assert "fix" in result
    finally:
        mcp_server._cache = old_cache


def test_get_deps_failure_in_list_documents():
    """_file_list_documents_impl returns structured error when deps fail."""
    old_cache = mcp_server._cache
    mcp_server._cache = None
    try:
        with patch.object(
            mcp_server, "_build_store_and_embed",
            side_effect=RuntimeError("bad config"),
        ):
            result = mcp_server._file_list_documents_impl()
        assert isinstance(result, dict)
        assert result["error"] is True
        assert result["code"] == "service_unavailable"
    finally:
        mcp_server._cache = old_cache


def test_get_deps_failure_in_recent():
    """_file_recent_impl returns structured error when deps fail."""
    old_cache = mcp_server._cache
    mcp_server._cache = None
    try:
        with patch.object(
            mcp_server, "_build_store_and_embed",
            side_effect=RuntimeError("bad config"),
        ):
            result = mcp_server._file_recent_impl()
        assert isinstance(result, dict)
        assert result["error"] is True
        assert result["code"] == "service_unavailable"
    finally:
        mcp_server._cache = old_cache


def test_get_deps_failure_in_facets():
    """_file_facets_impl returns structured error when deps fail."""
    old_cache = mcp_server._cache
    mcp_server._cache = None
    try:
        with patch.object(
            mcp_server, "_build_store_and_embed",
            side_effect=RuntimeError("Config not found"),
        ):
            result = mcp_server._file_facets_impl()
        assert isinstance(result, dict)
        assert result["error"] is True
        assert result["code"] == "service_unavailable"
    finally:
        mcp_server._cache = old_cache


def test_get_deps_failure_in_status():
    """_file_status_impl returns structured error when deps fail."""
    old_cache = mcp_server._cache
    mcp_server._cache = None
    try:
        with patch.object(
            mcp_server, "_build_store_and_embed",
            side_effect=RuntimeError("bad config"),
        ):
            result = mcp_server._file_status_impl()
        assert isinstance(result, dict)
        assert result["error"] is True
        assert result["code"] == "service_unavailable"
    finally:
        mcp_server._cache = old_cache


# ---------------------------------------------------------------------------
# _enrich_doc_list helper tests
# ---------------------------------------------------------------------------


def test_enrich_doc_list_adds_mtime_iso():
    """_enrich_doc_list should add mtime_iso field from mtime timestamp."""
    docs = [{"doc_id": "a.md", "mtime": 1700000000.0, "tags": "recipe,korean"}]
    mcp_server._enrich_doc_list(docs)
    assert docs[0]["mtime_iso"] is not None
    assert docs[0]["mtime_iso"].startswith("2023-11-14")
    assert docs[0]["tags"] == ["recipe", "korean"]


def test_enrich_doc_list_none_mtime():
    """_enrich_doc_list should set mtime_iso to None when mtime is missing."""
    docs = [{"doc_id": "a.md", "tags": "test"}]
    mcp_server._enrich_doc_list(docs)
    assert docs[0]["mtime_iso"] is None
    assert docs[0]["tags"] == ["test"]


def test_enrich_doc_list_invalid_mtime():
    """_enrich_doc_list should handle invalid mtime values gracefully."""
    docs = [{"doc_id": "a.md", "mtime": "not-a-number"}]
    mcp_server._enrich_doc_list(docs)
    assert docs[0]["mtime_iso"] is None


def test_enrich_doc_list_no_tags():
    """_enrich_doc_list should not fail when tags is absent."""
    docs = [{"doc_id": "a.md", "mtime": 1700000000.0}]
    mcp_server._enrich_doc_list(docs)
    assert "tags" not in docs[0] or docs[0].get("tags") is None or docs[0].get("tags") == ""


def test_enrich_doc_list_tags_already_list():
    """_enrich_doc_list should not split tags that are already a list."""
    docs = [{"doc_id": "a.md", "mtime": 1.0, "tags": ["a", "b"]}]
    mcp_server._enrich_doc_list(docs)
    assert docs[0]["tags"] == ["a", "b"]


def test_enrich_doc_list_empty_list():
    """_enrich_doc_list should handle empty list without error."""
    docs = []
    mcp_server._enrich_doc_list(docs)
    assert docs == []


def test_get_deps_failure_in_get_chunk():
    """_file_get_chunk_impl returns structured error when deps fail."""
    old_cache = mcp_server._cache
    mcp_server._cache = None
    try:
        with patch.object(
            mcp_server, "_build_store_and_embed",
            side_effect=RuntimeError("bad config"),
        ):
            result = mcp_server._file_get_chunk_impl("x.md", "c:0")
        assert isinstance(result, dict)
        assert result["error"] is True
        assert result["code"] == "service_unavailable"
    finally:
        mcp_server._cache = old_cache


def test_get_deps_failure_in_get_doc_chunks():
    """_file_get_doc_chunks_impl returns structured error when deps fail."""
    old_cache = mcp_server._cache
    mcp_server._cache = None
    try:
        with patch.object(
            mcp_server, "_build_store_and_embed",
            side_effect=RuntimeError("bad config"),
        ):
            result = mcp_server._file_get_doc_chunks_impl("x.md")
        assert isinstance(result, dict)
        assert result["error"] is True
        assert result["code"] == "service_unavailable"
    finally:
        mcp_server._cache = old_cache


# ---------------------------------------------------------------------------
# Index update surfaces failed_docs
# ---------------------------------------------------------------------------


def test_index_update_returns_started_status():
    """_file_index_update_impl must return immediately with status='started'.

    The indexer now runs in a background subprocess so the MCP server stays
    responsive. Callers poll file_status to learn when indexing completes and
    whether it had failures.
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        old_cache = mcp_server._cache
        try:
            with patch("mcp_server.load_config", return_value={"index_root": tmpdir}):
                mcp_server._cache = None
                result = mcp_server._file_index_update_impl(config_path="config.yaml")

            assert result["status"] == "started", (
                f"Expected 'started' (non-blocking), got {result!r}"
            )
            assert "pid" in result
            assert isinstance(result["pid"], int)
        finally:
            mcp_server._cache = old_cache


def test_index_update_no_failures():
    """Alias kept for historical coverage: same contract as test_index_update_returns_started_status."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        old_cache = mcp_server._cache
        try:
            with patch("mcp_server.load_config", return_value={"index_root": tmpdir}):
                mcp_server._cache = None
                result = mcp_server._file_index_update_impl(config_path="config.yaml")

            # Non-blocking: always returns "started", never "completed"
            assert result["status"] == "started"
            assert "failed_count" not in result
        finally:
            mcp_server._cache = old_cache


# ---------------------------------------------------------------------------
# Search diagnostics in response
# ---------------------------------------------------------------------------


def test_search_response_includes_diagnostics():
    """file_search response should include diagnostics dict alongside results."""
    from unittest.mock import MagicMock
    from search_hybrid import SearchResult

    hit = SearchHit(
        doc_id="a.md", loc="c:0", snippet="test snippet", text="test text",
        score=0.9, source_type="md",
    )
    mock_result = SearchResult(
        hits=[hit],
        diagnostics={
            "vector_search_active": True,
            "keyword_search_active": True,
            "reranker_applied": True,
            "degraded": False,
        },
    )

    old_cache = mcp_server._cache
    try:
        mock_store = MagicMock()
        mock_embed = MagicMock()
        mock_config = {"search": {"vector_top_k": 50, "keyword_top_k": 50, "rrf_k": 60, "recency": {}}}
        mcp_server._cache = (mock_store, mock_embed, mock_config)

        with patch("mcp_server.hybrid_search", return_value=mock_result):
            with patch("mcp_server.build_reranker", return_value=None):
                result = mcp_server._file_search_impl("test query")

        assert isinstance(result, dict)
        # Success response must have results+diagnostics but NOT error
        assert "results" in result
        assert "diagnostics" in result
        assert "error" not in result  # distinguishes success from error response
        assert isinstance(result["results"], list)
        assert len(result["results"]) == 1
        assert result["results"][0]["doc_id"] == "a.md"
        assert result["diagnostics"]["vector_search_active"] is True
        assert result["diagnostics"]["keyword_search_active"] is True
        assert result["diagnostics"]["reranker_applied"] is True
        assert result["diagnostics"]["degraded"] is False
    finally:
        mcp_server._cache = old_cache


def test_search_response_degraded_flag():
    """When search is degraded, diagnostics.degraded should be True."""
    from unittest.mock import MagicMock
    from search_hybrid import SearchResult

    hit = SearchHit(
        doc_id="a.md", loc="c:0", snippet="test", text="test", score=0.5,
    )
    mock_result = SearchResult(
        hits=[hit],
        diagnostics={
            "vector_search_active": False,
            "keyword_search_active": False,
            "reranker_applied": False,
            "degraded": True,
        },
    )

    old_cache = mcp_server._cache
    try:
        mock_store = MagicMock()
        mock_embed = MagicMock()
        mock_config = {"search": {"recency": {}}}
        mcp_server._cache = (mock_store, mock_embed, mock_config)

        with patch("mcp_server.hybrid_search", return_value=mock_result):
            with patch("mcp_server.build_reranker", return_value=None):
                result = mcp_server._file_search_impl("test query")

        assert result["diagnostics"]["degraded"] is True
        assert result["diagnostics"]["keyword_search_active"] is False
    finally:
        mcp_server._cache = old_cache


# ---------------------------------------------------------------------------
# file_status health fields
# ---------------------------------------------------------------------------


def test_file_status_includes_health():
    """file_status should include a health section with fts, reranker, and failure info."""
    import json
    import tempfile
    from unittest.mock import MagicMock
    from datetime import datetime, timezone

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write index_metadata.json with 1 failure
        meta = {
            "last_run_at": datetime.now(timezone.utc).isoformat(),
            "doc_count": 10,
            "chunk_count": 50,
            "failed_count": 1,
            "failed_docs": ["broken.pdf"],
        }
        meta_path = Path(tmpdir) / "index_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        old_cache = mcp_server._cache
        try:
            mock_store = MagicMock()
            mock_store.list_doc_ids.return_value = ["a.md", "b.md"]
            mock_store.count_chunks.return_value = 50
            mock_store._metadata_subfields.return_value = {"doc_id", "title"}
            mock_store.fts_available.return_value = True

            mock_config = {
                "index_root": tmpdir,
                "embeddings": {"provider": "openrouter"},
                "search": {
                    "reranker": {
                        "enabled": True,
                        "provider": "deepinfra",
                        "model": "Qwen/Qwen3-Reranker-8B",
                    },
                },
            }
            mcp_server._cache = (mock_store, MagicMock(), mock_config)

            # Mock reranker health check to return 200 (DeepInfra uses httpx.post)
            import httpx
            with patch.object(httpx, "post", return_value=MagicMock(status_code=200)):
                result = mcp_server._file_status_impl()

            assert "health" in result
            health = result["health"]
            assert health["fts_available"] is True
            assert health["reranker_enabled"] is True
            assert health["reranker_responsive"] is True
            assert health["last_index_failed_count"] == 1
        finally:
            mcp_server._cache = old_cache


# ---------------------------------------------------------------------------
# enr_doc_type / enr_topics passthrough
# ---------------------------------------------------------------------------


def test_search_passes_enr_doc_type():
    """_file_search_impl should pass enr_doc_type through to hybrid_search."""
    from unittest.mock import MagicMock, call
    from search_hybrid import SearchResult

    mock_result = SearchResult(hits=[], diagnostics={
        "vector_search_active": True, "keyword_search_active": True,
        "reranker_applied": False, "degraded": False,
    })

    old_cache = mcp_server._cache
    try:
        mock_store = MagicMock()
        mock_embed = MagicMock()
        mock_config = {"search": {"recency": {}}}
        mcp_server._cache = (mock_store, mock_embed, mock_config)

        with patch("mcp_server.hybrid_search", return_value=mock_result) as mock_hs:
            with patch("mcp_server.build_reranker", return_value=None):
                mcp_server._file_search_impl(
                    "test query", enr_doc_type="Geotechnical Report",
                )
        _, kwargs = mock_hs.call_args
        assert kwargs["enr_doc_type"] == "Geotechnical Report"
    finally:
        mcp_server._cache = old_cache


def test_search_passes_enr_topics():
    """_file_search_impl should pass enr_topics through to hybrid_search."""
    from unittest.mock import MagicMock
    from search_hybrid import SearchResult

    mock_result = SearchResult(hits=[], diagnostics={
        "vector_search_active": True, "keyword_search_active": True,
        "reranker_applied": False, "degraded": False,
    })

    old_cache = mcp_server._cache
    try:
        mock_store = MagicMock()
        mock_embed = MagicMock()
        mock_config = {"search": {"recency": {}}}
        mcp_server._cache = (mock_store, mock_embed, mock_config)

        with patch("mcp_server.hybrid_search", return_value=mock_result) as mock_hs:
            with patch("mcp_server.build_reranker", return_value=None):
                mcp_server._file_search_impl(
                    "test query", enr_topics="machine learning,NLP",
                )
        _, kwargs = mock_hs.call_args
        assert kwargs["enr_topics"] == "machine learning,NLP"
    finally:
        mcp_server._cache = old_cache


def test_search_enr_params_default_none():
    """When enr_doc_type/enr_topics are not passed, they should default to None."""
    from unittest.mock import MagicMock
    from search_hybrid import SearchResult

    mock_result = SearchResult(hits=[], diagnostics={
        "vector_search_active": True, "keyword_search_active": True,
        "reranker_applied": False, "degraded": False,
    })

    old_cache = mcp_server._cache
    try:
        mock_store = MagicMock()
        mock_embed = MagicMock()
        mock_config = {"search": {"recency": {}}}
        mcp_server._cache = (mock_store, mock_embed, mock_config)

        with patch("mcp_server.hybrid_search", return_value=mock_result) as mock_hs:
            with patch("mcp_server.build_reranker", return_value=None):
                mcp_server._file_search_impl("test query")
        _, kwargs = mock_hs.call_args
        assert kwargs["enr_doc_type"] is None
        assert kwargs["enr_topics"] is None
    finally:
        mcp_server._cache = old_cache


# ---------------------------------------------------------------------------
# Taxonomy tool contracts
# ---------------------------------------------------------------------------


def test_taxonomy_list_empty_query():
    """_file_taxonomy_list_impl should return list (possibly empty) without error."""
    old_cache = mcp_server._cache
    try:
        mock_store = MagicMock()
        mock_embed = MagicMock()
        mock_config = {"index_root": "/tmp/test", "embeddings": {"provider": "openrouter"}}
        mcp_server._cache = (mock_store, mock_embed, mock_config)

        with patch("core.taxonomy.load_taxonomy_store") as mock_load:
            mock_tax = MagicMock()
            mock_tax.list_by_kind.return_value = [
                {"id": "tag:test", "kind": "tag", "name": "test", "description": "Test tag"},
            ]
            mock_load.return_value = mock_tax
            result = mcp_server._file_taxonomy_list_impl(kind="tag")
        assert isinstance(result, list)
    finally:
        mcp_server._cache = old_cache


def test_taxonomy_get_not_found():
    """_file_taxonomy_get_impl returns error for missing entry."""
    old_cache = mcp_server._cache
    try:
        mock_store = MagicMock()
        mock_embed = MagicMock()
        mock_config = {"index_root": "/tmp/test", "embeddings": {"provider": "openrouter"}}
        mcp_server._cache = (mock_store, mock_embed, mock_config)

        with patch("core.taxonomy.load_taxonomy_store") as mock_load:
            mock_tax = MagicMock()
            mock_tax.get.return_value = None
            mock_load.return_value = mock_tax
            result = mcp_server._file_taxonomy_get_impl("tag:nonexistent")
        assert result["error"] is True
        assert result["code"] == "not_found"
    finally:
        mcp_server._cache = old_cache


def test_taxonomy_add_invalid_kind():
    """_file_taxonomy_add_impl rejects invalid kind."""
    result = mcp_server._file_taxonomy_add_impl("invalid_kind", "test", "desc")
    assert result["error"] is True
    assert result["code"] == "invalid_parameter"


def test_taxonomy_update_empty_id():
    """_file_taxonomy_update_impl rejects empty id."""
    result = mcp_server._file_taxonomy_update_impl("", description="new desc")
    assert result["error"] is True
    assert result["code"] == "invalid_parameter"


def test_taxonomy_delete_empty_id():
    """_file_taxonomy_delete_impl rejects empty id."""
    result = mcp_server._file_taxonomy_delete_impl("")
    assert result["error"] is True
    assert result["code"] == "invalid_parameter"


def test_taxonomy_search_empty_query():
    """_file_taxonomy_search_impl rejects empty query."""
    result = mcp_server._file_taxonomy_search_impl("")
    assert result["error"] is True
    assert result["code"] == "empty_query"


def test_hit_to_dict_includes_taxonomy_fields():
    """_hit_to_dict should include enr_suggested_tags and enr_suggested_folder."""
    hit = SearchHit(
        doc_id="a.md", loc="c:0", snippet="test", text="test text",
        score=0.5,
        enr_suggested_tags="work, finance",
        enr_suggested_folder="Financial/",
    )
    d = mcp_server._hit_to_dict(hit)
    assert d["enr_suggested_tags"] == "work, finance"
    assert d["enr_suggested_folder"] == "Financial/"


def test_file_status_health_reranker_disabled():
    """When reranker is disabled, reranker_responsive should be None."""
    import json
    import tempfile
    from unittest.mock import MagicMock
    from datetime import datetime, timezone

    with tempfile.TemporaryDirectory() as tmpdir:
        meta = {
            "last_run_at": datetime.now(timezone.utc).isoformat(),
            "doc_count": 5,
            "chunk_count": 20,
        }
        meta_path = Path(tmpdir) / "index_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        old_cache = mcp_server._cache
        try:
            mock_store = MagicMock()
            mock_store.list_doc_ids.return_value = ["a.md"]
            mock_store.count_chunks.return_value = 20
            mock_store._metadata_subfields.return_value = {"doc_id"}
            mock_store.fts_available.return_value = False

            mock_config = {
                "index_root": tmpdir,
                "embeddings": {"provider": "openrouter"},
                "search": {"reranker": {"enabled": False}},
            }
            mcp_server._cache = (mock_store, MagicMock(), mock_config)

            result = mcp_server._file_status_impl()

            health = result["health"]
            assert health["fts_available"] is False
            assert health["reranker_enabled"] is False
            assert health["reranker_responsive"] is None
            assert health["last_index_failed_count"] == 0
        finally:
            mcp_server._cache = old_cache
