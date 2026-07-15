"""MCP output contract tests — verify response shapes match documented API.

No external services needed. Uses mocks and direct function calls."""

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.storage import SearchHit
import mcp_server


# ---------------------------------------------------------------------------
# MCP tool schema contract
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_file_search_tool_schema_includes_complex_filter():
    """FastMCP-published file_search schema should expose the complex filter argument."""
    if not mcp_server.HAS_MCP:
        pytest.skip("mcp package not installed")

    tools = await mcp_server.mcp.list_tools()
    file_search = next(tool for tool in tools if tool.name == "file_search")

    assert "filter" in file_search.inputSchema["properties"]
    assert file_search.inputSchema["properties"]["filter"]["title"] == "Filter"
    assert "return" in file_search.inputSchema["properties"]
    assert "return_mode" not in file_search.inputSchema["properties"]
    assert file_search.inputSchema["properties"]["return"]["default"] == "slim"
    assert "content_max_character" in file_search.inputSchema["properties"]
    assert "Supported operators: eq, ne, contains, prefix, in, and, or, not" in (
        file_search.description or ""
    )


@pytest.mark.anyio
async def test_file_search_tool_dispatch_maps_return_alias():
    """FastMCP should accept MCP arg 'return' and call implementation with return_mode."""
    if not mcp_server.HAS_MCP:
        pytest.skip("mcp package not installed")

    with patch("mcp_server._file_search_impl", return_value={"results": [], "diagnostics": {}}) as mock:
        await mcp_server.mcp.call_tool(
            "file_search",
            {"query": "lease application", "return": "full", "content_max_character": 123},
        )

    assert mock.call_args.kwargs["return_mode"] == "full"
    assert mock.call_args.kwargs["content_max_character"] == 123


@pytest.mark.anyio
async def test_file_index_document_dispatches_via_to_thread():
    """Single-doc indexing must leave the event loop free for /health."""
    if not mcp_server.HAS_MCP:
        pytest.skip("mcp package not installed")

    async def fake_to_thread(func, /, *args, **kwargs):
        return {"func": func, "args": args, "kwargs": kwargs}

    with patch(
        "mcp_server._file_index_document_impl",
        return_value={"status": "indexed", "doc_id": "documents::00abc"},
    ) as mock_impl, patch("asyncio.to_thread", side_effect=fake_to_thread) as mock_to_thread:
        await mcp_server.mcp.call_tool(
            "file_index_document",
            {"target": "email-attachments/x@00abc@.pdf", "source_name": "documents", "force": True},
        )

    assert mock_to_thread.await_count == 1
    mock_impl.assert_not_called()
    _, kwargs = mock_to_thread.await_args
    assert kwargs == {
        "target": "email-attachments/x@00abc@.pdf",
        "source_name": "documents",
        "force": True,
    }


# ---------------------------------------------------------------------------
# Slim default + top_k aliases (Hermes TICKET-docorganizer-filesearch-default-slim)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_file_search_schema_slim_default_and_topk_aliases():
    """Default return mode is 'slim', default top_k is 8, and the common
    result-count aliases are first-class schema params (not silently ignored)."""
    if not mcp_server.HAS_MCP:
        pytest.skip("mcp package not installed")

    tools = await mcp_server.mcp.list_tools()
    file_search = next(tool for tool in tools if tool.name == "file_search")
    props = file_search.inputSchema["properties"]

    assert props["return"]["default"] == "slim"
    assert props["top_k"]["default"] == 8
    for alias in ("k", "n", "max_results", "num_results", "limit"):
        assert alias in props, f"alias {alias!r} missing from schema"


@pytest.mark.anyio
async def test_file_search_dispatch_maps_topk_aliases():
    """k / n / max_results / num_results / limit act as top_k aliases; an
    explicit top_k wins over an alias."""
    if not mcp_server.HAS_MCP:
        pytest.skip("mcp package not installed")

    for alias in ("k", "n", "max_results", "num_results", "limit"):
        with patch(
            "mcp_server._file_search_impl",
            return_value={"results": [], "diagnostics": {}},
        ) as mock:
            await mcp_server.mcp.call_tool("file_search", {"query": "x", alias: 3})
        assert mock.call_args.kwargs["top_k"] == 3, f"alias {alias!r} not honored"

    with patch(
        "mcp_server._file_search_impl",
        return_value={"results": [], "diagnostics": {}},
    ) as mock:
        await mcp_server.mcp.call_tool(
            "file_search", {"query": "x", "top_k": 5, "k": 3}
        )
    assert mock.call_args.kwargs["top_k"] == 5


def _comm_hit(**extra):
    meta = {
        "sender": "Lee Donnelly",
        "sent_at": "2026-07-01 15:04:00+00:00",
        "channel_name": "Lee Donnelly (+1908...)",
        "source_message_id": "AC123abc",
        "direction": "inbound",
        **extra,
    }
    return SearchHit(
        doc_id="comm_messages::quo/AC123abc",
        loc="c:0",
        snippet="Hi, is the 2BR still available?",
        text="Hi, is the 2BR still available?",
        score=0.91,
        source_type="pg_message",
        title="AC123abc",
        rel_path="quo/AC123abc",
        tags="",
        extra_metadata=meta,
    )


def test_slim_hit_comm_message_flat_with_direction():
    """Slim shape for a comm hit: flat sent_at/channel/sender/direction/
    source_id, no verbose empties, no nested custom_meta, no content blob.
    title/rel_path that merely repeat the message id are omitted too."""
    d = mcp_server._slim_hit_to_dict(_comm_hit())

    assert d["doc_id"] == "comm_messages::quo/AC123abc"
    assert d["snippet"] == "Hi, is the 2BR still available?"
    assert d["direction"] == "inbound"
    assert d["sender"] == "Lee Donnelly"
    assert d["sent_at"] == "2026-07-01 15:04:00+00:00"
    assert d["channel"] == "Lee Donnelly (+1908...)"
    assert d["source_id"] == "AC123abc"

    for verbose_key in (
        "tags", "keywords", "description", "author", "section", "size",
        "status", "custom_meta", "content", "content_truncated",
        "enr_summary", "enr_topics",
    ):
        assert verbose_key not in d, f"{verbose_key!r} should not be in slim output"
    # redundant identity echoes dropped: title == source_id, rel_path in doc_id
    assert "title" not in d
    assert "rel_path" not in d


def test_slim_hit_document_keeps_title_and_path_omits_comm_fields():
    """Slim shape for a non-comm document: title/rel_path present, comm
    fields and empty values omitted entirely."""
    hit = SearchHit(
        doc_id="documents::00001",
        loc="p:0:c:1",
        snippet="Lease agreement for 12 Main St...",
        text="Lease agreement for 12 Main St...",
        score=0.88,
        source_type="pdf",
        title="Lease Agreement",
        rel_path="Projects/lease@00001@.pdf",
        extra_metadata={},
    )
    d = mcp_server._slim_hit_to_dict(hit)

    assert d["title"] == "Lease Agreement"
    assert d["rel_path"] == "Projects/lease@00001@.pdf"
    for comm_key in ("direction", "sender", "sent_at", "channel", "source_id"):
        assert comm_key not in d


def test_file_search_impl_slim_is_default_return_mode():
    """_file_search_impl defaults to slim output."""
    fake_result = MagicMock()
    fake_result.hits = [_comm_hit()]
    fake_result.diagnostics = {}
    with patch("mcp_server._get_deps", return_value=(MagicMock(), MagicMock(), {})), \
         patch("mcp_server.build_reranker", return_value=None), \
         patch("mcp_server.hybrid_search", return_value=fake_result):
        out = mcp_server._file_search_impl(query="lee donnelly")
    assert "results" in out
    row = out["results"][0]
    assert row.get("direction") == "inbound"
    assert "content" not in row


# ---------------------------------------------------------------------------
# sort="recent" — one-call "did we respond?" (Hermes follow-up, #0110)
# ---------------------------------------------------------------------------


def _timed_hit(doc_id: str, mtime: float, direction: str) -> SearchHit:
    return SearchHit(
        doc_id=doc_id,
        loc="c:0",
        snippet=f"message {doc_id}",
        text=f"message {doc_id}",
        score=0.5,
        mtime=mtime,
        source_type="pg_message",
        extra_metadata={
            "sent_at": f"2026-07-{int(mtime):02d} 12:00:00+00:00",
            "direction": direction,
            "sender": "X",
            "source_message_id": doc_id.rsplit("/", 1)[-1],
        },
    )


def _search_ctx(fake_result):
    return (
        patch("mcp_server._get_deps", return_value=(MagicMock(), MagicMock(), {})),
        patch("mcp_server.build_reranker", return_value="RERANKER"),
        patch("mcp_server.hybrid_search", return_value=fake_result),
    )


def test_file_search_sort_recent_orders_newest_first_and_truncates():
    """sort='recent' returns the query-scoped pool sorted by mtime desc,
    truncated to top_k — even when relevance order differs."""
    # relevance order: oldest first (worst case for a recency verdict)
    hits = [
        _timed_hit("comm_messages::quo/a", 1.0, "inbound"),
        _timed_hit("comm_messages::quo/b", 3.0, "outbound"),
        _timed_hit("comm_messages::quo/c", 2.0, "inbound"),
        _timed_hit("comm_messages::quo/d", 5.0, "outbound"),
        _timed_hit("comm_messages::quo/e", 4.0, "inbound"),
    ]
    fake_result = MagicMock()
    fake_result.hits = hits
    fake_result.diagnostics = {}
    deps, rr, hs = _search_ctx(fake_result)
    with deps, rr, hs as mock_search:
        out = mcp_server._file_search_impl(query="prospect", sort="recent", top_k=3)

    ids = [r["doc_id"] for r in out["results"]]
    assert ids == ["comm_messages::quo/d", "comm_messages::quo/e", "comm_messages::quo/b"]
    assert [r["direction"] for r in out["results"]] == ["outbound", "inbound", "outbound"]
    # pool widened beyond top_k and reranker skipped (its order is discarded)
    assert mock_search.call_args.kwargs["final_top_k"] >= 50
    assert mock_search.call_args.kwargs["reranker"] is None
    assert out["diagnostics"]["sort"] == "recent"


def test_file_search_sort_default_keeps_relevance_order():
    """Without sort, relevance order is untouched and the reranker is used."""
    hits = [
        _timed_hit("comm_messages::quo/a", 1.0, "inbound"),
        _timed_hit("comm_messages::quo/b", 3.0, "outbound"),
    ]
    fake_result = MagicMock()
    fake_result.hits = hits
    fake_result.diagnostics = {}
    deps, rr, hs = _search_ctx(fake_result)
    with deps, rr, hs as mock_search:
        out = mcp_server._file_search_impl(query="prospect", top_k=2)

    assert [r["doc_id"] for r in out["results"]] == [
        "comm_messages::quo/a", "comm_messages::quo/b",
    ]
    assert mock_search.call_args.kwargs["reranker"] == "RERANKER"
    assert mock_search.call_args.kwargs["final_top_k"] == 2


def test_file_search_sort_invalid_errors_not_ignored():
    """An unsupported sort value must error loudly, never be silently ignored."""
    out = mcp_server._file_search_impl(query="x", sort="alphabetical")
    assert out.get("error") is True
    assert out["code"] == "invalid_parameter"


# ---------------------------------------------------------------------------
# diagnostics opt-in (Hermes follow-up: block measured larger than the data)
# ---------------------------------------------------------------------------


def test_file_search_impl_diagnostics_opt_out_drops_block():
    """include_diagnostics=False returns results only — no diagnostics key."""
    fake_result = MagicMock()
    fake_result.hits = [_comm_hit()]
    fake_result.diagnostics = {"degraded": False, "candidate_counts": {"vector": 50}}
    deps, rr, hs = _search_ctx(fake_result)
    with deps, rr, hs:
        out = mcp_server._file_search_impl(query="x", include_diagnostics=False)
    assert "diagnostics" not in out
    assert "degraded" not in out
    assert len(out["results"]) == 1


def test_file_search_impl_diagnostics_opt_out_keeps_degraded_flag():
    """Even with diagnostics off, a degraded pipeline must stay visible —
    silent degradation is how bad results get trusted (#0106 lesson)."""
    fake_result = MagicMock()
    fake_result.hits = [_comm_hit()]
    fake_result.diagnostics = {"degraded": True}
    deps, rr, hs = _search_ctx(fake_result)
    with deps, rr, hs:
        out = mcp_server._file_search_impl(query="x", include_diagnostics=False)
    assert "diagnostics" not in out
    assert out["degraded"] is True


def test_file_search_impl_diagnostics_default_on_for_internal_callers():
    """Impl-level default keeps diagnostics for internal/REST callers
    (api_server passes payload kwargs straight through)."""
    fake_result = MagicMock()
    fake_result.hits = [_comm_hit()]
    fake_result.diagnostics = {"degraded": False}
    deps, rr, hs = _search_ctx(fake_result)
    with deps, rr, hs:
        out = mcp_server._file_search_impl(query="x")
    assert "diagnostics" in out


@pytest.mark.anyio
async def test_file_search_tool_diagnostics_off_by_default():
    """MCP tool layer defaults include_diagnostics to False (hot-path lean)."""
    if not mcp_server.HAS_MCP:
        pytest.skip("mcp package not installed")

    tools = await mcp_server.mcp.list_tools()
    fs = next(t for t in tools if t.name == "file_search")
    assert fs.inputSchema["properties"]["include_diagnostics"]["default"] is False

    with patch(
        "mcp_server._file_search_impl",
        return_value={"results": []},
    ) as mock:
        await mcp_server.mcp.call_tool("file_search", {"query": "x"})
    assert mock.call_args.kwargs["include_diagnostics"] is False

    with patch(
        "mcp_server._file_search_impl",
        return_value={"results": []},
    ) as mock:
        await mcp_server.mcp.call_tool(
            "file_search", {"query": "x", "include_diagnostics": True}
        )
    assert mock.call_args.kwargs["include_diagnostics"] is True


@pytest.mark.anyio
async def test_file_search_dispatch_sort_and_order_by_alias():
    """MCP layer: sort passes through; order_by acts as an alias of sort."""
    if not mcp_server.HAS_MCP:
        pytest.skip("mcp package not installed")

    with patch(
        "mcp_server._file_search_impl",
        return_value={"results": [], "diagnostics": {}},
    ) as mock:
        await mcp_server.mcp.call_tool("file_search", {"query": "x", "sort": "recent"})
    assert mock.call_args.kwargs["sort"] == "recent"

    with patch(
        "mcp_server._file_search_impl",
        return_value={"results": [], "diagnostics": {}},
    ) as mock:
        await mcp_server.mcp.call_tool("file_search", {"query": "x", "order_by": "recent"})
    assert mock.call_args.kwargs["sort"] == "recent"


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


def test_build_store_and_embed_uses_recovery_open():
    config = {"index_root": "/tmp/doc-index", "lancedb": {"table": "chunks"}}
    sentinel_store = object()
    sentinel_embed = object()

    with patch.object(mcp_server, "load_config", return_value=config):
        with patch.object(
            mcp_server,
            "open_store_with_recovery",
            return_value=sentinel_store,
        ) as open_store:
            with patch.object(
                mcp_server,
                "build_embed_provider",
                return_value=sentinel_embed,
            ):
                store, embed, loaded = mcp_server._build_store_and_embed()

    assert store is sentinel_store
    assert embed is sentinel_embed
    assert loaded is config
    open_store.assert_called_once_with(
        Path("/tmp/doc-index"),
        "chunks",
        logger_obj=mcp_server.logger,
        auto_recover=True,
    )


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

    class CompletedProc:
        pid = 424260

        def poll(self):
            return 0

    with tempfile.TemporaryDirectory() as tmpdir:
        old_cache = mcp_server._cache
        try:
            with patch("mcp_server.load_config", return_value={"index_root": tmpdir}), patch(
                "subprocess.Popen", return_value=CompletedProc()
            ):
                mcp_server._cache = None
                result = mcp_server._file_index_update_impl(config_path="config.yaml")

            assert result["status"] == "started", (
                f"Expected 'started' (non-blocking), got {result!r}"
            )
            assert "pid" in result
            assert isinstance(result["pid"], int)
            supervisor = mcp_server._get_index_run_supervisor({"index_root": tmpdir})
            deadline = time.time() + 2
            while supervisor.status_summary()["current"] is not None and time.time() < deadline:
                time.sleep(0.01)
        finally:
            mcp_server._cache = old_cache


def test_index_update_no_failures():
    """Alias kept for historical coverage: same contract as test_index_update_returns_started_status."""
    import tempfile

    class CompletedProc:
        pid = 424261

        def poll(self):
            return 0

    with tempfile.TemporaryDirectory() as tmpdir:
        old_cache = mcp_server._cache
        try:
            with patch("mcp_server.load_config", return_value={"index_root": tmpdir}), patch(
                "subprocess.Popen", return_value=CompletedProc()
            ):
                mcp_server._cache = None
                result = mcp_server._file_index_update_impl(config_path="config.yaml")

            # Non-blocking: always returns "started", never "completed"
            assert result["status"] == "started"
            assert "failed_count" not in result
            supervisor = mcp_server._get_index_run_supervisor({"index_root": tmpdir})
            deadline = time.time() + 2
            while supervisor.status_summary()["current"] is not None and time.time() < deadline:
                time.sleep(0.01)
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


def test_source_scoped_empty_degraded_search_fails_closed(tmp_path):
    """Source-scoped degraded empty results are inconclusive, not proof of absence."""
    from search_hybrid import SearchResult

    mock_result = SearchResult(
        hits=[],
        diagnostics={
            "vector_search_active": True,
            "keyword_search_active": False,
            "reranker_applied": True,
            "degraded": True,
        },
    )

    old_cache = mcp_server._cache
    try:
        mock_store = MagicMock()
        mock_embed = MagicMock()
        mock_config = {"index_root": str(tmp_path), "search": {"recency": {}}}
        mcp_server._cache = (mock_store, mock_embed, mock_config)

        with patch("mcp_server.hybrid_search", return_value=mock_result):
            with patch("mcp_server.build_reranker", return_value=None):
                result = mcp_server._file_search_impl("rent payment", source_name="sor")

        assert result["error"] is True
        assert result["code"] == "source_search_degraded"
        assert "results" not in result
        assert "diagnostics" not in result
    finally:
        mcp_server._cache = old_cache


def test_file_search_compact_mode_keeps_source_content_and_excludes_bloat():
    """return='compact' response should keep source content without raw node bloat."""
    from search_hybrid import SearchResult

    hit = SearchHit(
        doc_id="sor::task/400",
        loc="c:0",
        snippet="Lease switch: Erika Pruitt...",
        text="",
        score=0.9,
        source_type="sor_task",
        title="Lease switch: Erika Pruitt",
        tags="lease,zillow",
        folder="task",
        status="Active-High Priority",
        custom_meta='{"source_id":"zillow-123"}',
        enr_summary="Lease switch summary.",
        enr_key_facts="Fee paid, lease pending",
        enr_suggested_folder="Real Estate/Applications/" * 200,
        extra_metadata={
            "_node_content": (
                '{"embedding": null, "metadata": {"created": "2026-05-06", '
                '"mtime": 1778092369.0, "size": 980, "section": "Node Section"}, '
                '"text": "Details:\\nsource line item source line item source line item"}'
            ),
            "embedding": None,
            "relationships": {"source": "previous"},
            "source_name": "sor",
            "document_id": "task/400",
            "owner": "Dan",
            "section": "Details",
        },
    )
    mock_result = SearchResult(hits=[hit], diagnostics={"degraded": False})

    old_cache = mcp_server._cache
    try:
        mcp_server._cache = (MagicMock(), MagicMock(), {"search": {"recency": {}}})
        with patch("mcp_server.hybrid_search", return_value=mock_result):
            with patch("mcp_server.build_reranker", return_value=None):
                result = mcp_server._file_search_impl("lease", return_mode="compact", content_max_character=25)
    finally:
        mcp_server._cache = old_cache

    row = result["results"][0]
    assert row["doc_id"] == "sor::task/400"
    assert row["source_name"] == "sor"
    assert row["document_id"] == "task/400"
    assert row["owner"] == "Dan"
    assert row["section"] == "Details"
    assert row["created"] == "2026-05-06"
    assert row["mtime"] == 1778092369.0
    assert row["size"] == 980
    assert row["content"] == "Details:\nsource line item"
    assert row["content_truncated"] is True
    assert row["enr_summary"] == "Lease switch summary."
    assert row["enr_key_facts"] == "Fee paid, lease pending"
    assert "enr_suggested_folder" not in row
    assert "_node_content" not in row
    assert "embedding" not in row
    assert "relationships" not in row


def test_file_search_full_return_preserves_existing_verbose_fields():
    """Full search response should preserve current verbose field behavior."""
    from search_hybrid import SearchResult

    hit = SearchHit(
        doc_id="sor::task/400",
        loc="c:0",
        snippet="Lease switch: Erika Pruitt...",
        text="Details: full source line item",
        score=0.9,
        enr_suggested_folder="Real Estate/Applications/",
        extra_metadata={
            "_node_content": '{"embedding": null, "text": "wrapped source"}',
            "source_name": "sor",
        },
    )
    mock_result = SearchResult(hits=[hit], diagnostics={"degraded": False})

    old_cache = mcp_server._cache
    try:
        mcp_server._cache = (MagicMock(), MagicMock(), {"search": {"recency": {}}})
        with patch("mcp_server.hybrid_search", return_value=mock_result):
            with patch("mcp_server.build_reranker", return_value=None):
                result = mcp_server._file_search_impl("lease", return_mode="full")
    finally:
        mcp_server._cache = old_cache

    row = result["results"][0]
    assert row["_node_content"] == '{"embedding": null, "text": "wrapped source"}'
    assert row["enr_suggested_folder"] == "Real Estate/Applications/"
    assert row["source_name"] == "sor"
    assert "content" not in row


def test_file_search_invalid_return_mode_is_structured_error():
    """Unknown return mode should fail before search runs."""
    result = mcp_server._file_search_impl("lease", return_mode="verbose")

    assert result["error"] is True
    assert result["code"] == "invalid_parameter"


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
            "warning_count": 3,
            "warning_counts": {
                "enrichment_failed": 2,
                "fts_rebuild_failed": 1,
            },
            "enrichment_failed_count": 2,
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
            assert health["last_index_warning_count"] == 3
            assert health["last_index_warning_counts"] == {
                "enrichment_failed": 2,
                "fts_rebuild_failed": 1,
            }
            assert health["last_enrichment_failed_count"] == 2
        finally:
            mcp_server._cache = old_cache


def test_file_status_includes_cached_deep_health_source_coverage(tmp_path):
    """Deep health check should cache deterministic source coverage for 10 minutes."""
    import json
    from doc_id_store import DocIDStore

    (tmp_path / "index_metadata.json").write_text(
        json.dumps({"last_run_at": "2026-05-28T18:00:00Z", "failed_count": 0})
    )
    registry = DocIDStore(tmp_path / "doc_registry.db")
    registry.register("documents::doc-1", "doc-1.md", source_name="documents")
    registry.register("sor::task-1", "task/1", source_name="sor")
    registry.close()

    old_cache = mcp_server._cache
    old_signature = mcp_server._cache_index_signature
    old_identity = mcp_server._cache_identity
    old_deep_cache = mcp_server._deep_health_cache
    try:
        mock_store = MagicMock()
        mock_store.list_doc_ids.side_effect = [
            ["documents::doc-1"],
            ["documents::doc-1", "sor::task-1"],
        ]
        mock_store.count_chunks.return_value = 2
        mock_store._metadata_subfields.return_value = {"doc_id", "source_name"}
        mock_store.fts_available.return_value = True

        mcp_server._cache = (
            mock_store,
            MagicMock(),
            {
                "index_root": str(tmp_path),
                "embeddings": {"provider": "openrouter"},
                "search": {"reranker": {"enabled": False}},
                "sources": [
                    {"type": "filesystem", "name": "documents"},
                    {"type": "postgres", "name": "sor"},
                ],
            },
        )
        mcp_server._cache_index_signature = mcp_server._index_metadata_signature(mcp_server._cache[2])
        mcp_server._cache_identity = id(mcp_server._cache)
        mcp_server._deep_health_cache = None

        first = mcp_server._file_status_impl()
        second = mcp_server._file_status_impl()

        first_deep = first["health"]["deep_check"]
        second_deep = second["health"]["deep_check"]
        assert first_deep["cached"] is False
        assert first_deep["ttl_seconds"] == 600
        assert first_deep["uses_llm"] is False
        assert first_deep["last_ran_at"]
        assert first_deep["sources"]["documents"]["status"] == "ok"
        assert first_deep["sources"]["sor"]["status"] == "critical"
        assert first_deep["sources"]["sor"]["registry_doc_count"] == 1
        assert first_deep["sources"]["sor"]["index_doc_count"] == 0
        assert first_deep["overall"] == "critical"

        assert second_deep["cached"] is True
        assert second_deep["last_ran_at"] == first_deep["last_ran_at"]
        assert second_deep["sources"]["sor"]["status"] == "critical"
    finally:
        mcp_server._cache = old_cache
        mcp_server._cache_index_signature = old_signature
        mcp_server._cache_identity = old_identity
        mcp_server._deep_health_cache = old_deep_cache


def test_file_status_persists_deep_health_snapshot_for_restart_cache(tmp_path):
    """Deep health cache should survive process restart via index_health.json."""
    import json
    from doc_id_store import DocIDStore

    (tmp_path / "index_metadata.json").write_text(
        json.dumps({"last_run_at": "2026-05-28T18:00:00Z", "failed_count": 0})
    )
    registry = DocIDStore(tmp_path / "doc_registry.db")
    registry.register("documents::doc-1", "doc-1.md", source_name="documents")
    registry.register("sor::task-1", "task/1", source_name="sor")
    registry.close()

    config = {
        "index_root": str(tmp_path),
        "embeddings": {"provider": "openrouter"},
        "search": {"reranker": {"enabled": False}},
        "sources": [
            {"type": "filesystem", "name": "documents"},
            {"type": "postgres", "name": "sor"},
        ],
    }
    old_cache = mcp_server._cache
    old_signature = mcp_server._cache_index_signature
    old_identity = mcp_server._cache_identity
    old_deep_cache = mcp_server._deep_health_cache
    try:
        first_store = MagicMock()
        first_store.list_doc_ids.return_value = ["documents::doc-1"]
        first_store.count_chunks.return_value = 1
        first_store._metadata_subfields.return_value = {"doc_id", "source_name"}
        first_store.fts_available.return_value = True
        mcp_server._cache = (first_store, MagicMock(), config)
        mcp_server._cache_index_signature = mcp_server._index_metadata_signature(config)
        mcp_server._cache_identity = id(mcp_server._cache)
        mcp_server._deep_health_cache = None

        first = mcp_server._file_status_impl()
        health_path = tmp_path / "index_health.json"

        assert health_path.exists()
        persisted = json.loads(health_path.read_text())
        assert persisted["payload"]["last_ran_at"] == first["health"]["deep_check"]["last_ran_at"]

        second_store = MagicMock()
        second_store.list_doc_ids.return_value = ["documents::doc-1", "sor::task-1"]
        second_store.count_chunks.return_value = 2
        second_store._metadata_subfields.return_value = {"doc_id", "source_name"}
        second_store.fts_available.return_value = True
        mcp_server._cache = (second_store, MagicMock(), config)
        mcp_server._cache_index_signature = mcp_server._index_metadata_signature(config)
        mcp_server._cache_identity = id(mcp_server._cache)
        mcp_server._deep_health_cache = None

        second = mcp_server._file_status_impl()
        second_deep = second["health"]["deep_check"]
        assert second_deep["cached"] is True
        assert second_deep["last_ran_at"] == first["health"]["deep_check"]["last_ran_at"]
        assert second_deep["sources"]["sor"]["status"] == "critical"
    finally:
        mcp_server._cache = old_cache
        mcp_server._cache_index_signature = old_signature
        mcp_server._cache_identity = old_identity
        mcp_server._deep_health_cache = old_deep_cache


def test_file_status_deep_health_includes_source_freshness_fields(tmp_path):
    """Deep health should report latest registry and indexed timestamps per source."""
    import json
    from doc_id_store import DocIDStore

    (tmp_path / "index_metadata.json").write_text(
        json.dumps({"last_run_at": "2026-05-28T18:00:00Z", "failed_count": 0})
    )
    registry = DocIDStore(tmp_path / "doc_registry.db")
    registry.register("documents::doc-1", "doc-1.md", source_name="documents")
    registry.register("sor::task-1", "task/1", source_name="sor")
    registry._conn.execute(
        "UPDATE doc_registry SET created = 1500, first_seen_at = 1500, last_seen_at = 1500 "
        "WHERE source_name = 'sor'"
    )
    registry._conn.commit()
    registry.close()

    old_cache = mcp_server._cache
    old_signature = mcp_server._cache_index_signature
    old_identity = mcp_server._cache_identity
    old_deep_cache = mcp_server._deep_health_cache
    try:
        mock_store = MagicMock()
        mock_store.list_doc_ids.return_value = ["documents::doc-1", "sor::task-1"]
        mock_store.count_chunks.return_value = 2
        mock_store.list_recent_docs.return_value = [
            {"doc_id": "documents::doc-1", "mtime": 1000.0},
            {"doc_id": "sor::task-1", "mtime": 2000.0},
        ]
        mock_store._metadata_subfields.return_value = {"doc_id", "source_name", "mtime"}
        mock_store.fts_available.return_value = True

        mcp_server._cache = (
            mock_store,
            MagicMock(),
            {
                "index_root": str(tmp_path),
                "embeddings": {"provider": "openrouter"},
                "search": {"reranker": {"enabled": False}},
                "sources": [{"type": "postgres", "name": "sor"}],
            },
        )
        mcp_server._cache_index_signature = mcp_server._index_metadata_signature(mcp_server._cache[2])
        mcp_server._cache_identity = id(mcp_server._cache)
        mcp_server._deep_health_cache = None

        result = mcp_server._file_status_impl()
        sor = result["health"]["deep_check"]["sources"]["sor"]

        assert sor["registry_latest_seen_at"] == "1970-01-01T00:25:00+00:00"
        assert sor["index_latest_mtime"] == 2000.0
        assert sor["index_latest_mtime_iso"] == "1970-01-01T00:33:20+00:00"
        assert result["health"]["deep_check"]["checks"]["index_freshness_available"] is True
    finally:
        mcp_server._cache = old_cache
        mcp_server._cache_index_signature = old_signature
        mcp_server._cache_identity = old_identity
        mcp_server._deep_health_cache = old_deep_cache


def test_registry_source_stats_dedupes_legacy_and_namespaced_document_ids(tmp_path):
    """Registry health should count canonical doc IDs, not both legacy and namespaced rows."""
    from doc_id_store import DocIDStore

    registry = DocIDStore(tmp_path / "doc_registry.db")
    registry.register("0003D", "docs/a@0003D@.pdf", source_name="documents")
    registry.register("documents::0003D", "docs/a@0003D@.pdf", source_name="documents")
    registry.register("sor::task/1", "task/1", source_name="sor")
    registry.close()

    stats, error = mcp_server._registry_source_stats(tmp_path)

    assert error is None
    assert stats["documents"]["doc_count"] == 1
    assert stats["documents"]["raw_doc_count"] == 2
    assert stats["documents"]["duplicate_doc_count"] == 1
    assert stats["sor"]["doc_count"] == 1
    assert stats["sor"]["raw_doc_count"] == 1
    assert stats["sor"]["duplicate_doc_count"] == 0


def test_file_status_deep_health_treats_no_text_docs_as_processed(tmp_path):
    """Docs that were scanned but produced no text should not look like index loss."""
    import json
    from doc_id_store import DocIDStore

    (tmp_path / "index_metadata.json").write_text(
        json.dumps({"last_run_at": "2026-05-30T07:00:00Z", "failed_count": 0})
    )
    (tmp_path / "indexer.log").write_text(
        "2026-05-30 07:00:01,000 WARNING prefect.task_runs: "
        "No text extracted: documents::sidecar\n"
    )
    registry = DocIDStore(tmp_path / "doc_registry.db")
    registry.register("documents::indexed", "docs/indexed.md", source_name="documents")
    registry.register("documents::sidecar", "docs/sidecar.json", source_name="documents")
    registry.close()

    old_cache = mcp_server._cache
    old_signature = mcp_server._cache_index_signature
    old_identity = mcp_server._cache_identity
    old_deep_cache = mcp_server._deep_health_cache
    try:
        mock_store = MagicMock()
        mock_store.list_doc_ids.return_value = ["documents::indexed"]
        mock_store.count_chunks.return_value = 1
        mock_store.list_recent_docs.return_value = [{"doc_id": "documents::indexed", "mtime": 1000.0}]
        mock_store._metadata_subfields.return_value = {"doc_id", "source_name", "mtime"}
        mock_store.fts_available.return_value = True

        mcp_server._cache = (
            mock_store,
            MagicMock(),
            {
                "index_root": str(tmp_path),
                "embeddings": {"provider": "openrouter"},
                "search": {"reranker": {"enabled": False}},
                "sources": [{"type": "filesystem", "name": "documents"}],
            },
        )
        mcp_server._cache_index_signature = mcp_server._index_metadata_signature(mcp_server._cache[2])
        mcp_server._cache_identity = id(mcp_server._cache)
        mcp_server._deep_health_cache = None

        result = mcp_server._file_status_impl()
        documents = result["health"]["deep_check"]["sources"]["documents"]

        assert documents["status"] == "ok"
        assert documents["reason"] == "indexed_or_not_extractable"
        assert documents["not_extractable_doc_count"] == 1
        assert documents["unindexed_registry_doc_count"] == 0
    finally:
        mcp_server._cache = old_cache
        mcp_server._cache_index_signature = old_signature
        mcp_server._cache_identity = old_identity
        mcp_server._deep_health_cache = old_deep_cache


def test_recent_provider_failures_classifies_openrouter_403_logs(tmp_path):
    """Recent OpenRouter 403s should become loud provider health failures."""
    from datetime import datetime, timezone

    now = datetime(2026, 5, 29, 18, 0, 0, tzinfo=timezone.utc).timestamp()
    (tmp_path / "indexer.log").write_text(
        "\n".join(
            [
                "2026-05-28 17:59:59,000 ERROR providers.embed.openrouter_embed: "
                "OpenRouter embedding API error: 403 old failure outside window",
                "2026-05-29 17:40:00,000 INFO httpx: HTTP Request: POST "
                'https://openrouter.ai/api/v1/embeddings "HTTP/1.1 200 OK"',
                "2026-05-29 17:41:31,383 ERROR providers.embed.openrouter_embed: "
                "OpenRouter embedding API error: 403 "
                '{"error":{"message":"Key limit exceeded (monthly limit)","code":403}}',
                "2026-05-29 17:41:31,493 ERROR providers.llm.openrouter_llm: "
                "OpenRouter API error: 403 "
                '{"error":{"message":"Key limit exceeded (monthly limit)","code":403}}',
            ]
        )
    )

    result = mcp_server._recent_provider_failures(tmp_path, now=now)

    assert result["status"] == "critical"
    assert result["total_count"] == 2
    assert result["lookback_seconds"] == 86400
    assert set(result["by_key"]) == {"openrouter_embeddings", "openrouter_chat"}
    assert result["by_key"]["openrouter_embeddings"]["http_status"] == 403
    assert result["by_key"]["openrouter_embeddings"]["severity"] == "critical"
    assert result["by_key"]["openrouter_embeddings"]["last_seen_at"] == "2026-05-29T17:41:31+00:00"
    assert "Key limit exceeded" in result["by_key"]["openrouter_embeddings"]["sample"]
    assert result["by_key"]["openrouter_chat"]["operation"] == "chat"


def test_recent_provider_failures_marks_transient_dns_recovered_after_success(tmp_path):
    """A retryable provider failure followed by 200 OK should not stay critical."""
    from datetime import datetime, timezone

    now = datetime(2026, 6, 1, 21, 0, 0, tzinfo=timezone.utc).timestamp()
    (tmp_path / "indexer.log").write_text(
        "\n".join(
            [
                "2026-06-01 20:37:20,319 WARNING providers.llm.openrouter_llm: "
                "generate() attempt 1/2 failed (ConnectError: [Errno -3] "
                "Temporary failure in name resolution), retrying in 5s...",
                "2026-06-01 20:37:20,324 WARNING providers.embed.openrouter_embed: "
                "embed attempt 1/2 failed (ConnectError: [Errno -3] "
                "Temporary failure in name resolution), retrying in 5s...",
                "2026-06-01 20:37:25,001 INFO httpx: HTTP Request: POST "
                'https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"',
                "2026-06-01 20:37:25,002 INFO httpx: HTTP Request: POST "
                'https://openrouter.ai/api/v1/embeddings "HTTP/1.1 200 OK"',
            ]
        )
    )

    result = mcp_server._recent_provider_failures(tmp_path, now=now)

    assert result["status"] == "ok"
    assert result["total_count"] == 2
    assert result["recovered_count"] == 2
    assert result["by_key"]["openrouter_chat"]["recovered"] is True
    assert result["by_key"]["openrouter_chat"]["severity"] == "ok"
    assert result["by_key"]["openrouter_embeddings"]["recovered"] is True
    assert result["by_key"]["openrouter_embeddings"]["severity"] == "ok"


def test_recent_provider_failures_classifies_ocr_vision_logs(tmp_path):
    """Ticket #0251: OCR/vision provider outages must be visible to the
    provider-failure probe — 21 connection-refused describes previously
    scored total_count=0 because only openrouter/deepinfra lines matched."""
    from datetime import datetime, timezone

    now = datetime(2026, 7, 10, 19, 0, 0, tzinfo=timezone.utc).timestamp()
    (tmp_path / "indexer.log").write_text(
        "\n".join(
            [
                "2026-07-10 18:44:35,921 WARNING extractors: OCR describe failed for "
                "/data/documents/quo-attachments/Jermaine-125S13-finishing/photo1.jpg: "
                "[Errno 111] Connection refused",
                "2026-07-10 18:45:00,100 WARNING providers.ocr.ollama_vision: "
                "Vision describe still empty after 3 retries for /data/documents/x.png",
                "2026-07-10 18:46:12,004 WARNING extractors: OCR failed for page 3: "
                "ollama describe exceeded wall deadline 300s",
            ]
        )
    )

    result = mcp_server._recent_provider_failures(tmp_path, now=now)

    assert result["status"] == "degraded"
    assert result["total_count"] == 3
    assert result["by_key"]["ocr_vision_describe"]["count"] == 2
    assert result["by_key"]["ocr_vision_describe"]["operation"] == "describe"
    # sample tracks the most recent failure line for the key
    assert "Vision describe still empty" in result["by_key"]["ocr_vision_describe"]["sample"]
    assert result["by_key"]["ocr_vision_page"]["count"] == 1


def test_recent_provider_failures_ocr_vision_recovery_clears_status(tmp_path):
    """A vision describe / page-OCR success (2xx to the provider endpoints)
    after failures marks the OCR keys recovered, so status returns to ok."""
    from datetime import datetime, timezone

    now = datetime(2026, 7, 10, 20, 0, 0, tzinfo=timezone.utc).timestamp()
    (tmp_path / "indexer.log").write_text(
        "\n".join(
            [
                "2026-07-10 18:44:35,921 WARNING extractors: OCR describe failed for "
                "/data/documents/a.jpg: [Errno 111] Connection refused",
                "2026-07-10 18:46:12,004 WARNING extractors: OCR failed for page 3: "
                "[Errno 111] Connection refused",
                "2026-07-10 19:30:00,000 INFO httpx: HTTP Request: POST "
                'http://192.168.68.70:11434/api/chat "HTTP/1.1 200 OK"',
                "2026-07-10 19:31:00,000 INFO httpx: HTTP Request: POST "
                'http://192.168.68.70:8790/extract "HTTP/1.1 200 OK"',
            ]
        )
    )

    result = mcp_server._recent_provider_failures(tmp_path, now=now)

    assert result["status"] == "ok"
    assert result["recovered_count"] == 2
    assert result["by_key"]["ocr_vision_describe"]["recovered"] is True
    assert result["by_key"]["ocr_vision_page"]["recovered"] is True


def test_file_status_surfaces_recent_provider_failures_from_logs(tmp_path):
    """file_status should summarize recent provider failures without live provider probes."""
    import json
    from datetime import datetime, timezone

    (tmp_path / "index_metadata.json").write_text(
        json.dumps({"last_run_at": datetime.now(timezone.utc).isoformat(), "failed_count": 0})
    )
    (tmp_path / "indexer.log").write_text(
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S,000')} "
        "ERROR providers.embed.openrouter_embed: OpenRouter embedding API error: 403 "
        '{"error":{"message":"Key limit exceeded (monthly limit)","code":403}}\n'
    )

    old_cache = mcp_server._cache
    old_signature = mcp_server._cache_index_signature
    old_identity = mcp_server._cache_identity
    old_deep_cache = mcp_server._deep_health_cache
    try:
        mock_store = MagicMock()
        mock_store.list_doc_ids.return_value = ["documents::doc-1"]
        mock_store.count_chunks.return_value = 1
        mock_store.list_recent_docs.return_value = [{"doc_id": "documents::doc-1", "mtime": 1000.0}]
        mock_store._metadata_subfields.return_value = {"doc_id", "source_name", "mtime"}
        mock_store.fts_available.return_value = True

        mcp_server._cache = (
            mock_store,
            MagicMock(),
            {
                "index_root": str(tmp_path),
                "embeddings": {"provider": "openrouter"},
                "search": {"reranker": {"enabled": False}},
                "sources": [{"type": "filesystem", "name": "documents"}],
            },
        )
        mcp_server._cache_index_signature = mcp_server._index_metadata_signature(mcp_server._cache[2])
        mcp_server._cache_identity = id(mcp_server._cache)
        mcp_server._deep_health_cache = None

        result = mcp_server._file_status_impl()

        health = result["health"]
        assert health["provider_status"] == "critical"
        assert health["provider_failures"]["total_count"] == 1
        assert health["provider_failures"]["by_key"]["openrouter_embeddings"]["http_status"] == 403
        assert health["deep_check"]["overall"] == "critical"
        assert health["deep_check"]["checks"]["provider_status"] == "critical"
    finally:
        mcp_server._cache = old_cache
        mcp_server._cache_index_signature = old_signature
        mcp_server._cache_identity = old_identity
        mcp_server._deep_health_cache = old_deep_cache


def test_file_status_reports_zombie_indexer_as_not_running():
    """Zombie indexer PID should not surface as a live background run."""
    import subprocess
    import sys
    import tempfile
    import time

    with tempfile.TemporaryDirectory() as tmpdir:
        pid_file = Path(tmpdir) / "indexer.pid"
        zombie = subprocess.Popen(
            [sys.executable, "-c", "pass"],
            start_new_session=True,
        )
        time.sleep(0.1)
        pid_file.write_text(str(zombie.pid))

        old_cache = mcp_server._cache
        try:
            mock_store = MagicMock()
            mock_store.list_doc_ids.return_value = ["a.md"]
            mock_store.count_chunks.return_value = 1
            mock_store._metadata_subfields.return_value = {"doc_id"}
            mock_store.fts_available.return_value = True

            mcp_server._cache = (
                mock_store,
                MagicMock(),
                {
                    "index_root": tmpdir,
                    "embeddings": {"provider": "openrouter"},
                    "search": {"reranker": {"enabled": False}},
                },
            )

            result = mcp_server._file_status_impl()
            assert result["indexer_running"] is False
            assert "indexer_pid" not in result
        finally:
            mcp_server._cache = old_cache
            zombie.wait()
            pid_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# enr_doc_type / enr_topics passthrough
# ---------------------------------------------------------------------------


def test_search_passes_enr_doc_type():
    """_file_search_impl should pass enr_doc_type through to hybrid_search."""
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


def test_file_status_refreshes_cache_after_lance_manifest_error(tmp_path):
    """A stale Lance table handle should be reopened once before status fails."""
    old_cache = mcp_server._cache
    old_signature = getattr(mcp_server, "_cache_index_signature", None)
    old_identity = getattr(mcp_server, "_cache_identity", None)
    try:
        bad_store = MagicMock()
        bad_store.list_doc_ids.side_effect = RuntimeError(
            "Dataset at path data/index/chunks.lance/_versions/1240.manifest was not found"
        )

        good_store = MagicMock()
        good_store.list_doc_ids.return_value = ["a.md"]
        good_store.count_chunks.return_value = 1
        good_store._metadata_subfields.return_value = {"doc_id"}
        good_store.fts_available.return_value = True

        config = {
            "index_root": str(tmp_path),
            "embeddings": {"provider": "openrouter"},
            "search": {"reranker": {"enabled": False}},
        }

        mcp_server._cache = (bad_store, MagicMock(), config)
        mcp_server._cache_index_signature = None
        mcp_server._cache_identity = id(mcp_server._cache)

        with patch.object(
            mcp_server,
            "_build_store_and_embed",
            return_value=(good_store, MagicMock(), config),
        ) as build:
            result = mcp_server._file_status_impl()

        assert "error" not in result
        assert result["doc_count"] == 1
        assert result["chunk_count"] == 1
        assert build.call_count == 1
    finally:
        mcp_server._cache = old_cache
        mcp_server._cache_index_signature = old_signature
        mcp_server._cache_identity = old_identity


def test_file_status_ignores_zombie_indexer_pid(tmp_path):
    """Zombie indexer PID should be treated as not running and cleaned up."""
    zombie_pid = os.fork()
    if zombie_pid == 0:
        os._exit(0)

    pid_file = tmp_path / "indexer.pid"
    pid_file.write_text(str(zombie_pid))

    deadline = time.time() + 5
    status_path = Path(f"/proc/{zombie_pid}/status")
    while time.time() < deadline:
        try:
            status = status_path.read_text()
        except FileNotFoundError:
            break
        if "State:\tZ" in status:
            break
        time.sleep(0.01)
    else:
        os.waitpid(zombie_pid, 0)
        raise AssertionError("Failed to create zombie child for test")

    old_cache = mcp_server._cache
    try:
        mock_store = MagicMock()
        mock_store.list_doc_ids.return_value = ["a.md"]
        mock_store.count_chunks.return_value = 1
        mock_store._metadata_subfields.return_value = {"doc_id"}
        mock_store.fts_available.return_value = True

        mcp_server._cache = (
            mock_store,
            MagicMock(),
            {
                "index_root": str(tmp_path),
                "embeddings": {"provider": "openrouter"},
                "search": {"reranker": {"enabled": False}},
            },
        )

        result = mcp_server._file_status_impl()

        assert result["indexer_running"] is False
        assert "indexer_pid" not in result
        assert not pid_file.exists(), "Zombie PID file should be removed during status check"
    finally:
        mcp_server._cache = old_cache
        try:
            os.waitpid(zombie_pid, os.WNOHANG)
        except ChildProcessError:
            pass


def test_file_status_ignores_non_indexer_pid_file(tmp_path):
    """A reused PID for another process must not be treated as an indexer."""
    import subprocess
    import sys

    other_proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )

    pid_file = tmp_path / "indexer.pid"
    pid_file.write_text(str(other_proc.pid))

    old_cache = mcp_server._cache
    try:
        mock_store = MagicMock()
        mock_store.list_doc_ids.return_value = ["a.md"]
        mock_store.count_chunks.return_value = 1
        mock_store._metadata_subfields.return_value = {"doc_id"}
        mock_store.fts_available.return_value = True

        mcp_server._cache = (
            mock_store,
            MagicMock(),
            {
                "index_root": str(tmp_path),
                "embeddings": {"provider": "openrouter"},
                "search": {"reranker": {"enabled": False}},
            },
        )

        result = mcp_server._file_status_impl()

        assert result["indexer_running"] is False
        assert "indexer_pid" not in result
        assert not pid_file.exists(), "Foreign PID file should be removed during status check"
    finally:
        mcp_server._cache = old_cache
        other_proc.terminate()
        other_proc.wait()


# ---------------------------------------------------------------------------
# /health probe (_health_probe) — unauthenticated docker-health endpoint
# ---------------------------------------------------------------------------


def test_health_probe_ok_when_idle_and_no_warnings(tmp_path):
    """Idle indexer with no recorded FTS failures probes healthy — and always
    carries index-filesystem telemetry (#0232)."""
    payload, status_code = mcp_server._health_probe({"index_root": str(tmp_path)})

    assert status_code == 200
    assert payload["status"] == "ok"
    assert payload["indexer"] == "idle"
    assert 0 <= payload["disk_used_percent"] <= 100
    assert payload["disk_free_bytes"] > 0
    assert payload["disk_max_percent"] == 90.0


def _disk_usage(used_percent: float):
    """A shutil.disk_usage-shaped triple at the given used percentage."""
    from collections import namedtuple

    total = 100 * 2**30
    used = int(total * used_percent / 100)
    return namedtuple("usage", "total used free")(total, used, total - used)


def test_health_probe_disk_high_water_503s(tmp_path):
    """At/above DISK_USAGE_MAX_PERCENT the probe must 503: disk-full on the
    index filesystem is an outage class (#0232 grew Lance garbage to 93% of
    /data) and docker-health is the surface operators watch."""
    with patch("mcp_server.shutil.disk_usage", return_value=_disk_usage(93)):
        payload, status_code = mcp_server._health_probe({"index_root": str(tmp_path)})

    assert status_code == 503
    assert payload["status"] == "disk_full"
    assert payload["disk_used_percent"] == 93.0
    assert payload["disk_free_bytes"] == _disk_usage(93).free


def test_health_probe_disk_threshold_env_override_and_invalid_fallback(tmp_path):
    """DISK_USAGE_MAX_PERCENT tunes the high-water mark; junk or out-of-range
    values fall back to the default 90 instead of disabling the check."""
    with patch("mcp_server.shutil.disk_usage", return_value=_disk_usage(75)):
        with patch.dict(os.environ, {"DISK_USAGE_MAX_PERCENT": "70"}):
            payload, status_code = mcp_server._health_probe({"index_root": str(tmp_path)})
        assert status_code == 503
        assert payload["status"] == "disk_full"
        assert payload["disk_max_percent"] == 70.0

        for bad in ("junk", "0", "-5", "250"):
            with patch.dict(os.environ, {"DISK_USAGE_MAX_PERCENT": bad}):
                payload, status_code = mcp_server._health_probe({"index_root": str(tmp_path)})
            assert status_code == 200, f"DISK_USAGE_MAX_PERCENT={bad}"
            assert payload["disk_max_percent"] == 90.0


def test_health_probe_stalled_payload_includes_disk_fields(tmp_path):
    """Disk telemetry must compose with the frozen-indexer 503, not vanish."""
    hb = tmp_path / "indexer.heartbeat"
    hb.write_text("beat")
    stale = time.time() - 4000
    os.utime(hb, (stale, stale))

    with patch("mcp_server._resolve_indexer_pid", return_value=(True, 12345)):
        with patch("mcp_server.shutil.disk_usage", return_value=_disk_usage(50)):
            payload, status_code = mcp_server._health_probe({"index_root": str(tmp_path)})

    assert status_code == 503
    assert payload["status"] == "stalled"
    assert payload["disk_used_percent"] == 50.0


def test_health_probe_fts_failure_composes_with_disk_fields(tmp_path):
    """FTS failure detail and disk telemetry appear together; when the disk is
    also over the high-water mark, disk-full wins the status (more urgent)."""
    import json

    (tmp_path / "index_metadata.json").write_text(json.dumps({
        "warning_counts": {"fts_rebuild_failed": 1},
    }))

    with patch("mcp_server.shutil.disk_usage", return_value=_disk_usage(50)):
        payload, status_code = mcp_server._health_probe({"index_root": str(tmp_path)})
    assert status_code == 503
    assert payload["status"] == "degraded"
    assert payload["disk_used_percent"] == 50.0

    with patch("mcp_server.shutil.disk_usage", return_value=_disk_usage(95)):
        payload, status_code = mcp_server._health_probe({"index_root": str(tmp_path)})
    assert status_code == 503
    assert payload["status"] == "disk_full"
    assert payload["fts_rebuild_failed"] == 1


def test_health_probe_degraded_when_fts_rebuild_failed(tmp_path):
    """A recorded fts_rebuild_failed warning must 503 the probe (#0106).

    Keyword search silently going stale used to be invisible: the flow ends
    Completed() and fts_available stays true. The probe is the one unauthenticated
    surface docker-health can watch, so persistent FTS failure must show here.
    """
    import json

    (tmp_path / "index_metadata.json").write_text(json.dumps({
        "last_run_at": "2026-07-02T06:05:15+00:00",
        "warning_counts": {"fts_rebuild_failed": 1},
    }))

    payload, status_code = mcp_server._health_probe({"index_root": str(tmp_path)})

    assert status_code == 503
    assert payload["status"] == "degraded"
    assert payload["fts_rebuild_failed"] == 1


def test_health_probe_running_fresh_heartbeat_is_ok(tmp_path):
    """A live indexer with a fresh heartbeat probes healthy."""
    (tmp_path / "indexer.heartbeat").write_text("beat")

    with patch("mcp_server._resolve_indexer_pid", return_value=(True, 12345)):
        payload, status_code = mcp_server._health_probe({"index_root": str(tmp_path)})

    assert status_code == 200
    assert payload["status"] == "ok"
    assert payload["indexer"] == "running"
    assert payload["indexer_pid"] == 12345
    assert payload["heartbeat_age_s"] == 0


def test_health_probe_stalled_heartbeat_still_503s(tmp_path):
    """The pre-existing frozen-indexer detection survives the FTS addition."""
    hb = tmp_path / "indexer.heartbeat"
    hb.write_text("beat")
    stale = time.time() - 4000  # default max age is 1800s
    os.utime(hb, (stale, stale))

    with patch("mcp_server._resolve_indexer_pid", return_value=(True, 12345)):
        payload, status_code = mcp_server._health_probe({"index_root": str(tmp_path)})

    assert status_code == 503
    assert payload["status"] == "stalled"


def test_health_probe_running_with_fts_failure_degrades(tmp_path):
    """FTS failure is surfaced even while an indexer run is in progress."""
    import json

    (tmp_path / "indexer.heartbeat").write_text("beat")
    (tmp_path / "index_metadata.json").write_text(json.dumps({
        "warning_counts": {"fts_rebuild_failed": 1},
    }))

    with patch("mcp_server._resolve_indexer_pid", return_value=(True, 12345)):
        payload, status_code = mcp_server._health_probe({"index_root": str(tmp_path)})

    assert status_code == 503
    assert payload["status"] == "degraded"
    assert payload["fts_rebuild_failed"] == 1


def test_health_probe_idle_after_killed_indexer_stays_failed(tmp_path):
    """SIGKILL terminal state must survive PID cleanup and keep health red."""
    import json

    terminal = {
        "run_id": "killed-run",
        "status": "signaled",
        "pid": 12345,
        "source_name": None,
        "started_at": "2026-07-15T12:00:00+00:00",
        "finished_at": "2026-07-15T12:05:00+00:00",
        "exit_code": None,
        "termination_signal": 9,
        "terminal_reason": "process_exit",
        "peak_rss_bytes": 123456,
    }
    (tmp_path / "index_run_state.json").write_text(json.dumps({
        "version": 1,
        "current": None,
        "last_attempt": terminal,
        "last_success": None,
    }))

    payload, status_code = mcp_server._health_probe({"index_root": str(tmp_path)})

    assert status_code == 503
    assert payload["status"] == "index_failed"
    assert payload["indexer"] == "idle"
    assert payload["index_run"]["unresolved_failure"] is True
    assert payload["index_run"]["latest_terminal"]["termination_signal"] == 9


def test_file_status_exposes_last_attempt_success_and_terminal_freshness(tmp_path):
    import json

    terminal = {
        "run_id": "failed-after-success",
        "status": "failed",
        "pid": 12345,
        "source_name": "documents",
        "started_at": "2026-07-15T12:00:00+00:00",
        "finished_at": "2026-07-15T12:05:00+00:00",
        "exit_code": 1,
        "termination_signal": None,
        "terminal_reason": "process_exit",
        "peak_rss_bytes": 123456,
    }
    success = {
        **terminal,
        "run_id": "older-success",
        "status": "succeeded",
        "finished_at": "2026-07-14T12:05:00+00:00",
        "exit_code": 0,
        "terminal_reason": "clean_exit",
    }
    (tmp_path / "index_run_state.json").write_text(json.dumps({
        "version": 1,
        "current": None,
        "last_attempt": terminal,
        "last_success": success,
    }))

    store = MagicMock()
    store.list_doc_ids.return_value = ["a.md"]
    store.count_chunks.return_value = 1
    store._metadata_subfields.return_value = {"doc_id"}
    store.fts_available.return_value = True
    config = {
        "index_root": str(tmp_path),
        "embeddings": {"provider": "openrouter"},
        "search": {"reranker": {"enabled": False}},
    }
    old_cache = mcp_server._cache
    try:
        mcp_server._cache = (store, MagicMock(), config)
        with patch("mcp_server._get_deep_health", return_value={}):
            result = mcp_server._file_status_impl()
    finally:
        mcp_server._cache = old_cache

    assert result["index_run"]["last_attempt"]["run_id"] == "failed-after-success"
    assert result["index_run"]["last_success"]["run_id"] == "older-success"
    assert result["index_run"]["latest_terminal"]["status"] == "failed"
    assert result["index_run"]["unresolved_failure"] is True
