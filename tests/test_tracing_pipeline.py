# Pipeline + MCP span instrumentation tests (staging gate Task 4).
#
# Wiring mirrors tests/test_indexing_roundtrip.int.test.py: REAL extractors +
# REAL LanceDB + MockEmbedProvider (no API keys), but drives the production
# pipeline functions (process_doc_task via .fn, hybrid_search) with tracing
# enabled, then asserts on the REAL span JSONL files written by the exporter.
#
# Enrichment note: the pipeline wiring below has no llm_generator, so
# enrich_document is never called by process_doc_task (matching the roundtrip
# wiring, where enrichment is disabled). The `enrich` span is therefore
# asserted via a direct enrich_document() call with a stub generator.
#
# Ordering note: tests/test_tracing.py sorts before this file, so its
# "disabled is a no-op" test still runs before any provider exists here.

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.tracing import get_tracer, setup_tracing, shutdown_tracing


_MD_CONTENT = """\
---
title: Tracing Pipeline Note
tags: [tracing, testing]
status: active
---

# Tracing Pipeline Note

Machine learning embeddings power semantic document search.

## Details

Keyword search uses BM25 scoring via the tantivy FTS index.
Hybrid search fuses vector and keyword results with RRF.
"""


class MockEmbedProvider:
    """Deterministic 768d vectors for offline testing."""

    def embed_texts(self, texts):
        return [[0.1] * 768 for _ in texts]

    def embed_query(self, query):
        return [0.1] * 768


def _read_spans(span_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for f in sorted(span_dir.glob("*.jsonl"))
        for line in f.read_text().splitlines()
        if line.strip()
    ]


@pytest.fixture
def span_dir(tmp_path):
    d = tmp_path / "spans"
    setup_tracing(
        {"tracing": {"enabled": True, "directory": str(d)}},
        service_name="test-pipeline",
    )
    yield d
    shutdown_tracing()  # idempotent if the test already shut down


# ---------------------------------------------------------------------------
# Pipeline stages: process_doc (parent) -> extract / embed / store.upsert,
# then search.hybrid over the same store.
# ---------------------------------------------------------------------------


def _index_note(tmp_path):
    """Index _MD_CONTENT through the real pipeline; return (store, embed)."""
    import flow_index_vault as fiv
    from lancedb_store import LanceDBStore
    from llama_index.core.node_parser import SentenceSplitter

    vault = tmp_path / "vault"
    vault.mkdir()
    md_file = vault / "note.md"
    md_file.write_text(_MD_CONTENT)

    store = LanceDBStore(str(tmp_path / "index"), "test_chunks")
    embed = MockEmbedProvider()

    fiv._RUNTIME.clear()
    fiv._RUNTIME.update(
        {
            "store": store,
            "embed_provider": embed,
            "splitter": SentenceSplitter(chunk_size=300, chunk_overlap=50),
            "semantic_splitter": None,
            "semantic_threshold": 0,
            "ocr_provider": None,
            "config": {},
        }
    )
    doc = {
        "doc_id": "documents::00abc",
        "rel_path": "note.md",
        "abs_path": str(md_file),
        "mtime": md_file.stat().st_mtime,
        "size": md_file.stat().st_size,
        "ext": "md",
        "source_name": "documents",
    }
    try:
        fiv.process_doc_task.fn(doc)
        store.create_fts_index()
    finally:
        fiv._RUNTIME.clear()
    return store, embed


def test_pipeline_stages_emit_spans(tmp_path, span_dir):
    from search_hybrid import hybrid_search

    store, embed = _index_note(tmp_path)
    result = hybrid_search(store, embed, "machine learning embedding", final_top_k=5)
    assert len(result) > 0  # pipeline actually worked; spans were additive

    shutdown_tracing()
    spans = _read_spans(span_dir)
    by_name = {s["name"]: s for s in spans}

    parent = by_name.get("process_doc")
    assert parent is not None, f"no process_doc span; got {sorted(by_name)}"
    assert parent["attributes"]["doc_id"] == "documents::00abc"
    assert parent["attributes"]["rel_path"] == "note.md"
    assert parent["attributes"]["source"] == "documents"

    for child_name in ("extract", "embed", "store.upsert"):
        child = by_name.get(child_name)
        assert child is not None, f"no {child_name} span; got {sorted(by_name)}"
        assert child["trace_id"] == parent["trace_id"], child_name
        assert child["parent_span_id"] == parent["span_id"], child_name

    assert by_name["embed"]["attributes"]["chunk_count"] >= 1

    search_span = by_name.get("search.hybrid")
    assert search_span is not None, f"no search.hybrid span; got {sorted(by_name)}"
    assert search_span["attributes"]["top_k"] == 5
    # The search ran outside process_doc, so it must be its own trace.
    assert search_span["trace_id"] != parent["trace_id"]


def test_attachment_context_is_embedded_and_traced(tmp_path, span_dir):
    import flow_index_vault as fiv
    from communication_context import CommunicationMessage, SourceWindowContextProvider
    from extractors import ExtractionResult
    from lancedb_store import LanceDBStore
    from llama_index.core.node_parser import SentenceSplitter
    from sources.base import SourceRecord

    class StaticSource:
        def extract(self, _record):
            return ExtractionResult.from_text("simulated video walkthrough")

    store = LanceDBStore(str(tmp_path / "index"), "attachment_chunks")
    embed = MockEmbedProvider()
    record = SourceRecord(
        doc_id="clip.mp4",
        source_type="video",
        natural_key="clip.mp4",
        mtime=1.0,
        size=123,
        metadata={
            "source": "quo",
            "source_message_id": "attachment-1",
            "source_channel_id": "ops",
            "sent_at": "2026-06-01T10:00:30Z",
        },
    )
    provider = SourceWindowContextProvider.from_messages(
        [
            CommunicationMessage(
                source_message_id="msg-001",
                sender="Alice",
                sent_at="2026-06-01T10:00:00Z",
                text="quarterly zephyr report before marker",
                origin_source="quo",
                channel_id="ops",
            ),
            CommunicationMessage(
                source_message_id="msg-002",
                sender="Bob",
                sent_at="2026-06-01T10:01:00Z",
                text="marmalade budget after marker",
                origin_source="quo",
                channel_id="ops",
            ),
        ],
        message_channels={"msg-001": "ops", "msg-002": "ops"},
    )
    doc = {
        "doc_id": "documents::clip",
        "rel_path": "clip.mp4",
        "mtime": 1.0,
        "size": 123,
        "ext": "mp4",
        "source_type": "video",
        "source_name": "documents",
    }
    fiv._RUNTIME.clear()
    fiv._RUNTIME.update(
        {
            "store": store,
            "embed_provider": embed,
            "splitter": SentenceSplitter(chunk_size=300, chunk_overlap=50),
            "semantic_splitter": None,
            "semantic_threshold": 0,
            "config": {},
            "sources_by_name": {"documents": StaticSource()},
            "source_records_by_ns_doc_id": {doc["doc_id"]: record},
            "communication_context_provider": provider,
        }
    )
    try:
        fiv.process_doc_task.fn(doc)
        chunks = store.get_doc_chunks(doc["doc_id"])
    finally:
        fiv._RUNTIME.clear()

    assert chunks
    stored = "\n".join(chunk.text for chunk in chunks)
    assert "quarterly zephyr report before marker" in stored
    assert "marmalade budget after marker" in stored

    shutdown_tracing()
    spans = _read_spans(span_dir)
    parent = next(s for s in spans if s["name"] == "process_doc")
    context_span = next(s for s in spans if s["name"] == "attachment.context.resolve")
    assert context_span["trace_id"] == parent["trace_id"]
    assert context_span["parent_span_id"] == parent["span_id"]
    assert context_span["attributes"] == {
        "before_count": 1,
        "after_count": 1,
        "has_context": True,
    }


def test_media_recall_survives_rerank_cutoff_and_emits_audit_spans(span_dir):
    from core.storage import SearchHit
    from search_hybrid import Reranker, hybrid_search

    class Embed:
        def embed_query(self, _query):
            return [1.0, 0.0]

    class KeepOrderReranker(Reranker):
        def rerank(self, _query, hits):
            return hits

    class Store:
        def _build_where_clause(self, **kwargs):
            return kwargs.get("source_type") or ""

        def vector_search(self, _vector, top_k, where):
            if where == "video":
                return [SearchHit("target-video", "video:c:0", "", "", 9.0, "video")]
            return [
                SearchHit(f"message-{i:03d}", "c:0", "", "", 100.0 - i, "pg_message")
                for i in range(min(top_k, 70))
            ]

        def keyword_search(self, _query, top_k, where):
            return self.vector_search(None, top_k, where)

    result = hybrid_search(
        Store(),
        Embed(),
        "show quarterly zephyr property video",
        vector_top_k=70,
        keyword_top_k=70,
        final_top_k=10,
        reranker=KeepOrderReranker(),
        media_intent_slots=1,
    )
    assert "target-video" in [hit.doc_id for hit in result]
    assert result.diagnostics["media_intent_recall"] == 1

    shutdown_tracing()
    spans = _read_spans(span_dir)
    search_span = next(s for s in spans if s["name"] == "search.hybrid")
    recall = next(s for s in spans if s["name"] == "search.media_recall")
    quota = next(s for s in spans if s["name"] == "search.media_quota")
    for child in (recall, quota):
        assert child["trace_id"] == search_span["trace_id"]
        assert child["parent_span_id"] == search_span["span_id"]
    assert recall["attributes"]["missing_types"] == "video"
    assert recall["attributes"]["recalled_count"] == 1
    assert quota["attributes"]["returned_media_count"] == 1


def test_scan_vault_task_emits_scan_span(tmp_path, span_dir):
    import flow_index_vault as fiv

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text(_MD_CONTENT)

    fiv._RUNTIME.clear()
    try:
        records = fiv.scan_vault_task.fn(vault, ["**/*.md"], [])
    finally:
        fiv._RUNTIME.clear()
    assert len(records) == 1

    shutdown_tracing()
    spans = _read_spans(span_dir)
    assert any(s["name"] == "scan" for s in spans), [s["name"] for s in spans]


def test_enrich_document_emits_enrich_span(span_dir):
    from doc_enrichment import enrich_document

    generator = MagicMock()
    generator.generate.return_value = '{"summary": "a note", "doc_type": "note"}'

    enrich_document(
        text="Some document text about testing.",
        title="T",
        source_type="md",
        generator=generator,
    )

    shutdown_tracing()
    spans = _read_spans(span_dir)
    assert any(s["name"] == "enrich" for s in spans), [s["name"] for s in spans]


# ---------------------------------------------------------------------------
# MCP tool-call spans (generic wrapper applied at registration time)
# ---------------------------------------------------------------------------


def test_mcp_tool_call_emits_span(span_dir):
    import mcp_server

    if not mcp_server.HAS_MCP:
        pytest.skip("mcp package not installed")

    with patch("mcp_server._file_taxonomy_list_impl", return_value=[]):
        mcp_server.file_taxonomy_list(kind="tag", status="active")

    shutdown_tracing()
    spans = _read_spans(span_dir)
    span = next(
        (s for s in spans if s["name"] == "mcp.tool.file_taxonomy_list"), None
    )
    assert span is not None, [s["name"] for s in spans]
    assert span["attributes"]["tool"] == "file_taxonomy_list"
    # kind/status are not on the allowlist and must not be recorded
    assert "kind" not in span["attributes"]
    assert "status" not in span["attributes"]


def test_mcp_tool_span_records_only_allowlisted_scalar_args(span_dir):
    import mcp_server

    if not mcp_server.HAS_MCP:
        pytest.skip("mcp package not installed")

    with patch("mcp_server._file_search_impl", return_value={"results": []}):
        mcp_server.file_search(
            query="private user text", top_k=5, source_name="documents"
        )

    shutdown_tracing()
    spans = _read_spans(span_dir)
    span = next((s for s in spans if s["name"] == "mcp.tool.file_search"), None)
    assert span is not None, [s["name"] for s in spans]
    assert span["attributes"]["top_k"] == 5
    assert span["attributes"]["source_name"] == "documents"
    # NEVER record full arguments or document/query text
    assert "query" not in span["attributes"]
    assert not any(
        "private user text" in str(v) for v in span["attributes"].values()
    )


def test_mcp_file_search_tool_span_parents_search_hybrid(tmp_path, span_dir):
    """The linkage Task 9's server-side coverage check leans on: the
    downstream search.hybrid span must parent under mcp.tool.file_search via
    OTEL context propagation — asserted against the REAL registered tool
    wrapper and the REAL hybrid search over a real store."""
    import mcp_server

    if not mcp_server.HAS_MCP:
        pytest.skip("mcp package not installed")

    store, embed = _index_note(tmp_path)
    with patch("mcp_server._get_deps", return_value=(store, embed, {})):
        resp = mcp_server.file_search(query="machine learning embedding", top_k=3)
    assert resp.get("results"), resp

    shutdown_tracing()
    spans = _read_spans(span_dir)
    by_name = {s["name"]: s for s in spans}
    tool_span = by_name.get("mcp.tool.file_search")
    search_span = by_name.get("search.hybrid")
    assert tool_span is not None, sorted(by_name)
    assert search_span is not None, sorted(by_name)
    assert search_span["trace_id"] == tool_span["trace_id"]
    assert search_span["parent_span_id"] == tool_span["span_id"]


# ---------------------------------------------------------------------------
# LLMTraceRecorder join: records carry current trace/span ids when inside a span
# ---------------------------------------------------------------------------


def test_llm_trace_recorder_joins_current_trace(tmp_path, span_dir):
    from providers.llm.trace_recorder import LLMTraceRecorder

    trace_dir = tmp_path / "llm-traces"
    recorder = LLMTraceRecorder(
        provider="testprov", model="m1", enabled=True, directory=trace_dir
    )

    with get_tracer("pipeline").start_as_current_span("outer"):
        recorder.record(request={"payload": {}}, success=True, latency_ms=1.0)
    # Outside any span: no ids must be attached
    recorder.record(request={"payload": {}}, success=True, latency_ms=1.0)

    shutdown_tracing()
    rows = [
        json.loads(line)
        for f in sorted(trace_dir.glob("*.jsonl"))
        for line in f.read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 2

    spans = _read_spans(span_dir)
    outer = next(s for s in spans if s["name"] == "outer")
    assert rows[0]["trace_id"] == outer["trace_id"]
    assert rows[0]["span_id"] == outer["span_id"]
    assert "trace_id" not in rows[1]
    assert "span_id" not in rows[1]
