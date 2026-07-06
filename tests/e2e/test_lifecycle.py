"""Upload → index → search → fetch → download lifecycle, driven from outside.

Provider-sim determinism notes: embeddings are content-hash-derived (vector
ranking is arbitrary), but FTS/BM25 and the sim reranker (lexical overlap) are
real — so assertions search for distinctive lexical markers:
  - note.md contains the phrase "quixotic manganese lighthouse"
  - OCR'd docs contain "[ocr] <filename> <sha12>"
  - media docs contain "[transcript] simulated <kinds> transcript <sha12>"

Under the opt-in real-API stage (E2E_REAL=1, media + enrichment are live), real
transcription output is non-deterministic, so the media assertion relaxes from
"searchable by the sim marker" to "the media docs went through the pipeline and
are indexed". OCR stays on the sim in that mode, so its marker assertion holds.
"""
import os

import pytest

from tests.e2e.client import search_hits as _hits
from tests.e2e.conftest import NOTE_PHRASE

pytestmark = pytest.mark.anyio
E2E_REAL = os.environ.get("E2E_REAL") == "1"


def _rel_paths(results: list[dict]) -> list[str]:
    return [r.get("rel_path", "") for r in results]


async def test_note_searchable_by_phrase(indexed_corpus, mcp_session):
    results = _hits(await mcp_session.call_tool_json(
        "file_search", {"query": NOTE_PHRASE, "top_k": 8}))
    assert results, "no hits for the note.md phrase"
    note_hits = [r for r in results if "note" in r.get("rel_path", "")]
    assert note_hits, f"note.md not in hits: {_rel_paths(results)}"
    top = note_hits[0]
    assert top.get("doc_id") and top.get("loc"), top
    assert "lighthouse" in (top.get("snippet") or "").lower() or top.get("title")


async def test_png_ocr_pipeline_searchable(indexed_corpus, mcp_session):
    # The sim's OCR text is "[ocr] <stored filename> <sha12>"; the stored file
    # keeps its "diagram" stem (ID-alias rename only inserts @xxxxx@).
    results = _hits(await mcp_session.call_tool_json(
        "file_search", {"query": "ocr diagram png", "top_k": 8}))
    png_hits = [r for r in results if "diagram" in r.get("rel_path", "")]
    assert png_hits, f"OCR-derived diagram.png doc not found: {_rel_paths(results)}"


async def test_audio_and_video_pipelines_searchable(indexed_corpus, mcp_session):
    if E2E_REAL:
        # Real transcription is non-deterministic (and the fixtures are synthetic
        # tone/testsrc clips), so assert each media file completed the real
        # pipeline — document.indexed emitted for it — rather than searching for
        # specific transcript content. hook_events is the sink snapshot captured
        # while the corpus indexed.
        indexed = [str(e.get("rel_path") or e.get("abs_path") or "")
                   for e in indexed_corpus["hook_events"]]
        assert any(p.endswith(".wav") for p in indexed), f"clip.wav did not index: {indexed}"
        assert any(p.endswith(".mp4") for p in indexed), f"clip.mp4 did not index: {indexed}"
        return
    results = _hits(await mcp_session.call_tool_json(
        "file_search", {"query": "transcript simulated", "top_k": 10}))
    paths = _rel_paths(results)
    assert any(p.endswith(".wav") for p in paths), f"clip.wav transcript missing: {paths}"
    assert any(p.endswith(".mp4") for p in paths), f"clip.mp4 transcript missing: {paths}"


async def test_chunk_roundtrip(indexed_corpus, mcp_session):
    hit = _hits(await mcp_session.call_tool_json(
        "file_search", {"query": NOTE_PHRASE, "top_k": 3}))[0]

    chunk = await mcp_session.call_tool_json(
        "file_get_chunk", {"doc_id": hit["doc_id"], "loc": hit["loc"]})
    assert isinstance(chunk, dict) and not chunk.get("error"), chunk
    assert "lighthouse" in chunk["text"].lower()

    chunks = await mcp_session.call_tool_json(
        "file_get_doc_chunks", {"doc_id": hit["doc_id"]})
    assert isinstance(chunks, list) and chunks, chunks
    assert hit["loc"] in [c.get("loc") for c in chunks]
    assert any("lighthouse" in (c.get("text") or "").lower() for c in chunks)


async def test_download_original_bytes(indexed_corpus, mcp_session, api):
    from tests.e2e.conftest import FIXTURES

    hit = _hits(await mcp_session.call_tool_json(
        "file_search", {"query": NOTE_PHRASE, "top_k": 3}))[0]
    rel_path = hit.get("rel_path")
    assert rel_path, f"slim hit missing rel_path: {hit}"

    resp = await api.get(f"/api/documents/{rel_path}")
    assert resp.status_code == 200, resp.text
    # Content is untouched by the ID-alias rename — bytes must match exactly.
    assert resp.content == (FIXTURES / "note.md").read_bytes()


async def test_search_sort_recent(indexed_corpus, mcp_session):
    # Query the note phrase (present in both sim and real modes) — this test
    # asserts recency ordering, not media content.
    results = _hits(await mcp_session.call_tool_json(
        "file_search", {"query": NOTE_PHRASE, "top_k": 10, "sort": "recent"}))
    assert results
    for r in results:
        assert r.get("doc_id") and r.get("loc"), r
    # newest-first: mtimes (when present) must be non-increasing
    mtimes = [r["mtime"] for r in results if isinstance(r.get("mtime"), (int, float))]
    assert mtimes == sorted(mtimes, reverse=True)


async def test_search_with_source_type_filter(indexed_corpus, mcp_session):
    results = _hits(await mcp_session.call_tool_json(
        "file_search",
        {"query": "lighthouse survey", "top_k": 10, "source_type": "md"}))
    assert results, "filtered search returned nothing"
    for r in results:
        assert r.get("source_type") == "md", r


async def test_rest_search_parity(indexed_corpus, api):
    resp = await api.post("/api/search", json={"query": NOTE_PHRASE, "top_k": 8})
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert not payload.get("error"), payload
    assert any("note" in r.get("rel_path", "") for r in payload["results"]), payload
