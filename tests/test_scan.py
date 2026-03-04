"""Tests for scan_vault_task and glob matching."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from flow_index_vault import (
    scan_vault_task,
    _matches_any,
    _split_markdown_by_headings,
    _build_chunk_context,
    _split_section,
    _RUNTIME,
)


# --- _matches_any ---

def test_matches_md_root():
    assert _matches_any("note.md", ["**/*.md"]) is True

def test_matches_md_nested():
    assert _matches_any("sub/folder/note.md", ["**/*.md"]) is True

def test_no_match_txt():
    assert _matches_any("note.txt", ["**/*.md"]) is False

def test_matches_pdf():
    assert _matches_any("docs/file.pdf", ["**/*.pdf"]) is True

def test_exclude_obsidian():
    assert _matches_any(".obsidian/config.json", [".obsidian/**"]) is True

def test_exclude_trash():
    assert _matches_any(".trash/old.md", [".trash/**"]) is True

def test_exclude_ds_store():
    assert _matches_any("folder/.DS_Store", ["**/.DS_Store"]) is True

def test_no_exclude_normal():
    assert _matches_any("notes/hello.md", [".obsidian/**", ".trash/**"]) is False


# --- scan_vault_task ---

def test_scan_finds_md_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "note1.md").write_text("hello")
        (root / "note2.md").write_text("world")
        sub = root / "sub"
        sub.mkdir()
        (sub / "note3.md").write_text("nested")
        (root / "ignore.txt").write_text("skip me")

        records = scan_vault_task.fn(root, ["**/*.md"], [])
        doc_ids = {r["doc_id"] for r in records}
        assert doc_ids == {"note1.md", "note2.md", "sub/note3.md"}


def test_scan_excludes_patterns():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "good.md").write_text("keep")
        trash = root / ".trash"
        trash.mkdir()
        (trash / "old.md").write_text("discard")

        records = scan_vault_task.fn(root, ["**/*.md"], [".trash/**"])
        doc_ids = {r["doc_id"] for r in records}
        assert doc_ids == {"good.md"}


def test_scan_empty_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        records = scan_vault_task.fn(tmpdir, ["**/*.md"], [])
        assert records == []


def test_scan_record_fields():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "test.md").write_text("content")

        records = scan_vault_task.fn(root, ["**/*.md"], [])
        assert len(records) == 1
        r = records[0]
        assert r["doc_id"] == "test.md"
        assert r["ext"] == "md"
        assert r["size"] > 0
        assert r["mtime"] > 0
        assert Path(r["abs_path"]).exists()


# --- _split_markdown_by_headings ---

def test_heading_split_no_headings():
    text = "Just some plain text\nwith multiple lines."
    sections = _split_markdown_by_headings(text)
    assert len(sections) == 1
    assert sections[0][0] == ""
    assert "plain text" in sections[0][1]


def test_heading_split_single_heading():
    text = "# Title\nSome content here."
    sections = _split_markdown_by_headings(text)
    assert len(sections) == 1
    assert sections[0][0] == "Title"
    assert "# Title" in sections[0][1]


def test_heading_split_hierarchy():
    text = "Preamble\n\n# Intro\nIntro text\n\n## Setup\nSetup text\n\n### Prereqs\nPrereq list\n\n## Config\nConfig text"
    sections = _split_markdown_by_headings(text)
    assert len(sections) == 5
    assert sections[0] == ("", "Preamble")
    assert sections[1][0] == "Intro"
    assert sections[2][0] == "Intro > Setup"
    assert sections[3][0] == "Intro > Setup > Prereqs"
    assert sections[4][0] == "Intro > Config"


def test_heading_split_h2_resets_h3():
    text = "# A\ntext\n## B\ntext\n### C\ntext\n## D\ntext"
    sections = _split_markdown_by_headings(text)
    breadcrumbs = [s[0] for s in sections]
    assert "A > B > C" in breadcrumbs
    assert "A > D" in breadcrumbs


# --- _build_chunk_context ---

def test_context_basic():
    """Header includes title; excludes type/folder/tags/author to stay lean."""
    meta = {"title": "Report", "source_type": "pdf", "folder": "2-Area", "tags": "tax,2023"}
    ctx = _build_chunk_context(meta)
    assert ctx.startswith("[")
    assert "Document: Report" in ctx
    assert ctx.endswith("\n\n")
    # These should NOT be in the header (kept lean for FTS quality)
    assert "Type:" not in ctx
    assert "Folder:" not in ctx
    assert "Tags:" not in ctx


def test_context_with_page():
    meta = {"title": "Report", "source_type": "pdf"}
    ctx = _build_chunk_context(meta, page=5)
    assert "Page: 5" in ctx


def test_context_with_section():
    meta = {"title": "Note", "source_type": "md"}
    ctx = _build_chunk_context(meta, section="Setup > Prerequisites")
    assert "Section: Setup > Prerequisites" in ctx


def test_context_empty_meta():
    ctx = _build_chunk_context({})
    assert ctx == ""


def test_context_with_topics():
    meta = {"title": "Report", "source_type": "pdf", "enr_topics": "soil analysis, foundation design"}
    ctx = _build_chunk_context(meta)
    assert "Topics: soil analysis, foundation design" in ctx


def test_context_with_summary():
    meta = {"title": "Report", "source_type": "pdf", "enr_summary": "A geotechnical report."}
    ctx = _build_chunk_context(meta)
    assert "Summary: A geotechnical report." in ctx
    lines = ctx.strip().split("\n")
    assert len(lines) == 2
    assert lines[1] == "Summary: A geotechnical report."


def test_context_no_summary_no_extra_line():
    meta = {"title": "Report", "source_type": "pdf", "enr_summary": ""}
    ctx = _build_chunk_context(meta)
    assert "Summary" not in ctx


def test_context_excludes_description():
    """Description is stored in metadata columns, not in chunk header."""
    meta = {"title": "Report", "source_type": "pdf", "description": "A detailed geotechnical study."}
    ctx = _build_chunk_context(meta)
    assert "Description" not in ctx


def test_context_excludes_author():
    """Author is stored in metadata columns, not in chunk header."""
    meta = {"title": "Report", "source_type": "pdf", "author": "Dan Park"}
    ctx = _build_chunk_context(meta)
    assert "Author" not in ctx


def test_context_only_title_topics_summary():
    """Full example: header has title + topics bracket line, then summary."""
    meta = {"title": "Report", "source_type": "pdf", "folder": "2-Area",
            "enr_topics": "geotechnical", "enr_summary": "Findings."}
    ctx = _build_chunk_context(meta)
    assert "Document: Report" in ctx
    assert "Topics: geotechnical" in ctx
    assert "Summary: Findings." in ctx
    # Excluded metadata
    assert "Folder:" not in ctx
    assert "Type:" not in ctx


# --- _split_section ---

class _FakeSplitter:
    """Mimics SentenceSplitter: returns the text as one chunk if short enough."""
    def __init__(self, max_chars=100):
        self._max = max_chars

    def split_text(self, text):
        if len(text) <= self._max:
            return [text]
        chunks = []
        for i in range(0, len(text), self._max):
            chunks.append(text[i:i + self._max])
        return chunks


class _FakeSemanticSplitter:
    """Mimics SemanticSplitterNodeParser: splits text at double-newlines."""
    def get_nodes_from_documents(self, docs):
        from types import SimpleNamespace
        text = docs[0].text
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        return [SimpleNamespace(text=p) for p in parts]


def test_split_section_no_semantic():
    """Without semantic splitter, falls back to SentenceSplitter."""
    splitter = _FakeSplitter(max_chars=50)
    chunks = _split_section("short text", splitter)
    assert chunks == ["short text"]


def test_split_section_below_threshold():
    """Text below threshold uses SentenceSplitter even if semantic is available."""
    splitter = _FakeSplitter(max_chars=50)
    semantic = _FakeSemanticSplitter()
    chunks = _split_section("short text", splitter, semantic, semantic_threshold=1000)
    assert chunks == ["short text"]


def test_split_section_above_threshold():
    """Text above threshold uses semantic splitter then SentenceSplitter."""
    long_text = "Topic one about animals. " * 20 + "\n\n" + "Topic two about machines. " * 20
    splitter = _FakeSplitter(max_chars=200)
    semantic = _FakeSemanticSplitter()
    chunks = _split_section(long_text, splitter, semantic, semantic_threshold=100)
    assert len(chunks) >= 2
    assert any("animals" in c for c in chunks)
    assert any("machines" in c for c in chunks)


# --- Symlink cycle guard ---


def test_scan_symlink_cycle_does_not_hang():
    """A symlink cycle (a/link -> a) should not cause an infinite loop."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sub = root / "a"
        sub.mkdir()
        (sub / "note.md").write_text("hello")

        # Create a symlink cycle: a/link -> a
        link = sub / "link"
        os.symlink(str(sub), str(link))

        # Should complete without hanging (timeout protects CI)
        import signal

        def _timeout_handler(signum, frame):
            raise TimeoutError("scan_vault_task hung on symlink cycle")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(5)  # 5 second timeout
        try:
            records = scan_vault_task.fn(root, ["**/*.md"], [])
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        doc_ids = {r["doc_id"] for r in records}
        # Should find the file once (not duplicated via cycle)
        assert "a/note.md" in doc_ids
        assert len(records) == 1


# --- _RUNTIME cleared at flow start (Fix 3) ---


def test_runtime_cleared_at_flow_start():
    """_RUNTIME should be cleared at the start of index_vault_flow."""
    _RUNTIME["stale_key"] = "leftover"
    _RUNTIME["_warnings"] = ["old warning"]

    # Mock get_run_logger (no Prefect context in unit tests) and
    # make load_config raise early so we don't need a real vault.
    import logging
    with patch("flow_index_vault.get_run_logger", return_value=logging.getLogger("test")):
        with patch("flow_index_vault.load_config", side_effect=ValueError("test abort")):
            with pytest.raises(ValueError, match="test abort"):
                from flow_index_vault import index_vault_flow
                index_vault_flow.fn("dummy.yaml")

    assert "stale_key" not in _RUNTIME
    assert "_warnings" not in _RUNTIME


# --- zip strict=True catches length mismatch (Fix 4) ---


def test_zip_strict_catches_length_mismatch():
    """zip(strict=True) should raise ValueError when iterables differ in length.

    Documents the safety mechanism added to flow_index_vault.py.
    """
    with pytest.raises(ValueError):
        list(zip([1, 2, 3], [4, 5], strict=True))
