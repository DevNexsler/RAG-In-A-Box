"""Tests for scan_vault_task and glob matching."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from flow_index_vault import (
    scan_vault_task,
    _matches_any,
    _split_markdown_by_headings,
    _build_chunk_context,
    _split_section,
    _RUNTIME,
    write_index_metadata_task,
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

@pytest.fixture(autouse=True)
def _clear_doc_id_store():
    """Ensure scan tests run without a doc_id_store (tests expect rel_path as doc_id)."""
    saved = _RUNTIME.pop("doc_id_store", None)
    yield
    if saved is not None:
        _RUNTIME["doc_id_store"] = saved


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


def test_scan_skips_zero_byte_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "empty.pdf").write_bytes(b"")
        (root / "note.md").write_text("content")

        records = scan_vault_task.fn(root, ["**/*.pdf", "**/*.md"], [])

        doc_ids = {r["doc_id"] for r in records}
        assert doc_ids == {"note.md"}
        assert (root / "empty.pdf").exists()


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


# --- process_doc_task communication context ---


def test_index_flow_builds_context_provider_from_scanned_records():
    from communication_context import (
        build_context_provider_from_records,
        communication_item_from_record,
    )
    from sources.base import SourceRecord

    records = [
        {
            "doc_id": "comm::1",
            "source_name": "comm",
            "source_type": "pg_message",
            "rel_path": "zoho/1",
        },
        {
            "doc_id": "comm::2",
            "source_name": "comm",
            "source_type": "pg_message",
            "rel_path": "zoho/2",
        },
    ]
    source_records = {
        "comm::1": SourceRecord(
            doc_id="1",
            source_type="pg_message",
            natural_key="zoho/1",
            mtime=1.0,
            size=6,
            metadata={
                "_text": "Unit E",
                "source": "zoho_cliq",
                "source_message_id": "s1",
                "message_id": "1",
                "channel_id": "chan",
                "sent_at": "2026-01-01T10:00:00Z",
                "sender": "A",
            },
        ),
        "comm::2": SourceRecord(
            doc_id="2",
            source_type="pg_message",
            natural_key="zoho/2",
            mtime=2.0,
            size=9,
            metadata={
                "_text": "Also this",
                "source": "zoho_cliq",
                "source_message_id": "s2",
                "message_id": "2",
                "channel_id": "chan",
                "sent_at": "2026-01-01T10:00:10Z",
                "sender": "A",
            },
        ),
    }

    provider = build_context_provider_from_records(records, source_records, {})
    assert provider is not None

    item = communication_item_from_record(
        records[1],
        source_records["comm::2"].metadata,
    )
    assert item is not None
    envelope = provider.get_context_envelope(item)

    assert envelope.nearest_nonempty_before.text == "Unit E"


def test_process_doc_task_passes_context_to_enrichment(monkeypatch):
    """Communication context reaches enrichment and stored metadata."""
    from communication_context import (
        CommunicationMessage,
        ContextEnvelope,
    )
    from extractors import ExtractionResult
    from flow_index_vault import process_doc_task
    from sources.base import SourceRecord

    captured = {}

    class FakeSource:
        name = "documents"

        def extract(self, record):
            return ExtractionResult.from_text("image description", frontmatter={})

    class FakeProvider:
        def get_context_envelope(self, item):
            captured["item"] = item
            return ContextEnvelope(
                primary_item=item,
                same_channel_before=[
                    CommunicationMessage(
                        message_id="m1",
                        source_message_id="src-m1",
                        sender="Joycelyn",
                        sent_at="2026-01-01T10:00:00Z",
                        text="This is for Unit E",
                    )
                ],
                same_channel_after=[
                    CommunicationMessage(
                        message_id="m3",
                        sent_at="2026-01-01T10:00:03Z",
                        text="",
                    )
                ],
            )

    class FakeStore:
        def upsert_nodes(self, nodes):
            captured["metadata"] = nodes[0].metadata

    class FakeEmbed:
        def embed_texts(self, texts):
            return [[0.1] * 768 for _ in texts]

    def fake_enrich_document(*args, **kwargs):
        captured["context_text"] = kwargs.get("context_text", "")
        return {
            "enr_summary": "summary",
            "enr_doc_type": "image",
            "enr_entities_people": "",
            "enr_entities_places": "",
            "enr_entities_orgs": "",
            "enr_entities_dates": "",
            "enr_topics": "",
            "enr_keywords": "",
            "enr_key_facts": "",
            "enr_suggested_tags": "",
            "enr_suggested_folder": "",
            "enr_importance": "0.5",
            "enr_atomic_entities_people": "",
            "enr_atomic_entities_places": "",
            "enr_atomic_entities_orgs": "",
            "enr_atomic_entities_dates": "",
            "enr_atomic_topics": "",
            "enr_context_entities_people": "",
            "enr_context_entities_places": "Unit E",
            "enr_context_entities_orgs": "",
            "enr_context_entities_dates": "",
            "enr_context_topics": "",
            "enr_context_key_facts": "",
            "enr_context_relationship": "batch_label",
            "enr_context_confidence": "high",
            "enr_context_source_message_ids": "m1",
            "enr_context_warning": "",
        }

    monkeypatch.setattr("flow_index_vault.enrich_document", fake_enrich_document)
    monkeypatch.setattr("flow_index_vault.get_run_logger", lambda: MagicMock())
    _RUNTIME.clear()
    _RUNTIME.update(
        {
            "store": FakeStore(),
            "embed_provider": FakeEmbed(),
            "splitter": _FakeSplitter(),
            "config": {},
            "sources_by_name": {"documents": FakeSource()},
            "source_records_by_ns_doc_id": {
                "documents::photo": SourceRecord(
                    doc_id="photo",
                    source_type="img",
                    natural_key="photo.jpg",
                    mtime=1.0,
                    size=10,
                    metadata={
                        "source": "zoho_cliq",
                        "channel_id": "chan",
                        "sent_at": "2026-01-01T10:00:01Z",
                        "message_id": "m2",
                    },
                )
            },
            "communication_context_provider": FakeProvider(),
            "llm_generator": object(),
        }
    )

    process_doc_task.fn(
        {
            "doc_id": "documents::photo",
            "rel_path": "photo.jpg",
            "mtime": 1.0,
            "size": 10,
            "source_type": "img",
            "source_name": "documents",
        }
    )

    assert captured["item"].origin_source == "zoho_cliq"
    assert "BEFORE" in captured["context_text"]
    assert "This is for Unit E" in captured["context_text"]
    assert "message_id=m1" in captured["context_text"]
    assert captured["metadata"]["enr_context_entities_places"] == "Unit E"
    assert captured["metadata"]["enr_context_confidence"] == "high"
    assert captured["metadata"]["context_before_message_ids"] == "m1"
    assert captured["metadata"]["context_nearest_before_message_id"] == "m1"


def test_process_doc_task_includes_attachment_message_body_in_enrichment(monkeypatch):
    """Same-message attachment captions become primary enrichment text."""
    from extractors import ExtractionResult
    from flow_index_vault import process_doc_task
    from sources.base import SourceRecord

    captured = {}

    class FakeSource:
        name = "documents"

        def extract(self, record):
            return ExtractionResult.from_text("image shows a kitchen wall", frontmatter={})

    class FakeStore:
        def upsert_nodes(self, nodes):
            captured["metadata"] = nodes[0].metadata

    class FakeEmbed:
        def embed_texts(self, texts):
            captured["embedded_text"] = texts[0]
            return [[0.1] * 768 for _ in texts]

    def fake_enrich_document(*args, **kwargs):
        captured["enrichment_text"] = kwargs["text"] if "text" in kwargs else args[0]
        return {
            "enr_summary": "summary",
            "enr_doc_type": "image",
            "enr_entities_people": "",
            "enr_entities_places": "",
            "enr_entities_orgs": "",
            "enr_entities_dates": "",
            "enr_topics": "",
            "enr_keywords": "",
            "enr_key_facts": "",
            "enr_suggested_tags": "",
            "enr_suggested_folder": "",
            "enr_importance": "0.5",
            "enr_atomic_entities_people": "",
            "enr_atomic_entities_places": "",
            "enr_atomic_entities_orgs": "",
            "enr_atomic_entities_dates": "",
            "enr_atomic_topics": "",
            "enr_context_entities_people": "",
            "enr_context_entities_places": "",
            "enr_context_entities_orgs": "",
            "enr_context_entities_dates": "",
            "enr_context_topics": "",
            "enr_context_key_facts": "",
            "enr_context_relationship": "",
            "enr_context_confidence": "",
            "enr_context_source_message_ids": "",
            "enr_context_warning": "",
        }

    monkeypatch.setattr("flow_index_vault.enrich_document", fake_enrich_document)
    monkeypatch.setattr("flow_index_vault.get_run_logger", lambda: MagicMock())
    _RUNTIME.clear()
    _RUNTIME.update(
        {
            "store": FakeStore(),
            "embed_provider": FakeEmbed(),
            "splitter": _FakeSplitter(),
            "config": {},
            "sources_by_name": {"documents": FakeSource()},
            "source_records_by_ns_doc_id": {
                "documents::photo": SourceRecord(
                    doc_id="photo",
                    source_type="img",
                    natural_key="photo.jpg",
                    mtime=1.0,
                    size=10,
                    metadata={
                        "message_body": "163 washington # 2",
                        "source": "zoho_cliq",
                        "channel_id": "chan",
                        "sent_at": "2026-05-05T19:30:00Z",
                        "message_id": "13579",
                    },
                )
            },
            "llm_generator": object(),
        }
    )

    process_doc_task.fn(
        {
            "doc_id": "documents::photo",
            "rel_path": "photo.jpg",
            "mtime": 1.0,
            "size": 10,
            "source_type": "img",
            "source_name": "documents",
        }
    )

    assert "Communication message/caption: 163 washington # 2" in captured["enrichment_text"]
    assert "Attachment content:" in captured["enrichment_text"]
    assert "image shows a kitchen wall" in captured["enrichment_text"]
    assert captured["metadata"]["message_body"] == "163 washington # 2"
    assert "163 washington # 2" in captured["embedded_text"]


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


def test_missing_fts_rebuilds_on_noop_index_update(tmp_path):
    """No-op index updates should rebuild FTS if the index health check is red."""
    from sources.base import SourceRecord
    from flow_index_vault import index_vault_flow

    fake_store = MagicMock()
    fake_store.list_doc_ids.return_value = ["documents::doc-1"]
    fake_store.list_doc_mtimes.return_value = {"documents::doc-1": 1.0}
    fake_store.count_chunks.return_value = 1
    fake_store.fts_available.return_value = False

    fake_registry = MagicMock()
    fake_registry.count.return_value = 1

    fake_taxonomy = MagicMock()
    fake_taxonomy.count.return_value = 0

    class _FakeSource:
        name = "documents"

        def scan(self):
            return iter([
                SourceRecord(
                    doc_id="doc-1",
                    natural_key="doc-1.txt",
                    source_type="txt",
                    mtime=1.0,
                    size=4,
                    metadata={"abs_path": str(tmp_path / "doc-1.txt"), "ext": "txt"},
                )
            ])

        def set_ocr_provider(self, provider):
            return None

        def close(self):
            return None

    config = {
        "index_root": str(tmp_path / "index"),
        "sources": [{"type": "filesystem", "name": "documents", "root": str(tmp_path)}],
        "chunking": {"max_chars": 1800, "overlap": 200, "semantic": {"enabled": False}},
        "enrichment": {"enabled": False},
        "ocr": {"enabled": False},
        "lancedb": {"table": "chunks"},
        "pdf": {},
        "logging": {"level": "WARNING"},
    }

    with patch("flow_index_vault.get_run_logger", return_value=MagicMock()):
        with patch("flow_index_vault.load_config", return_value=config):
            with patch("flow_index_vault.open_store_with_recovery", return_value=fake_store):
                with patch("flow_index_vault.DocIDStore", return_value=fake_registry):
                    with patch("flow_index_vault.build_embed_provider", return_value=MagicMock()):
                        with patch("flow_index_vault.build_ocr_provider", return_value=None):
                            with patch("sources.build_source", return_value=_FakeSource()):
                                with patch("core.taxonomy.load_taxonomy_store", return_value=fake_taxonomy):
                                    with patch("flow_index_vault.diff_index_task", return_value=([], [])):
                                        with patch("flow_index_vault.process_doc_task"):
                                            with patch("flow_index_vault.delete_docs_task"):
                                                with patch("flow_index_vault.index_stats_task"):
                                                    with patch("flow_index_vault.write_index_metadata_task"):
                                                        index_vault_flow.fn("dummy.yaml")

    fake_store.create_fts_index.assert_called_once_with()


def test_index_flow_syncs_folder_taxonomy_from_sources(tmp_path):
    """Index flow should sync real folder paths into taxonomy before enrichment."""
    from sources.base import SourceRecord
    from flow_index_vault import index_vault_flow

    fake_store = MagicMock()
    fake_store.list_doc_ids.return_value = []
    fake_store.list_doc_mtimes.return_value = {}
    fake_store.count_chunks.return_value = 0
    fake_store.fts_available.return_value = True

    fake_registry = MagicMock()
    fake_registry.count.return_value = 1

    fake_taxonomy = MagicMock()
    fake_taxonomy.count.return_value = 1

    class _FakeSource:
        name = "documents"

        def scan(self):
            return iter([
                SourceRecord(
                    doc_id="doc-1",
                    natural_key="folder/doc-1.txt",
                    source_type="txt",
                    mtime=1.0,
                    size=4,
                    metadata={"abs_path": str(tmp_path / "folder" / "doc-1.txt"), "ext": "txt"},
                )
            ])

        def set_ocr_provider(self, provider):
            return None

        def close(self):
            return None

    config = {
        "index_root": str(tmp_path / "index"),
        "sources": [{"type": "filesystem", "name": "documents", "root": str(tmp_path)}],
        "chunking": {"max_chars": 1800, "overlap": 200, "semantic": {"enabled": False}},
        "enrichment": {"enabled": False},
        "ocr": {"enabled": False},
        "lancedb": {"table": "chunks"},
        "pdf": {},
        "logging": {"level": "WARNING"},
    }

    with patch("flow_index_vault.get_run_logger", return_value=MagicMock()):
        with patch("flow_index_vault.load_config", return_value=config):
            with patch("flow_index_vault.LanceDBStore", return_value=fake_store):
                with patch("flow_index_vault.DocIDStore", return_value=fake_registry):
                    with patch("flow_index_vault.build_embed_provider", return_value=MagicMock()):
                        with patch("flow_index_vault.build_ocr_provider", return_value=None):
                            with patch("sources.build_source", return_value=_FakeSource()):
                                with patch("core.taxonomy.load_taxonomy_store", return_value=fake_taxonomy):
                                    with patch("core.taxonomy.sync_folder_taxonomy_from_sources", return_value={
                                        "sources": 1,
                                        "discovered": 1,
                                        "added": 1,
                                        "existing": 0,
                                    }) as sync_mock:
                                        with patch("flow_index_vault.diff_index_task", return_value=([], [])):
                                            with patch("flow_index_vault.delete_docs_task"):
                                                with patch("flow_index_vault.index_stats_task"):
                                                    with patch("flow_index_vault.write_index_metadata_task"):
                                                        index_vault_flow.fn("dummy.yaml")

    sync_mock.assert_called_once()


def test_index_flow_source_scope_deletes_only_selected_source(tmp_path):
    """Source-scoped indexing must not delete rows from unscanned sources."""
    from sources.base import SourceRecord
    from flow_index_vault import index_vault_flow

    fake_store = MagicMock()
    fake_store.list_doc_ids.return_value = []
    fake_store.list_doc_mtimes.return_value = {
        "documents::old-doc": 1.0,
        "sor::old-task": 1.0,
    }
    fake_store.count_chunks.return_value = 0
    fake_store.fts_available.return_value = True

    fake_registry = MagicMock()
    fake_registry.count.return_value = 1

    fake_taxonomy = MagicMock()
    fake_taxonomy.count.return_value = 0

    class _FakeSource:
        def __init__(self, name, records):
            self.name = name
            self.records = records
            self.scanned = False

        def scan(self):
            self.scanned = True
            return iter(self.records)

        def set_ocr_provider(self, provider):
            return None

        def set_media_provider(self, provider):
            return None

    documents_source = _FakeSource(
        "documents",
        [
            SourceRecord(
                doc_id="new-doc",
                natural_key="new-doc.md",
                source_type="md",
                mtime=2.0,
                size=7,
                metadata={"abs_path": str(tmp_path / "new-doc.md"), "ext": "md"},
            )
        ],
    )
    sor_source = _FakeSource(
        "sor",
        [
            SourceRecord(
                doc_id="new-task",
                natural_key="task/1",
                source_type="task",
                mtime=2.0,
                size=7,
                metadata={"abs_path": "task/1", "ext": "task"},
            )
        ],
    )

    def build_fake_source(source_cfg, registry, pdf_config):
        return {"documents": documents_source, "sor": sor_source}[source_cfg["name"]]

    config = {
        "index_root": str(tmp_path / "index"),
        "sources": [
            {"type": "filesystem", "name": "documents", "root": str(tmp_path)},
            {"type": "postgres", "name": "sor"},
        ],
        "chunking": {"max_chars": 1800, "overlap": 200, "semantic": {"enabled": False}},
        "enrichment": {"enabled": False},
        "ocr": {"enabled": False},
        "media": {"enabled": False},
        "lancedb": {"table": "chunks"},
        "pdf": {},
        "logging": {"level": "WARNING"},
    }

    with patch("flow_index_vault.get_run_logger", return_value=MagicMock()):
        with patch("flow_index_vault.load_config", return_value=config):
            with patch("flow_index_vault.open_store_with_recovery", return_value=fake_store):
                with patch("flow_index_vault.DocIDStore", return_value=fake_registry):
                    with patch("flow_index_vault.build_embed_provider", return_value=MagicMock()):
                        with patch("flow_index_vault.build_ocr_provider", return_value=None):
                            with patch("flow_index_vault.build_media_provider", return_value=None):
                                with patch("sources.build_source", side_effect=build_fake_source):
                                    with patch("core.taxonomy.load_taxonomy_store", return_value=fake_taxonomy):
                                        with patch("core.taxonomy.sync_folder_taxonomy_from_sources", return_value={
                                            "sources": 1,
                                            "discovered": 0,
                                            "added": 0,
                                            "existing": 0,
                                        }):
                                            with patch("flow_index_vault._process_docs", return_value=[]):
                                                with patch("flow_index_vault.delete_docs_task") as delete_mock:
                                                    with patch("flow_index_vault.index_stats_task"):
                                                        with patch("flow_index_vault.write_index_metadata_task"):
                                                            index_vault_flow.fn("dummy.yaml", source_name="sor")

    assert documents_source.scanned is False
    assert sor_source.scanned is True
    delete_mock.assert_called_once_with(["sor::old-task"])


def test_index_flow_source_scope_skips_global_empty_registry_migration(tmp_path):
    """A source-scoped run must never wipe the whole table for registry migration."""
    from flow_index_vault import index_vault_flow

    fake_store = MagicMock()
    fake_store.list_doc_ids.return_value = ["documents::old-doc"]
    fake_store.list_doc_mtimes.return_value = {}
    fake_store.count_chunks.return_value = 0
    fake_store.fts_available.return_value = True

    fake_registry = MagicMock()
    fake_registry.count.return_value = 0

    fake_taxonomy = MagicMock()
    fake_taxonomy.count.return_value = 0

    class _FakeSource:
        name = "sor"

        def scan(self):
            return iter([])

        def set_ocr_provider(self, provider):
            return None

        def set_media_provider(self, provider):
            return None

    config = {
        "index_root": str(tmp_path / "index"),
        "sources": [{"type": "postgres", "name": "sor"}],
        "chunking": {"max_chars": 1800, "overlap": 200, "semantic": {"enabled": False}},
        "enrichment": {"enabled": False},
        "ocr": {"enabled": False},
        "media": {"enabled": False},
        "lancedb": {"table": "chunks"},
        "pdf": {},
        "logging": {"level": "WARNING"},
    }

    with patch("flow_index_vault.get_run_logger", return_value=MagicMock()):
        with patch("flow_index_vault.load_config", return_value=config):
            with patch("flow_index_vault.open_store_with_recovery", return_value=fake_store):
                with patch("flow_index_vault.DocIDStore", return_value=fake_registry):
                    with patch("flow_index_vault.build_embed_provider", return_value=MagicMock()):
                        with patch("flow_index_vault.build_ocr_provider", return_value=None):
                            with patch("flow_index_vault.build_media_provider", return_value=None):
                                with patch("sources.build_source", return_value=_FakeSource()):
                                    with patch("core.taxonomy.load_taxonomy_store", return_value=fake_taxonomy):
                                        with patch("core.taxonomy.sync_folder_taxonomy_from_sources", return_value={
                                            "sources": 1,
                                            "discovered": 0,
                                            "added": 0,
                                            "existing": 0,
                                        }):
                                            with patch("flow_index_vault._process_docs", return_value=[]):
                                                with patch("flow_index_vault.delete_docs_task"):
                                                    with patch("flow_index_vault.index_stats_task"):
                                                        with patch("flow_index_vault.write_index_metadata_task"):
                                                            with patch("lancedb.connect") as connect_mock:
                                                                index_vault_flow.fn("dummy.yaml", source_name="sor")

    connect_mock.assert_not_called()


def test_index_flow_injects_media_provider_into_sources(tmp_path):
    """Index flow should inject optional media provider into filesystem sources."""
    from flow_index_vault import index_vault_flow

    fake_store = MagicMock()
    fake_store.list_doc_ids.return_value = []
    fake_store.list_doc_mtimes.return_value = {}
    fake_store.count_chunks.return_value = 0
    fake_store.fts_available.return_value = True

    fake_registry = MagicMock()
    fake_registry.count.return_value = 1

    fake_taxonomy = MagicMock()
    fake_taxonomy.count.return_value = 0
    fake_media_provider = object()

    class _FakeSource:
        name = "documents"

        def __init__(self):
            self.media_provider = None

        def scan(self):
            return iter([])

        def set_ocr_provider(self, provider):
            return None

        def set_media_provider(self, provider):
            self.media_provider = provider

        def close(self):
            return None

    fake_source = _FakeSource()
    config = {
        "index_root": str(tmp_path / "index"),
        "sources": [{"type": "filesystem", "name": "documents", "root": str(tmp_path)}],
        "chunking": {"max_chars": 1800, "overlap": 200, "semantic": {"enabled": False}},
        "enrichment": {"enabled": False},
        "ocr": {"enabled": False},
        "media": {"enabled": True},
        "lancedb": {"table": "chunks"},
        "pdf": {},
        "logging": {"level": "WARNING"},
    }

    with patch("flow_index_vault.get_run_logger", return_value=MagicMock()):
        with patch("flow_index_vault.load_config", return_value=config):
            with patch("flow_index_vault.LanceDBStore", return_value=fake_store):
                with patch("flow_index_vault.DocIDStore", return_value=fake_registry):
                    with patch("flow_index_vault.build_embed_provider", return_value=MagicMock()):
                        with patch("flow_index_vault.build_ocr_provider", return_value=None):
                            with patch("flow_index_vault.build_media_provider", return_value=fake_media_provider):
                                with patch("sources.build_source", return_value=fake_source):
                                    with patch("core.taxonomy.load_taxonomy_store", return_value=fake_taxonomy):
                                        with patch("flow_index_vault.diff_index_task", return_value=([], [])):
                                            with patch("flow_index_vault.delete_docs_task"):
                                                with patch("flow_index_vault.index_stats_task"):
                                                    with patch("flow_index_vault.write_index_metadata_task"):
                                                        index_vault_flow.fn("dummy.yaml")

    assert fake_source.media_provider is fake_media_provider


# --- zip strict=True catches length mismatch (Fix 4) ---


def test_zip_strict_catches_length_mismatch():
    """zip(strict=True) should raise ValueError when iterables differ in length.

    Documents the safety mechanism added to flow_index_vault.py.
    """
    with pytest.raises(ValueError):
        list(zip([1, 2, 3], [4, 5], strict=True))


def test_write_index_metadata_counts_warnings_separately(tmp_path):
    """Warning counts must not masquerade as document processing failures."""
    write_index_metadata_task.fn(
        tmp_path,
        doc_count=2,
        chunk_count=3,
        failed_docs=["broken.pdf"],
        warnings=[
            "enrichment_failed:sor::1:402 Payment Required",
            "enrichment_failed:sor::2:402 Payment Required",
            "fts_rebuild_failed:timeout",
        ],
    )

    import json

    meta = json.loads((tmp_path / "index_metadata.json").read_text())
    assert meta["failed_count"] == 1
    assert meta["warning_count"] == 3
    assert meta["warning_counts"] == {
        "enrichment_failed": 2,
        "fts_rebuild_failed": 1,
    }
    assert meta["enrichment_failed_count"] == 2
