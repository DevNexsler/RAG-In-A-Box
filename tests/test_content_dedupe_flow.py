"""Integration tests for the exact-content dedupe gate in process_doc_task.

The same bytes arriving via different paths (one document attached to several
emails) must index once: first-seen path is canonical, later copies skip the
pipeline entirely, and the canonical carries duplicate provenance in its
index metadata and sidecar.
"""

import json
from pathlib import Path

import pytest

from unittest.mock import MagicMock, patch

import flow_index_vault as fiv
from doc_id_store import DocIDStore
from lancedb_store import LanceDBStore


class _MockEmbed:
    def embed_texts(self, texts):
        return [[0.1] * 768 for _ in texts]

    def embed_query(self, q):
        return [0.1] * 768


@pytest.fixture
def runtime(tmp_path):
    logger_patch = patch("flow_index_vault.get_run_logger", return_value=MagicMock())
    logger_patch.start()
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    store = LanceDBStore(tmp_path / "index", "chunks")
    registry = DocIDStore(tmp_path / "index" / "registry.db")
    from llama_index.core.node_parser import SentenceSplitter

    fiv._RUNTIME.clear()
    fiv._RUNTIME.update({
        "store": store,
        "doc_id_store": registry,
        "embed_provider": _MockEmbed(),
        "splitter": SentenceSplitter(chunk_size=512, chunk_overlap=20),
        "config": {
            "dedupe": {
                "enabled": True,
                "skip_duplicate_indexing": True,
                "update_canonical_metadata": True,
            },
            "enrichment": {"enabled": False},
            "pdf": {},
        },
    })
    yield docs_root, store, registry
    logger_patch.stop()
    fiv._RUNTIME.clear()
    registry.close()


def _make_doc(docs_root: Path, rel: str, content: str, doc_id: str) -> dict:
    p = docs_root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return {
        "doc_id": f"documents::{doc_id}",
        "rel_path": rel,
        "abs_path": str(p),
        "mtime": p.stat().st_mtime,
        "size": p.stat().st_size,
        "ext": "md",
        "source_name": "documents",
    }


def _register(registry: DocIDStore, doc: dict) -> None:
    registry.register(doc["doc_id"].split("::", 1)[1], doc["rel_path"])


def test_duplicate_content_indexes_once(runtime):
    docs_root, store, registry = runtime
    body = "# Lease agreement\nThe same lease document, byte for byte." * 5
    a = _make_doc(docs_root, "email-attachments/dan/msg1__mm0.md", body, "00001")
    b = _make_doc(docs_root, "email-attachments/nigel/msg2__mm0.md", body, "00002")
    _register(registry, a)
    _register(registry, b)

    fiv.process_doc_task.fn(a)
    fiv.process_doc_task.fn(b)

    doc_ids = set(r["doc_id"] for r in
                  store._vs.table.to_lance().to_table(columns=["doc_id"]).to_pylist())
    assert "documents::00001" in doc_ids
    assert "documents::00002" not in doc_ids  # duplicate skipped


def test_duplicate_marked_in_registry_with_canonical(runtime):
    docs_root, store, registry = runtime
    body = "identical bytes"
    a = _make_doc(docs_root, "f/one.md", body, "00001")
    b = _make_doc(docs_root, "g/two.md", body, "00002")
    _register(registry, a)
    _register(registry, b)

    fiv.process_doc_task.fn(a)
    fiv.process_doc_task.fn(b)

    refs = registry.duplicate_refs_for_canonical("00001")
    assert any(r["doc_id"] == "00002" for r in refs)


def test_canonical_sidecar_gets_duplicate_delivery_note(runtime):
    docs_root, store, registry = runtime
    body = "attachment bytes"
    a = _make_doc(docs_root, "email-attachments/dan/msgA__mm0.md", body, "00001")
    b = _make_doc(docs_root, "email-attachments/nigel/msgB__mm0.md", body, "00002")
    # canonical + duplicate sidecars in attachment-store style
    (docs_root / "email-attachments/dan/msgA__mm0.json").write_text(json.dumps(
        {"schema_version": 2, "source": "zoho_mail", "message": {"source_message_id": "<a@x>"}}))
    (docs_root / "email-attachments/nigel/msgB__mm0.json").write_text(json.dumps(
        {"schema_version": 2, "source": "zoho_mail",
         "message": {"source_message_id": "<b@x>", "from": {"address": "n@x.com"}}}))
    _register(registry, a)
    _register(registry, b)

    fiv.process_doc_task.fn(a)
    fiv.process_doc_task.fn(b)

    sidecar = json.loads((docs_root / "email-attachments/dan/msgA__mm0.json").read_text())
    deliveries = sidecar.get("duplicate_deliveries", [])
    assert len(deliveries) == 1
    assert deliveries[0]["rel_path"] == "email-attachments/nigel/msgB__mm0.md"
    assert deliveries[0]["message"]["source_message_id"] == "<b@x>"


def test_different_content_both_index(runtime):
    docs_root, store, registry = runtime
    a = _make_doc(docs_root, "f/one.md", "first unique content", "00001")
    b = _make_doc(docs_root, "g/two.md", "second different content", "00002")
    _register(registry, a)
    _register(registry, b)

    fiv.process_doc_task.fn(a)
    fiv.process_doc_task.fn(b)

    doc_ids = set(r["doc_id"] for r in
                  store._vs.table.to_lance().to_table(columns=["doc_id"]).to_pylist())
    assert {"documents::00001", "documents::00002"} <= doc_ids


def test_dedupe_disabled_indexes_everything(runtime):
    docs_root, store, registry = runtime
    fiv._RUNTIME["config"]["dedupe"]["enabled"] = False
    body = "identical bytes"
    a = _make_doc(docs_root, "f/one.md", body, "00001")
    b = _make_doc(docs_root, "g/two.md", body, "00002")
    _register(registry, a)
    _register(registry, b)

    fiv.process_doc_task.fn(a)
    fiv.process_doc_task.fn(b)

    doc_ids = set(r["doc_id"] for r in
                  store._vs.table.to_lance().to_table(columns=["doc_id"]).to_pylist())
    assert {"documents::00001", "documents::00002"} <= doc_ids
