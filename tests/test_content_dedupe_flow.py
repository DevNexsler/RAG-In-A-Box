"""Integration tests for the exact-content dedupe gate in process_doc_task.

The same bytes arriving via different paths (one document attached to several
emails) must index once: first-seen path is canonical, later copies skip the
pipeline entirely, and the canonical carries duplicate provenance in its
index metadata and sidecar.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest
import blake3

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


def test_provider_error_sidecars_do_not_become_content_duplicates(runtime):
    docs_root, store, registry = runtime
    payload = json.dumps(
        {
            "server": "qwen3-vl",
            "tool": "vl_describe",
            "error": "Call to qwen3-vl timed out after 60000ms",
            "issue": {
                "kind": "offline",
                "rawMessage": "Call to qwen3-vl timed out after 60000ms",
            },
        }
    )
    a = _make_doc(docs_root, "artifacts/first.jpg.vl.json", payload, "00001")
    b = _make_doc(docs_root, "artifacts/second.jpg.vl.json", payload, "00002")
    a["ext"] = "json"
    b["ext"] = "json"
    _register(registry, a)
    _register(registry, b)

    degradations = []
    for doc in (a, b):
        fiv.begin_degradation_capture()
        fiv.process_doc_task.fn(doc)
        degradations.append(fiv.collect_degradations())

    assert registry.duplicate_refs_for_canonical("00001") == []
    assert store.list_doc_ids() == []
    assert [[d.reason for d in noted] for noted in degradations] == [
        ["vision_sidecar_failed"],
        ["vision_sidecar_failed"],
    ]
    assert all(noted[0].transient for noted in degradations)


def test_provider_error_sidecar_is_retry_pending_when_dedupe_disabled(runtime):
    docs_root, store, registry = runtime
    payload = json.dumps(
        {
            "server": "qwen3-vl",
            "tool": "vl_describe",
            "error": "provider offline",
            "issue": {"kind": "offline", "rawMessage": "provider offline"},
        }
    )
    doc = _make_doc(docs_root, "artifacts/photo.jpg.vl.json", payload, "00001")
    doc["ext"] = "json"
    _register(registry, doc)
    fiv._RUNTIME["config"]["dedupe"]["enabled"] = False

    fiv.begin_degradation_capture()
    fiv.process_doc_task.fn(doc)

    assert store.list_doc_ids() == []
    assert [d.reason for d in fiv.collect_degradations()] == [
        "vision_sidecar_failed"
    ]
    assert fiv.collect_skips() == []


def test_provider_error_sidecar_repairs_legacy_false_dedupe_cohort(runtime):
    docs_root, store, registry = runtime
    payload = json.dumps(
        {
            "server": "qwen3-vl",
            "tool": "vl_describe",
            "error": "provider offline",
            "issue": {"kind": "offline", "rawMessage": "provider offline"},
        }
    )
    a = _make_doc(docs_root, "artifacts/first.jpg.vl.json", payload, "00001")
    b = _make_doc(docs_root, "artifacts/second.jpg.vl.json", payload, "00002")
    a["ext"] = "json"
    b["ext"] = "json"
    _register(registry, a)
    _register(registry, b)

    raw = Path(a["abs_path"]).read_bytes()
    digest = blake3.blake3(raw).digest()
    registry.claim_canonical_by_exact_hash(
        "00001", len(raw), digest, hash_algo="blake3"
    )
    registry.claim_canonical_by_exact_hash(
        "00002", len(raw), digest, hash_algo="blake3"
    )

    for doc in (a, b):
        fiv.begin_degradation_capture()
        fiv.process_doc_task.fn(doc)

    rows = registry._conn.execute(
        """
        SELECT doc_id, size_bytes, content_hash, hash_algo,
               dedupe_status, canonical_doc_id
        FROM doc_registry
        ORDER BY doc_id
        """
    ).fetchall()
    assert rows == [
        ("00001", None, None, None, "canonical", None),
        ("00002", None, None, None, "canonical", None),
    ]
    assert store.list_doc_ids() == []


def test_provider_error_sidecar_stays_excluded_when_legacy_cleanup_fails(runtime):
    docs_root, store, registry = runtime
    payload = json.dumps(
        {
            "server": "qwen3-vl",
            "tool": "vl_describe",
            "error": "provider offline",
            "issue": {"kind": "offline", "rawMessage": "provider offline"},
        }
    )
    doc = _make_doc(docs_root, "artifacts/photo.jpg.vl.json", payload, "00001")
    doc["ext"] = "json"
    _register(registry, doc)
    raw = Path(doc["abs_path"]).read_bytes()
    registry.claim_canonical_by_exact_hash(
        "00001", len(raw), blake3.blake3(raw).digest(), hash_algo="blake3"
    )

    fiv.begin_degradation_capture()
    with patch.object(store, "delete_by_doc_ids", side_effect=OSError("busy")):
        fiv.process_doc_task.fn(doc)

    assert store.list_doc_ids() == []
    noted = fiv.collect_degradations()
    assert [degradation.reason for degradation in noted] == ["vision_sidecar_failed"]


def test_missing_canonical_is_reopened_before_duplicate_skip(runtime):
    docs_root, store, registry = runtime
    body = "same valid document bytes"
    absent = _make_doc(docs_root, "f/absent.md", body, "00001")
    available = _make_doc(docs_root, "g/available.md", body, "00002")
    _register(registry, absent)
    _register(registry, available)

    raw = Path(absent["abs_path"]).read_bytes()
    digest = blake3.blake3(raw).digest()
    registry.claim_canonical_by_exact_hash(
        "00001", len(raw), digest, hash_algo="blake3"
    )
    registry.claim_canonical_by_exact_hash(
        "00002", len(raw), digest, hash_algo="blake3"
    )

    fiv.process_doc_task.fn(available)
    fiv.process_doc_task.fn(absent)

    indexed = set(store.list_doc_ids())
    assert indexed == {"documents::00002"}
    assert registry.duplicate_refs_for_canonical("00002")[0]["doc_id"] == "00001"


def test_concurrent_equal_content_waits_for_indexed_canonical(runtime):
    docs_root, store, registry = runtime
    body = "equal concurrent content"
    a = _make_doc(docs_root, "f/one.md", body, "00001")
    b = _make_doc(docs_root, "g/two.md", body, "00002")
    _register(registry, a)
    _register(registry, b)

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(fiv.process_doc_task.fn, (a, b)))

    indexed = set(store.list_doc_ids())
    assert len(indexed) == 1
    canonical = next(iter(indexed)).split("::", 1)[1]
    assert len(registry.duplicate_refs_for_canonical(canonical)) == 1


def test_successful_provider_sidecar_is_not_classified_as_error_artifact():
    payload = json.dumps(
        {
            "server": "qwen3-vl",
            "tool": "vl_describe",
            "result": "A distinct image description",
            "issue": None,
        }
    )
    assert fiv._provider_error_artifact(payload.encode(), "json") is None


def test_process_doc_task_emits_memory_subphases(runtime):
    docs_root, store, registry = runtime
    doc = _make_doc(docs_root, "notes/one.md", "# One\nBody text", "00001")
    _register(registry, doc)
    spans: list[tuple[str, str, dict]] = []

    class Observer:
        @contextmanager
        def measure(self, subphase, **fields):
            spans.append(("start", subphase, fields))
            try:
                yield
            finally:
                spans.append(("finish", subphase, fields))

    observer = Observer()
    fiv._RUNTIME["memory_observer"] = observer
    store.set_memory_observer(observer)

    fiv.process_doc_task.fn(doc)

    started = [subphase for event, subphase, _ in spans if event == "start"]
    assert started == [
        "extract",
        "enrichment",
        "embed",
        "storage_schema",
        "storage_delete",
        "storage_add",
    ]
    assert all(fields.get("doc_id") == doc["doc_id"] for _, _, fields in spans)


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
