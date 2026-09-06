"""Integration tests for the exact-content dedupe gate in process_doc_task.

The same bytes arriving via different paths (one document attached to several
emails) must index once: first-seen path is canonical, later copies skip the
pipeline entirely, and the canonical carries duplicate provenance in its
index metadata and sidecar.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Barrier, Event, Lock, Thread
from types import SimpleNamespace

import pytest
import blake3

from unittest.mock import MagicMock, patch

import flow_index_vault as fiv
from core.hook_outbox import HookOutbox
from doc_id_store import DocIDStore
from hooks.delivery import drain_due
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
            "index_root": str(tmp_path / "index"),
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
        ["vision_sidecar_failed:blocked_on_upstream"],
        ["vision_sidecar_failed:blocked_on_upstream"],
    ]
    assert all(not noted[0].transient for noted in degradations)


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
        "vision_sidecar_failed:blocked_on_upstream"
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
    assert [degradation.reason for degradation in noted] == [
        "vision_sidecar_failed:blocked_on_upstream"
    ]


def test_provider_error_cohort_cleanup_recovers_after_lance_delete_failure(runtime):
    docs_root, store, registry = runtime
    a = _make_doc(docs_root, "artifacts/first.jpg.vl.json", "stale first", "00001")
    b = _make_doc(docs_root, "artifacts/second.jpg.vl.json", "stale second", "00002")
    _register(registry, a)
    _register(registry, b)
    fiv.process_doc_task.fn(a)
    fiv.process_doc_task.fn(b)
    assert set(store.list_doc_ids()) == {"documents::00001", "documents::00002"}

    payload = json.dumps(
        {
            "server": "qwen3-vl",
            "tool": "vl_describe",
            "error": "provider offline",
            "issue": {"kind": "offline", "rawMessage": "provider offline"},
        }
    )
    for doc in (a, b):
        Path(doc["abs_path"]).write_text(payload)
        doc["ext"] = "json"
        bare_id = doc["doc_id"].split("::", 1)[1]
        registry.reset_exact_hash_cohort(bare_id)
    raw = payload.encode()
    digest = blake3.blake3(raw).digest()
    registry.claim_canonical_by_exact_hash(
        "00001", len(raw), digest, hash_algo="blake3"
    )
    registry.claim_canonical_by_exact_hash(
        "00002", len(raw), digest, hash_algo="blake3"
    )

    original_delete = store._vs.delete
    delete_attempts = 0

    def fail_first_delete(_vector_store, doc_id):
        nonlocal delete_attempts
        delete_attempts += 1
        if delete_attempts == 1:
            raise OSError("busy")
        return original_delete(doc_id)

    fiv.begin_degradation_capture()
    with patch.object(
        type(store._vs), "delete", autospec=True, side_effect=fail_first_delete
    ):
        fiv.process_doc_task.fn(a)
    retained_after_failure = registry.find_canonical_by_exact_hash(
        len(raw), digest, "blake3"
    )
    stale_after_failure = set(store.list_doc_ids())

    fiv.begin_degradation_capture()
    fiv.process_doc_task.fn(a)

    assert retained_after_failure is not None
    assert stale_after_failure == {"documents::00001"}
    assert registry.find_canonical_by_exact_hash(len(raw), digest, "blake3") is None
    assert store.list_doc_ids() == []


def test_provider_error_cohort_cleanup_blocks_identity_change_between_phases(runtime):
    docs_root, store, registry = runtime
    doc = _make_doc(docs_root, "artifacts/photo.jpg.vl.json", "stale", "00001")
    _register(registry, doc)
    fiv.process_doc_task.fn(doc)
    assert store.list_doc_ids() == ["documents::00001"]

    new_payload = b"replacement content"
    new_digest = blake3.blake3(new_payload).digest()
    second_registry = DocIDStore(registry.db_path)
    delete_entered = Event()
    release_delete = Event()
    mutation_started = Event()
    mutation_finished = Event()
    original_delete = store.delete_by_doc_ids

    def gated_delete(doc_ids):
        delete_entered.set()
        assert release_delete.wait(timeout=5)
        return original_delete(doc_ids)

    def mutate_identity():
        mutation_started.set()
        second_registry.update_dedupe_identity(
            "00001",
            size_bytes=len(new_payload),
            content_hash=new_digest,
            hash_algo="blake3",
            dedupe_status="canonical",
            canonical_doc_id=None,
        )
        mutation_finished.set()

    with (
        patch.object(store, "delete_by_doc_ids", side_effect=gated_delete),
        ThreadPoolExecutor(max_workers=2) as executor,
    ):
        cleanup = executor.submit(
            fiv._reset_invalid_dedupe_cohort,
            registry,
            store,
            "00001",
            "documents",
            MagicMock(),
        )
        assert delete_entered.wait(timeout=5)
        mutation = executor.submit(mutate_identity)
        assert mutation_started.wait(timeout=5)
        mutation_blocked_during_delete = not mutation_finished.wait(timeout=0.2)
        release_delete.set()
        cleanup.result(timeout=5)
        mutation.result(timeout=5)

    second_registry.close()
    assert mutation_blocked_during_delete
    assert registry.find_canonical_by_exact_hash(
        len(new_payload), new_digest, "blake3"
    ) is not None
    assert store.list_doc_ids() == []


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

    read_barrier = Barrier(2)
    canonical_upsert_entered = Event()
    release_canonical_upsert = Event()
    target_paths = {Path(a["abs_path"]), Path(b["abs_path"])}
    original_read_bytes = Path.read_bytes
    original_upsert_nodes = store.upsert_nodes

    def synchronized_read_bytes(path):
        raw = original_read_bytes(path)
        if path in target_paths:
            read_barrier.wait(timeout=5)
        return raw

    def gated_upsert(nodes):
        canonical_upsert_entered.set()
        assert release_canonical_upsert.wait(timeout=5)
        return original_upsert_nodes(nodes)

    with (
        patch.object(Path, "read_bytes", synchronized_read_bytes),
        patch.object(store, "upsert_nodes", side_effect=gated_upsert),
        ThreadPoolExecutor(max_workers=2) as executor,
    ):
        futures = [executor.submit(fiv.process_doc_task.fn, doc) for doc in (a, b)]
        assert canonical_upsert_entered.wait(timeout=5)
        release_canonical_upsert.set()
        for future in futures:
            future.result(timeout=10)

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


def test_provider_error_sidecar_without_issue_is_classified_as_non_content():
    payload = json.dumps(
        {
            "server": "qwen3-vl",
            "tool": "vl_describe",
            "error": "Call to qwen3-vl timed out after 60000ms",
        }
    )

    assert fiv._provider_error_artifact(payload.encode(), "json") == (
        "vision_sidecar_failed",
        True,
    )


def test_provider_error_sidecar_ignores_empty_content_placeholders():
    payload = json.dumps(
        {
            "server": "qwen3-vl",
            "tool": "vl_describe",
            "error": "provider offline",
            "result": None,
            "text": "",
        }
    )

    assert fiv._provider_error_artifact(payload.encode(), "json") == (
        "vision_sidecar_failed",
        True,
    )


def test_arbitrary_user_json_with_provider_like_fields_remains_content():
    payload = json.dumps(
        {
            "server": "production",
            "tool": "migration",
            "error": "Known historical incident",
            "content": "User-authored incident record",
        }
    )

    assert fiv._provider_error_artifact(payload.encode(), "json") is None


@pytest.mark.parametrize("value", [0, False])
def test_provider_like_user_json_with_scalar_content_remains_content(value):
    payload = json.dumps(
        {
            "server": "production",
            "tool": "migration",
            "error": "Known historical incident",
            "result": value,
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


def test_duplicate_video_delivery_indexes_context_alias_without_media_extraction(runtime):
    from communication_context import (
        SidecarContextProvider,
        communication_metadata_from_sidecar,
    )

    docs_root, store, registry = runtime
    body = "identical media bytes"
    canonical = _make_doc(docs_root, "f/canonical.md", body, "00001")
    duplicate = _make_doc(
        docs_root,
        "email-attachments/cesar/msgB__mm0.mp4",
        body,
        "00002",
    )
    duplicate["ext"] = "mp4"
    duplicate["source_type"] = "video"
    sidecar = docs_root / "email-attachments/cesar/msgB__mm0.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "source": "zoho_cliq",
                "message": {
                    "source_message_id": "attachment",
                    "sent_at": "2026-06-18T19:08:42Z",
                    "from": {"name": "Cesar"},
                },
                "channel": {"source_channel_id": "maintenance"},
                "media": {"media_index": 0, "media_type": "video/mp4"},
                "context": {
                    "same_channel_before": [
                        {
                            "source_message_id": "before",
                            "sent_at": "2026-06-18T19:07:07Z",
                            "text": "Were you able to complete Bullmans?",
                            "origin_source": "zoho_cliq",
                            "channel_id": "maintenance",
                        }
                    ],
                    "same_channel_after": [
                        {
                            "source_message_id": "after",
                            "sent_at": "2026-06-18T19:11:48Z",
                            "text": "Maybe tomorrow",
                            "origin_source": "zoho_cliq",
                            "channel_id": "maintenance",
                        }
                    ],
                },
            }
        )
    )
    _register(registry, canonical)
    _register(registry, duplicate)
    fiv.process_doc_task.fn(canonical)

    metadata = communication_metadata_from_sidecar(
        Path(duplicate["abs_path"]),
        sidecar,
    )
    fiv._RUNTIME["source_records_by_ns_doc_id"] = {
        duplicate["doc_id"]: SimpleNamespace(metadata=metadata)
    }
    fiv._RUNTIME["communication_context_provider"] = SidecarContextProvider()

    with patch(
        "flow_index_vault.extract_text",
        side_effect=AssertionError("duplicate media must not be extracted"),
    ):
        fiv.process_doc_task.fn(duplicate)

    alias_chunks = store.get_doc_chunks(duplicate["doc_id"])
    assert len(alias_chunks) == 1
    assert alias_chunks[0].source_type == "video"
    assert "Were you able to complete Bullmans?" in alias_chunks[0].text
    assert "Maybe tomorrow" in alias_chunks[0].text
    assert canonical["doc_id"] in alias_chunks[0].text
    assert set(store.list_doc_ids()) == {
        canonical["doc_id"],
        duplicate["doc_id"],
    }


def test_duplicate_callback_uses_alias_identity_and_canonical_payload(runtime, monkeypatch):
    docs_root, store, registry = runtime
    body = "Duplicate attachment body with indexed canonical content."
    canonical = _make_doc(docs_root, "f/canonical.md", body, "00001")
    duplicate = _make_doc(
        docs_root,
        "email-attachments/quo/attachment@00002@.md",
        body,
        "00002",
    )
    _register(registry, canonical)
    _register(registry, duplicate)
    fiv._RUNTIME["config"].update(
        {
            "index_root": str(docs_root.parent / "index"),
            "event_hooks": {
                "enabled": True,
                "hooks": [{"name": "cds", "events": ["document.indexed"]}],
            },
        }
    )
    monkeypatch.setattr(
        "flow_index_vault.drain_due",
        lambda outbox, **kwargs: {
            "accepted": 0,
            "retry_pending": 1,
            "redrive_required": 0,
        },
    )

    fiv.process_doc_task.fn(canonical)
    fiv.process_doc_task.fn(duplicate)

    deliveries = HookOutbox(fiv._RUNTIME["config"]["index_root"]).due(limit=4)
    event = next(
        delivery.event
        for delivery in deliveries
        if delivery.event["doc_id"] == duplicate["doc_id"]
    )
    assert event["doc_id"] == duplicate["doc_id"]
    assert event["rel_path"] == duplicate["rel_path"]
    assert body in event["text"]
    assert body in event["chunks"][0]["text"]
    assert event["metadata"]["enr_importance"] == "0.5"
    assert event["metadata"]["enr_importance_source"] == "default"


def test_duplicate_callback_delivers_one_request_per_event_id(runtime):
    """Fails if a dedup-skipped document's callback reaches the hook target twice.

    Every document dispatch drains the whole due outbox, so a second drainer
    that starts while a send is still in flight sends the same delivery again.
    Counted at the receiver on purpose: an accepted delivery row is deleted, so
    the extra POST leaves nothing behind in the outbox to assert on.
    """
    docs_root, store, registry = runtime
    body = "Duplicate attachment body with indexed canonical content."
    canonical = _make_doc(docs_root, "f/canonical.md", body, "00001")
    duplicate = _make_doc(
        docs_root,
        "email-attachments/quo/attachment@00002@.md",
        body,
        "00002",
    )
    _register(registry, canonical)
    _register(registry, duplicate)
    index_root = str(docs_root.parent / "index")
    received: list[str] = []
    overlapped = Event()

    class _Sink(BaseHTTPRequestHandler):
        def do_POST(self):
            event = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(event["event_id"])
            if event["doc_id"] == duplicate["doc_id"] and not overlapped.is_set():
                overlapped.set()
                # A second drainer — another index worker, or the scheduler
                # tick — runs while this delivery is still in flight.
                drain_due(HookOutbox(index_root), limit=64)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "updated"}')

        def log_message(self, *args):
            """Keep the sink out of the test log."""

    sink = ThreadingHTTPServer(("127.0.0.1", 0), _Sink)
    Thread(target=sink.serve_forever, daemon=True).start()
    fiv._RUNTIME["config"].update(
        {
            "index_root": index_root,
            "event_hooks": {
                "enabled": True,
                "hooks": [
                    {
                        "name": "cds",
                        "events": ["document.indexed"],
                        "url": f"http://127.0.0.1:{sink.server_address[1]}/hook",
                    }
                ],
            },
        }
    )
    try:
        fiv.process_doc_task.fn(canonical)
        fiv.process_doc_task.fn(duplicate)
    finally:
        sink.shutdown()
        sink.server_close()

    assert overlapped.is_set()
    assert sorted(received) == sorted(set(received))
    assert len(received) == 2  # canonical document + duplicate alias, once each


def test_duplicate_callback_payload_read_failure_does_not_fail_index(runtime, monkeypatch):
    docs_root, store, registry = runtime
    body = "Duplicate attachment body."
    canonical = _make_doc(docs_root, "f/canonical.md", body, "00001")
    duplicate = _make_doc(docs_root, "f/duplicate.md", body, "00002")
    _register(registry, canonical)
    _register(registry, duplicate)
    fiv.process_doc_task.fn(canonical)
    monkeypatch.setattr(
        store,
        "get_doc_chunks",
        MagicMock(side_effect=RuntimeError("canonical read unavailable")),
    )

    fiv.process_doc_task.fn(duplicate)

    assert store.contains_doc_id(canonical["doc_id"])
    assert not store.contains_doc_id(duplicate["doc_id"])


def test_duplicate_callback_orders_canonical_chunks_naturally(runtime):
    docs_root, store, _ = runtime
    store.get_doc_chunks = MagicMock(
        return_value=[
            SimpleNamespace(
                loc="p:1:c:0",
                snippet="first",
                text="first",
                title="Canonical",
                status="active",
            ),
            SimpleNamespace(
                loc="p:10:c:0", snippet="tenth", text="tenth", title="", status=""
            ),
            SimpleNamespace(
                loc="p:2:c:0", snippet="second", text="second", title="", status=""
            ),
        ]
    )
    duplicate = _make_doc(docs_root, "f/duplicate.pdf", "body", "00002")
    duplicate["ext"] = "pdf"

    event = fiv._build_duplicate_document_indexed_event(
        duplicate, "documents::00001"
    )

    assert [chunk["loc"] for chunk in event["chunks"]] == [
        "p:1:c:0",
        "p:2:c:0",
        "p:10:c:0",
    ]


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


def test_edited_canonical_with_cohort_reclaims_new_identity(runtime):
    """In-place edit of a canonical with live duplicates must not fail open.

    Before #0390 the stranded-cohort rejection was swallowed ("indexing
    normally"), so the registry kept the old hash identity while the index
    held the new content — and any forged claim on the ID clobbered the
    canonical's chunks the same way. The gate now dissolves the stale cohort
    and claims the new identity.
    """
    docs_root, store, registry = runtime
    body = "# Lease\nSame bytes, attached to two emails." * 10
    a = _make_doc(docs_root, "notes/lease.md", body, "00001")
    b = _make_doc(docs_root, "email-attachments/dan/lease-copy.md", body, "00002")
    _register(registry, a)
    _register(registry, b)
    fiv.process_doc_task.fn(a)
    fiv.process_doc_task.fn(b)  # becomes duplicate of 00001

    new_body = "# Lease v2\nCompletely revised terms." * 10
    p = docs_root / "notes/lease.md"
    p.write_text(new_body)
    a2 = {**a, "mtime": p.stat().st_mtime, "size": p.stat().st_size}
    fiv.process_doc_task.fn(a2)

    raw = new_body.encode("utf-8")
    row = registry.find_canonical_by_exact_hash(
        len(raw), blake3.blake3(raw).digest(), "blake3"
    )
    assert row is not None and row["doc_id"] == "00001", (
        "edited canonical must claim its new content identity"
    )
    # The stale cohort is dissolved: the old copy re-elects on its next pass.
    assert registry.duplicate_refs_for_canonical("00001") == []
    assert "documents::00001" in store.list_doc_ids()


def test_edited_canonical_matching_other_canonical_becomes_duplicate(runtime):
    """An edit that makes a cohort's canonical equal another doc's bytes."""
    docs_root, store, registry = runtime
    body_a = "# Notes A\noriginal content of the first cohort." * 10
    body_c = "# Standalone\nentirely different document." * 10
    a = _make_doc(docs_root, "notes/a.md", body_a, "00001")
    b = _make_doc(docs_root, "copies/a-copy.md", body_a, "00002")
    c = _make_doc(docs_root, "notes/c.md", body_c, "00003")
    for doc in (a, b, c):
        _register(registry, doc)
        fiv.process_doc_task.fn(doc)

    # Edit A in place so its bytes now equal C's.
    p = docs_root / "notes/a.md"
    p.write_text(body_c)
    a2 = {**a, "mtime": p.stat().st_mtime, "size": p.stat().st_size}
    fiv.process_doc_task.fn(a2)

    rows = {r["doc_id"]: r for r in map(
        registry._registry_row_to_dict,
        registry._conn.execute(
            "SELECT doc_id, rel_path, created, source_name, size_bytes, content_hash,"
            " hash_algo, dedupe_status, canonical_doc_id, archive_path,"
            " duplicate_reason, duplicate_of_doc_id, first_seen_at, last_seen_at"
            " FROM doc_registry WHERE doc_id IN ('00001', '00003')"
        ).fetchall(),
    )}
    assert rows["00001"]["dedupe_status"] == "duplicate"
    assert rows["00001"]["canonical_doc_id"] == "00003"
    assert rows["00003"]["dedupe_status"] == "canonical"


def test_cohort_whose_members_never_index_resets_at_most_once(runtime):
    """A cohort reset that changes nothing must not repeat every run (#1258).

    Both members hold real content but land no index rows — here because the
    upsert is a no-op, in production because processing fails terminally
    without a skip-ledger entry. Absence alone keeps looking like a lost
    canonical, so every pass reopened the cohort, re-elected the other member
    and found it absent again. The reset is bounded by the state it ran at:
    same members, same bytes, no second reset — and the member whose reset is
    suppressed is still indexed rather than skipped as a duplicate of a
    canonical that holds no rows (#0426).
    """
    docs_root, store, registry = runtime
    body = "content whose members never land index rows"
    a = _make_doc(docs_root, "f/a.md", body, "00001")
    b = _make_doc(docs_root, "g/b.md", body, "00002")
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

    resets = []
    original_reset = fiv._reset_invalid_dedupe_cohort

    def counting_reset(*args, **kwargs):
        affected = original_reset(*args, **kwargs)
        if affected:
            resets.append(list(affected))
        return affected

    indexed_doc_ids = []

    def recording_upsert(nodes):
        indexed_doc_ids.extend({n.metadata["doc_id"] for n in nodes})

    with (
        patch.object(fiv, "_reset_invalid_dedupe_cohort", counting_reset),
        patch.object(store, "upsert_nodes", side_effect=recording_upsert),
    ):
        for _ in range(3):
            fiv.process_doc_task.fn(a)
            fiv.process_doc_task.fn(b)

    assert len(resets) == 1, f"cohort reset repeated: {resets}"
    assert indexed_doc_ids.count("documents::00002") == 3


def _persist_run_skip_ledger(index_root: Path) -> None:
    """Merge the run's skip decisions into the ledger, as index_vault_flow does
    at the end of every run, and reset the per-run accumulators."""
    fiv._save_skip_ledger(
        index_root,
        fiv._merge_skip_ledger(
            fiv._load_skip_ledger(index_root),
            fiv._RUNTIME.get("skip_now", {}),
            fiv._RUNTIME.get("skip_clean", set()),
        ),
    )
    fiv._RUNTIME["skip_now"] = {}
    fiv._RUNTIME["skip_clean"] = set()


def test_skip_only_cohort_converges_after_one_pass(runtime, tmp_path):
    """Two members that both extract no text must settle in a single pass.

    Neither member ever reaches LanceDB — they are intentional
    `no_text_extracted` skips — so "canonical is absent from LanceDB" is not
    evidence that the canonical was lost. Reopening the cohort on it hands
    canonical status back and forth between the two members on every retry
    cycle, forever, while both runs report Completed (#1252).
    """
    docs_root, store, registry = runtime
    index_root = tmp_path / "index"
    fiv._RUNTIME["degraded_lock"] = Lock()
    fiv._RUNTIME["skip_now"] = {}
    fiv._RUNTIME["skip_clean"] = set()
    logger = fiv._get_logger()

    blank = "   \n\n\t\n   "
    a = _make_doc(docs_root, "quo/_indexes/one.md", blank, "001X6")
    b = _make_doc(docs_root, "quo/_indexes/two.md", blank, "001X7")
    _register(registry, a)
    _register(registry, b)

    fiv._process_docs([a, b])
    _persist_run_skip_ledger(index_root)

    # Both are ledgered as skipped, so the second pass is the bounded retry
    # that comes due 24-48h later — it must not re-elect a canonical.
    logger.warning.reset_mock()
    fiv._process_docs([b, a])
    _persist_run_skip_ledger(index_root)

    warnings = [str(call) for call in logger.warning.call_args_list]
    assert not [w for w in warnings if "reopening cohort" in w], warnings
    assert not [w for w in warnings if "dup-metadata update failed" in w], warnings
    assert store.list_doc_ids() == []
    # Settled means "no canonical pointer left to resolve", not "one member
    # holds canonical status forever": the cohort cannot produce an index row,
    # so a duplicate of it would put the content in no row at all (#2097).
    assert registry.duplicate_refs_for_canonical("001X6") == []
    assert _dedupe_statuses(registry) == {
        "001X6": "unindexable",
        "001X7": "unindexable",
    }
    ledger = fiv._load_skip_ledger(index_root)["docs"]
    assert set(ledger) == {"documents::001X6", "documents::001X7"}


def _dedupe_statuses(registry: DocIDStore) -> dict[str, str]:
    return {
        row[0]: row[1]
        for row in registry._conn.execute(
            "SELECT doc_id, dedupe_status FROM doc_registry"
        )
    }


def _seed_ledger(index_root: Path, ns_doc_ids: dict[str, list[str]]) -> None:
    """Write a persisted skip ledger, as an earlier run's merge would leave it."""
    fiv._save_skip_ledger(
        index_root,
        {
            "docs": {
                ns_doc_id: {
                    "reasons": reasons,
                    "change_key": "mtime:1.0",
                    "skipped_at": 1.0,
                }
                for ns_doc_id, reasons in ns_doc_ids.items()
            }
        },
    )


def test_phantom_canonical_cohort_is_repaired_into_an_empty_one(runtime, tmp_path):
    """The production shape of #2097, repaired on the duplicate's next pass.

    `001X6` is canonical and ledgered `no_text_extracted`, so it holds zero
    chunk rows and always will; `001X7` is its duplicate and is therefore
    skipped "against" it. The content is in NO row — not under the canonical
    (skipped) and not under the duplicate (skipped as a duplicate) — and the
    duplicate's callback has no payload to carry. One pass over the duplicate
    must leave no member pointing at a canonical that holds no content.
    """
    docs_root, store, registry = runtime
    index_root = tmp_path / "index"
    fiv._RUNTIME["degraded_lock"] = Lock()
    fiv._RUNTIME["skip_now"] = {}
    fiv._RUNTIME["skip_clean"] = set()

    blank = "   \n\n\t\n   "
    a = _make_doc(docs_root, "quo/_indexes/one.md", blank, "001X6")
    b = _make_doc(docs_root, "quo/_indexes/two.md", blank, "001X7")
    _register(registry, a)
    _register(registry, b)
    raw = Path(a["abs_path"]).read_bytes()
    digest = blake3.blake3(raw).digest()
    registry.claim_canonical_by_exact_hash("001X6", len(raw), digest, hash_algo="blake3")
    registry.claim_canonical_by_exact_hash("001X7", len(raw), digest, hash_algo="blake3")
    _seed_ledger(index_root, {"documents::001X6": ["no_text_extracted"]})
    assert [ref["doc_id"] for ref in registry.duplicate_refs_for_canonical("001X6")] == [
        "001X7"
    ]

    fiv._process_docs([b])

    assert registry.duplicate_refs_for_canonical("001X6") == []
    assert registry.find_canonical_by_exact_hash(len(raw), digest, "blake3") is None
    assert _phantom_canonicals(registry, store) == {}
    assert _dedupe_statuses(registry) == {
        "001X6": "unindexable",
        "001X7": "unindexable",
    }
    assert store.list_doc_ids() == []


def _phantom_canonicals(registry: DocIDStore, store: LanceDBStore) -> dict[str, list[str]]:
    """Duplicate rows whose canonical holds no chunk rows — the #2097 defect.

    The same assertion the production verifier makes against the deployed
    registry and index (`1252-1258-dedupe-cohort-bounded.sh`).
    """
    indexed = set(store.list_doc_ids())
    phantoms: dict[str, list[str]] = {}
    for row in registry._conn.execute(
        "SELECT doc_id, canonical_doc_id, source_name FROM doc_registry "
        "WHERE dedupe_status = 'duplicate' AND canonical_doc_id IS NOT NULL"
    ):
        doc_id, canonical, source_name = row
        namespaced = f"{source_name or 'documents'}::{canonical}"
        if namespaced not in indexed:
            phantoms.setdefault(namespaced, []).append(doc_id)
    return phantoms


def test_terminal_skip_document_is_not_elected_dedupe_canonical(runtime, tmp_path):
    """The election is where a phantom canonical becomes permanent (#2097).

    A document whose recorded verdict is that these bytes produce no index row
    must not win the election: every later copy would be marked a duplicate of
    a canonical that can never resolve. Later copies then form no duplicate
    row at all — the cohort is recorded as intentionally empty instead.
    """
    docs_root, store, registry = runtime
    index_root = tmp_path / "index"
    fiv._RUNTIME["degraded_lock"] = Lock()
    fiv._RUNTIME["skip_now"] = {}
    fiv._RUNTIME["skip_clean"] = set()

    blank = "   \n\n\t\n   "
    a = _make_doc(docs_root, "quo/_indexes/one.md", blank, "002JK")
    b = _make_doc(docs_root, "quo/_indexes/two.md", blank, "002JL")
    c = _make_doc(docs_root, "quo/_indexes/three.md", blank, "002JM")
    for doc in (a, b, c):
        _register(registry, doc)
    _seed_ledger(index_root, {"documents::002JK": ["no_text_extracted"]})

    fiv._process_docs([a])

    raw = Path(a["abs_path"]).read_bytes()
    digest = blake3.blake3(raw).digest()
    assert registry.find_canonical_by_exact_hash(len(raw), digest, "blake3") is None
    assert _dedupe_statuses(registry)["002JK"] == "unindexable"

    # Later copies of the same bytes are never handed to it, and the cohort
    # settles with no canonical pointer for anyone to resolve.
    fiv._process_docs([b, c])

    assert registry.duplicate_refs_for_canonical("002JK") == []
    assert _phantom_canonicals(registry, store) == {}
    assert set(_dedupe_statuses(registry).values()) == {"unindexable"}
    assert store.list_doc_ids() == []


def test_indexable_member_reclaims_canonical_from_an_empty_cohort(runtime, tmp_path):
    """An intentionally empty cohort is a verdict, not a dead end.

    When one member becomes indexable — its bytes changed, OCR came back, so
    its skip entry is gone — it wins the election it initiates and the cohort
    holds real content again.
    """
    docs_root, store, registry = runtime
    index_root = tmp_path / "index"
    fiv._RUNTIME["degraded_lock"] = Lock()
    fiv._RUNTIME["skip_now"] = {}
    fiv._RUNTIME["skip_clean"] = set()

    body = "Recovered attachment text that extracts cleanly."
    doc = _make_doc(docs_root, "quo/_indexes/one.md", body, "002JK")
    _register(registry, doc)
    raw = Path(doc["abs_path"]).read_bytes()
    digest = blake3.blake3(raw).digest()
    registry.mark_exact_hash_cohort_unindexable(
        "002JK", len(raw), digest, hash_algo="blake3", reason="terminal skip: no_text_extracted"
    )

    fiv._process_docs([doc])

    assert store.contains_doc_id("documents::002JK")
    assert _dedupe_statuses(registry)["002JK"] == "canonical"
    assert registry.find_canonical_by_exact_hash(len(raw), digest, "blake3") is not None


def test_duplicate_callback_is_announced_when_canonical_payload_vanishes(
    runtime, monkeypatch
):
    """A duplicate delivery is never dropped on a WARNING (#2097).

    The canonical holds content when the copy is marked its duplicate, so an
    empty payload here is a race. Returning without emitting anything is what
    made the loss silent on both sides: consumers were never told the document
    existed and the outbox had nothing to retry.
    """
    docs_root, store, registry = runtime
    body = "Duplicate attachment body with indexed canonical content."
    canonical = _make_doc(docs_root, "f/canonical.md", body, "00001")
    duplicate = _make_doc(docs_root, "f/duplicate.md", body, "00002")
    _register(registry, canonical)
    _register(registry, duplicate)
    index_root = str(docs_root.parent / "index")
    fiv._RUNTIME["config"].update(
        {
            "index_root": index_root,
            "event_hooks": {
                "enabled": True,
                "hooks": [{"name": "cds", "events": ["document.indexed"]}],
            },
        }
    )
    monkeypatch.setattr(
        "flow_index_vault.drain_due",
        lambda outbox, **kwargs: {
            "accepted": 0,
            "retry_pending": 1,
            "redrive_required": 0,
        },
    )

    fiv.process_doc_task.fn(canonical)
    monkeypatch.setattr(store, "get_doc_chunks", MagicMock(return_value=[]))
    fiv.process_doc_task.fn(duplicate)

    events = [
        delivery.event
        for delivery in HookOutbox(index_root).due(limit=4)
        if delivery.event["doc_id"] == duplicate["doc_id"]
    ]
    assert len(events) == 1, "the duplicate delivery must reach the outbox"
    assert events[0]["rel_path"] == duplicate["rel_path"]
    assert events[0]["metadata"]["canonical_doc_id"] == canonical["doc_id"]
    assert events[0]["metadata"]["canonical_payload_available"] == "false"
    assert events[0]["text"] == ""
