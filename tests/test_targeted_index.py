"""Tests for targeted single-document indexing (TICKET-6).

The full flow indexes a whole source; these cover indexing ONE file by
path/doc_id, reusing the same scan/doc_id/diff/extract path, idempotently,
and respecting the deposit-owned (no_rename) ID-alias convention.
"""

import sqlite3
import threading
from unittest.mock import MagicMock, patch

import pytest

import flow_index_vault as fiv
from doc_id_store import DocIDStore


def _fs_config(root, index_root):
    return {
        "index_root": str(index_root),
        "lancedb": {"table": "chunks"},
        "pdf": {"strategy": "text_then_ocr"},
        "sources": [
            {
                "type": "filesystem",
                "name": "documents",
                "root": str(root),
                "scan": {
                    "include": ["**/*.pdf", "**/*.png", "**/*.txt"],
                    "exclude": [],
                    "no_rename": ["email-attachments/", "quo-attachments/"],
                },
            }
        ],
    }


def _make_attachment(tmp_path, rel, data=b"%PDF-1.4 hello"):
    root = tmp_path / "docs"
    f = root / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(data)
    return root, f


# --------------------------------------------------------------------------
# resolve_single_record
# --------------------------------------------------------------------------

def test_resolve_single_record_preserves_alias_id_and_namespaces(tmp_path):
    root, f = _make_attachment(tmp_path, "email-attachments/laura/inspection@00abc@.pdf")
    store = DocIDStore(tmp_path / "reg.db")
    cfg = _fs_config(root, tmp_path / "index")

    rec = fiv.resolve_single_record(cfg, "documents", str(f), store)

    assert rec["doc_id"] == "documents::00abc"
    assert rec["rel_path"] == "email-attachments/laura/inspection@00abc@.pdf"
    assert rec["source_name"] == "documents"
    assert rec["ext"] == "pdf"
    assert f.exists()  # alias-tagged deposit file is never renamed


def test_resolve_single_record_deposit_path_assigns_id_without_rename(tmp_path):
    root, f = _make_attachment(tmp_path, "email-attachments/photo.png", data=b"\x89PNG hi")
    store = DocIDStore(tmp_path / "reg.db")
    cfg = _fs_config(root, tmp_path / "index")

    rec = fiv.resolve_single_record(cfg, "documents", "email-attachments/photo.png", store)

    assert rec["rel_path"] == "email-attachments/photo.png"  # NOT renamed (deposit-owned)
    assert f.exists()
    assert rec["doc_id"].startswith("documents::")

    # Idempotent: resolving the same deposit path returns the same doc_id.
    rec2 = fiv.resolve_single_record(cfg, "documents", "email-attachments/photo.png", store)
    assert rec2["doc_id"] == rec["doc_id"]


def test_resolve_single_record_accepts_doc_id_target(tmp_path):
    root, f = _make_attachment(tmp_path, "email-attachments/x@00ZZz@.pdf")
    store = DocIDStore(tmp_path / "reg.db")
    cfg = _fs_config(root, tmp_path / "index")
    fiv.resolve_single_record(cfg, "documents", str(f), store)  # register

    rec = fiv.resolve_single_record(cfg, "documents", "documents::00ZZz", store)
    assert rec["doc_id"] == "documents::00ZZz"
    assert rec["rel_path"] == "email-attachments/x@00ZZz@.pdf"


def test_resolve_single_record_finds_alias_injected_file_from_clean_path(tmp_path):
    """A prior full scan may have renamed the deposit to inject @id@ (when the
    source has no `no_rename`), but the caller (comm-data-store) only knows the
    clean pre-rename path. resolve must still find the aliased file on disk."""
    root, f = _make_attachment(tmp_path, "email-attachments/laura/report@001Hk@.pdf")
    store = DocIDStore(tmp_path / "reg.db")
    cfg = _fs_config(root, tmp_path / "index")

    rec = fiv.resolve_single_record(cfg, "documents", "email-attachments/laura/report.pdf", store)

    assert rec is not None
    assert rec["rel_path"] == "email-attachments/laura/report@001Hk@.pdf"
    assert rec["doc_id"] == "documents::001Hk"


def test_resolve_single_record_missing_file_returns_none(tmp_path):
    root, _ = _make_attachment(tmp_path, "email-attachments/real.pdf")
    store = DocIDStore(tmp_path / "reg.db")
    cfg = _fs_config(root, tmp_path / "index")

    assert fiv.resolve_single_record(cfg, "documents", "email-attachments/nope.pdf", store) is None


def test_resolve_single_record_unknown_source_raises(tmp_path):
    root, _ = _make_attachment(tmp_path, "email-attachments/real.pdf")
    store = DocIDStore(tmp_path / "reg.db")
    cfg = _fs_config(root, tmp_path / "index")

    with pytest.raises(ValueError):
        fiv.resolve_single_record(cfg, "nope", "email-attachments/real.pdf", store)


# --------------------------------------------------------------------------
# index_document_flow
# --------------------------------------------------------------------------

def test_index_document_flow_skips_unchanged_without_processing(tmp_path):
    root, f = _make_attachment(tmp_path, "email-attachments/a@00xyz@.txt", data=b"hello text")
    cfg = _fs_config(root, tmp_path / "index")
    store = MagicMock()
    store.list_doc_mtimes.return_value = {"documents::00xyz": f.stat().st_mtime}
    store.list_doc_change_hashes.return_value = {}

    with patch("flow_index_vault.load_config", return_value=cfg), \
         patch("flow_index_vault.open_store_with_recovery", return_value=store), \
         patch("flow_index_vault.process_doc_task") as proc:
        result = fiv.index_document_flow(target=str(f), source_name="documents")

    assert result["status"] == "skipped"
    assert result["reason"] == "unchanged"
    assert result["doc_id"] == "documents::00xyz"
    proc.fn.assert_not_called()


def test_index_document_flow_indexes_new_file(tmp_path):
    root, f = _make_attachment(tmp_path, "email-attachments/a@00xyz@.txt", data=b"hello text")
    cfg = _fs_config(root, tmp_path / "index")
    store = MagicMock()
    store.list_doc_mtimes.return_value = {}
    store.list_doc_change_hashes.return_value = {}

    with patch("flow_index_vault.load_config", return_value=cfg), \
         patch("flow_index_vault.open_store_with_recovery", return_value=store), \
         patch("flow_index_vault.build_embed_provider", return_value=MagicMock()), \
         patch("flow_index_vault.build_ocr_provider", return_value=None), \
         patch("flow_index_vault.build_media_provider", return_value=None), \
         patch("flow_index_vault.process_doc_task") as proc:
        result = fiv.index_document_flow(target=str(f), source_name="documents")

    assert result["status"] == "indexed"
    assert result["doc_id"] == "documents::00xyz"
    assert result["rel_path"] == "email-attachments/a@00xyz@.txt"
    proc.fn.assert_called_once()
    called_doc = proc.fn.call_args[0][0]
    assert called_doc["doc_id"] == "documents::00xyz"
    assert called_doc["source_name"] == "documents"


def test_index_document_flow_force_reindexes_unchanged(tmp_path):
    root, f = _make_attachment(tmp_path, "email-attachments/a@00xyz@.txt", data=b"hello text")
    cfg = _fs_config(root, tmp_path / "index")
    store = MagicMock()
    store.list_doc_mtimes.return_value = {"documents::00xyz": f.stat().st_mtime}
    store.list_doc_change_hashes.return_value = {}

    with patch("flow_index_vault.load_config", return_value=cfg), \
         patch("flow_index_vault.open_store_with_recovery", return_value=store), \
         patch("flow_index_vault.build_embed_provider", return_value=MagicMock()), \
         patch("flow_index_vault.build_ocr_provider", return_value=None), \
         patch("flow_index_vault.build_media_provider", return_value=None), \
         patch("flow_index_vault.process_doc_task") as proc:
        result = fiv.index_document_flow(target=str(f), source_name="documents", force=True)

    assert result["status"] == "indexed"
    proc.fn.assert_called_once()


def test_index_document_flow_runs_real_process_doc_task_without_prefect_context(tmp_path):
    """Regression: standalone indexing must not require an active Prefect flow/
    task run context. process_doc_task used get_run_logger() directly, which
    raises 'no active flow or task run context' when called outside a flow."""
    root, f = _make_attachment(
        tmp_path, "email-attachments/note@00txt@.txt", data=b"hello searchable text body"
    )
    cfg = _fs_config(root, tmp_path / "index")
    store = MagicMock()
    store.list_doc_mtimes.return_value = {}
    store.list_doc_change_hashes.return_value = {}
    embed = MagicMock()
    embed.embed_texts.side_effect = lambda texts: [[0.1, 0.2, 0.3] for _ in texts]

    with patch("flow_index_vault.load_config", return_value=cfg), \
         patch("flow_index_vault.open_store_with_recovery", return_value=store), \
         patch("flow_index_vault.build_embed_provider", return_value=embed), \
         patch("flow_index_vault.build_ocr_provider", return_value=None), \
         patch("flow_index_vault.build_media_provider", return_value=None), \
         patch("flow_index_vault.dispatch_event", create=True, return_value=[]) as dispatch:
        result = fiv.index_document_flow(target=str(f), source_name="documents")

    assert result["status"] == "indexed"
    store.upsert_nodes.assert_called_once()  # embedded chunks upserted
    dispatch.assert_called_once()  # document.indexed webhook emitted


def test_mcp_file_index_document_impl_wraps_flow():
    import mcp_server

    with patch(
        "flow_index_vault.index_document_flow",
        return_value={"status": "indexed", "doc_id": "documents::00abc"},
    ) as flow:
        result = mcp_server._file_index_document_impl(
            target="email-attachments/x@00abc@.pdf", source_name="documents"
        )

    assert result["status"] == "indexed"
    assert result["doc_id"] == "documents::00abc"
    flow.assert_called_once()
    _, kwargs = flow.call_args
    assert kwargs["target"] == "email-attachments/x@00abc@.pdf"
    assert kwargs["source_name"] == "documents"


def test_mcp_file_index_document_impl_requires_target():
    import mcp_server

    result = mcp_server._file_index_document_impl(target="")
    assert result.get("error") is True


def test_mcp_file_index_document_impl_surfaces_flow_exception():
    import mcp_server

    with patch("flow_index_vault.index_document_flow", side_effect=RuntimeError("boom")):
        result = mcp_server._file_index_document_impl(target="email-attachments/x.pdf")

    assert result.get("error") is True
    assert result["code"] == "index_failed"


# --------------------------------------------------------------------------
# Single-doc ledger bookkeeping (#0264)
#
# The targeted path (REST per-attachment, MCP file_index_document) runs the
# same process_doc_task as the full flow, but historically never began a
# degradation capture nor merged outcomes into the ledgers — every
# note_degradation()/note_skip() from extraction was silently dropped. During
# the 2026-07 vision outage that indexed 28 image attachments as metadata-only
# stubs with NO degraded-ledger entry: no retry, no backfill, permanent
# content loss that looked like a successful index.
# --------------------------------------------------------------------------


class _EmptyDescribeOCR:
    """Outage-shaped provider: describe() exhausts retries and returns ''."""

    def extract(self, file_path, page=None):
        return ""

    def describe(self, file_path):
        return ""


def _make_png(tmp_path, rel):
    from PIL import Image

    root = tmp_path / "docs"
    f = root / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (600, 800), "white").save(f)
    return root, f


def _run_single_doc(tmp_path, root, f, ocr_provider):
    cfg = _fs_config(root, tmp_path / "index")
    store = MagicMock()
    store.list_doc_mtimes.return_value = {}
    store.list_doc_change_hashes.return_value = {}
    embed = MagicMock()
    embed.embed_texts.side_effect = lambda texts: [[0.1, 0.2, 0.3] for _ in texts]

    with patch("flow_index_vault.load_config", return_value=cfg), \
         patch("flow_index_vault.open_store_with_recovery", return_value=store), \
         patch("flow_index_vault.build_embed_provider", return_value=embed), \
         patch("flow_index_vault.build_ocr_provider", return_value=ocr_provider), \
         patch("flow_index_vault.build_media_provider", return_value=None), \
         patch("flow_index_vault.dispatch_event", create=True, return_value=[]):
        result = fiv.index_document_flow(target=str(f), source_name="documents")
    return result, store


def _ledger(tmp_path, name):
    import json

    path = tmp_path / "index" / name
    if not path.exists():
        return {"docs": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def test_index_document_flow_stub_indexed_image_lands_in_degraded_ledger(tmp_path):
    """An image whose describe() came back empty during an outage still indexes
    (metadata-only stub is better than nothing) but MUST land in degraded_docs.json
    so the retry machinery re-describes it after provider recovery.

    Post-fallback architecture: production ALWAYS wraps the describe provider
    (build_ocr_provider), so this test injects the wrapped shape. With no reachable
    fallback (dark), the wrapper raises transient for the unconfirmed empty — the
    stub lands as a *transient* ocr_describe_failed: retried every run and, per
    #0251, never burning the attempt cap."""
    from providers.ocr.fallback import FallbackOCRProvider

    root, f = _make_png(tmp_path, "quo-attachments/photo@00img@.png")

    # Match production: the describe provider is always wrapped. No fallback endpoint
    # (dark) => an unconfirmed empty is a transient degradation, not a silent clean stub.
    wrapped = FallbackOCRProvider(_EmptyDescribeOCR(), describe_fallback=None)
    result, store = _run_single_doc(tmp_path, root, f, wrapped)

    assert result["status"] == "indexed"
    store.upsert_nodes.assert_called_once()  # the stub WAS indexed…
    entry = _ledger(tmp_path, "degraded_docs.json")["docs"].get("documents::00img")
    assert entry is not None, (
        "stub-indexed doc has no degraded-ledger entry — nothing will retry it"
    )
    assert "ocr_describe_failed" in entry["reasons"]
    assert entry.get("attempts", 0) == 0  # transient outage must not burn the cap (#0251)


def test_index_document_flow_describe_failure_lands_in_degraded_ledger(tmp_path):
    """Outage shape #2: describe() raising (host down) must also be persisted
    by the targeted path, not just noted into a dropped thread-local."""

    class BoomOCR:
        def extract(self, file_path, page=None):
            return ""

        def describe(self, file_path):
            raise ConnectionError("vision host down")

    root, f = _make_png(tmp_path, "quo-attachments/photo@00imh@.png")

    result, _ = _run_single_doc(tmp_path, root, f, BoomOCR())

    assert result["status"] == "indexed"
    entry = _ledger(tmp_path, "degraded_docs.json")["docs"].get("documents::00imh")
    assert entry is not None
    assert "ocr_describe_failed" in entry["reasons"]


def test_index_document_flow_clean_run_clears_ledger_entries(tmp_path):
    """Self-heal parity with the full flow: a clean targeted re-index drops the
    doc from both ledgers."""
    import json

    root, f = _make_png(tmp_path, "quo-attachments/photo@00imi@.png")
    index_root = tmp_path / "index"
    index_root.mkdir(parents=True)
    (index_root / "degraded_docs.json").write_text(json.dumps(
        {"docs": {"documents::00imi": {"reasons": ["ocr_describe_empty"], "attempts": 1}}}
    ))

    class GoodOCR:
        def extract(self, file_path, page=None):
            return ""

        def describe(self, file_path):
            return "A photo of a leaking kitchen faucet, water pooling under the sink."

    result, _ = _run_single_doc(tmp_path, root, f, GoodOCR())

    assert result["status"] == "indexed"
    assert "documents::00imi" not in _ledger(tmp_path, "degraded_docs.json")["docs"]


def test_index_document_flow_skip_outcome_lands_in_skip_ledger(tmp_path):
    """Permanent skip decisions (corrupt source) on the targeted path must land
    in the skip ledger, same as the full flow."""
    root, f = _make_attachment(
        tmp_path, "email-attachments/broken@00bad@.pdf", data=b"not a real pdf"
    )

    result, _ = _run_single_doc(tmp_path, root, f, None)

    entry = _ledger(tmp_path, "skip_docs.json")["docs"].get("documents::00bad")
    assert entry is not None
    assert "pdf_unreadable" in entry["reasons"]


def test_index_document_flow_missing_file_returns_error(tmp_path):
    root, _ = _make_attachment(tmp_path, "email-attachments/real.pdf")
    cfg = _fs_config(root, tmp_path / "index")
    store = MagicMock()

    with patch("flow_index_vault.load_config", return_value=cfg), \
         patch("flow_index_vault.open_store_with_recovery", return_value=store), \
         patch("flow_index_vault.process_doc_task") as proc:
        result = fiv.index_document_flow(target="email-attachments/nope.pdf", source_name="documents")

    assert result["status"] == "error"
    assert result["reason"] == "not_found"
    proc.fn.assert_not_called()


def test_index_document_flow_durably_queues_during_full_writer_session(
    tmp_path, monkeypatch
):
    from core.index_request_queue import IndexRequestQueue
    from core.index_write_lock import index_write_lock

    config = _fs_config(tmp_path / "docs", tmp_path / "index")
    monkeypatch.setattr(fiv, "load_config", lambda _path: config)
    acquired = threading.Event()
    release = threading.Event()

    def hold_full_writer():
        with index_write_lock(config["index_root"], "chunks"):
            acquired.set()
            release.wait(10)

    holder = threading.Thread(target=hold_full_writer)
    holder.start()
    try:
        assert acquired.wait(5)
        result = fiv.index_document_flow(target="queued.pdf")
    finally:
        release.set()
        holder.join(5)

    assert result == {
        "status": "queued",
        "reason": "index_write_in_progress",
        "source_name": "documents",
        "target": "queued.pdf",
        "revision": 1,
    }
    [queued] = IndexRequestQueue(config["index_root"]).pending(
        "chunks", limit=10
    )
    assert queued.target == "queued.pdf"


def test_index_document_flow_enqueue_failure_never_attempts_table_lock(
    tmp_path, monkeypatch
):
    config = _fs_config(tmp_path / "docs", tmp_path / "index")
    monkeypatch.setattr(fiv, "load_config", lambda _path: config)

    with patch(
        "flow_index_vault.IndexRequestQueue.enqueue",
        side_effect=sqlite3.OperationalError("disk full"),
    ), patch("flow_index_vault.index_write_lock") as lock:
        with pytest.raises(sqlite3.OperationalError, match="disk full"):
            fiv.index_document_flow(target="queued.pdf")

    lock.assert_not_called()


def test_drain_prioritizes_current_request_is_bounded_and_retains_failures(
    tmp_path,
):
    from core.index_request_queue import IndexRequestQueue

    queue = IndexRequestQueue(tmp_path)
    failed = queue.enqueue("chunks", "documents", "a.pdf")
    queue.enqueue("chunks", "documents", "b.pdf")
    current = queue.enqueue("chunks", "mail", "c.pdf")
    store = object()
    registry = object()
    seen = []

    def process(config, request, passed_store, passed_registry):
        assert passed_store is store
        assert passed_registry is registry
        seen.append(request.target)
        if request.id == failed.id:
            raise RuntimeError("provider offline")
        return {"status": "indexed", "target": request.target}

    with patch("flow_index_vault._index_document_unlocked", side_effect=process):
        results = fiv._drain_index_requests(
            {},
            queue,
            "chunks",
            store,
            registry,
            limit=2,
            prioritize=(current.source_name, current.target),
        )

    assert seen == ["c.pdf", "a.pdf"]
    assert results[("mail", "c.pdf")]["status"] == "indexed"
    assert results[("documents", "a.pdf")]["status"] == "queued"
    pending = queue.pending("chunks", limit=10)
    assert [(item.target, item.attempts) for item in pending] == [
        ("a.pdf", 1),
        ("b.pdf", 0),
    ]


def test_drain_keeps_force_revision_enqueued_during_older_processing(tmp_path):
    from core.index_request_queue import IndexRequestQueue

    queue = IndexRequestQueue(tmp_path)
    old = queue.enqueue("chunks", "documents", "same.pdf")

    def process(_config, request, _store, _registry):
        queue.enqueue(
            request.table_name,
            request.source_name,
            request.target,
            force=True,
        )
        return {"status": "indexed"}

    with patch("flow_index_vault._index_document_unlocked", side_effect=process):
        fiv._drain_index_requests(
            {}, queue, "chunks", object(), object(), limit=1
        )

    [pending] = queue.pending("chunks", limit=10)
    assert pending.id == old.id
    assert pending.revision == old.revision + 1
    assert pending.force is True


def test_drain_completes_terminal_not_found_request(tmp_path):
    from core.index_request_queue import IndexRequestQueue

    queue = IndexRequestQueue(tmp_path)
    queue.enqueue("chunks", "documents", "gone.pdf")
    with patch(
        "flow_index_vault._index_document_unlocked",
        return_value={"status": "error", "reason": "not_found"},
    ):
        fiv._drain_index_requests(
            {}, queue, "chunks", object(), object(), limit=1
        )

    assert queue.pending("chunks", limit=10) == []


def test_index_document_flow_confirmed_blank_clears_ledger_entry(tmp_path):
    """A fallback-CONFIRMED blank (primary empty + fallback ALSO empty) is clean:
    it indexes the metadata stub and DROPS any prior degraded-ledger entry (parity
    with a recovered/clean run via the clean_now path), so a genuinely blank image
    stops being retried instead of churning forever."""
    import json
    from providers.ocr.fallback import FallbackOCRProvider

    root, f = _make_png(tmp_path, "quo-attachments/photo@00imk@.png")
    index_root = tmp_path / "index"
    index_root.mkdir(parents=True)
    (index_root / "degraded_docs.json").write_text(json.dumps(
        {"docs": {"documents::00imk": {"reasons": ["ocr_describe_failed"], "attempts": 0}}}
    ))

    # primary empty + fallback returns "" -> two independent models agree: blank.
    wrapped = FallbackOCRProvider(_EmptyDescribeOCR(), describe_fallback=lambda p: "")
    result, store = _run_single_doc(tmp_path, root, f, wrapped)

    assert result["status"] == "indexed"
    store.upsert_nodes.assert_called_once()  # stub still indexed
    assert "documents::00imk" not in _ledger(tmp_path, "degraded_docs.json")["docs"]
