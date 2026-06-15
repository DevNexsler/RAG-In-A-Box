"""Tests for targeted single-document indexing (TICKET-6).

The full flow indexes a whole source; these cover indexing ONE file by
path/doc_id, reusing the same scan/doc_id/diff/extract path, idempotently,
and respecting the deposit-owned (no_rename) ID-alias convention.
"""

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
