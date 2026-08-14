from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.node_parser import SentenceSplitter

from extractors import ExtractionResult
from core.hook_outbox import HookOutbox
from flow_index_vault import _RUNTIME, process_doc_task
from sources.base import SourceRecord


class _FakeSource:
    name = "documents"

    def __init__(self, text):
        self.text = text

    def extract(self, record):
        return ExtractionResult.from_text(self.text)


def _setup_runtime(
    store,
    text="OCR text from image",
    *,
    index_root="/tmp/hook-integration-index",
    event_hooks=None,
):
    record = SourceRecord(
        doc_id="000hF",
        source_type="img",
        natural_key="email-attachments/gunther/photo@000hF@.jpg",
        mtime=1.0,
        size=len(text),
        metadata={"abs_path": "/data/documents/email-attachments/gunther/photo@000hF@.jpg", "ext": "jpg"},
    )
    embed_provider = MagicMock()
    embed_provider.embed_texts.return_value = [[0.1, 0.2, 0.3]]
    _RUNTIME.clear()
    _RUNTIME.update(
        {
            "store": store,
            "embed_provider": embed_provider,
            "splitter": SentenceSplitter(chunk_size=512, chunk_overlap=0),
            "semantic_splitter": None,
            "semantic_threshold": 0,
            "ocr_provider": None,
            "config": {
                "index_root": str(index_root),
                "event_hooks": event_hooks if event_hooks is not None else {"enabled": True},
            },
            "sources_by_name": {"documents": _FakeSource(text)},
            "source_records_by_ns_doc_id": {"documents::000hF": record},
        }
    )


def _doc():
    return {
        "doc_id": "documents::000hF",
        "rel_path": "email-attachments/gunther/photo@000hF@.jpg",
        "abs_path": "/data/documents/email-attachments/gunther/photo@000hF@.jpg",
        "mtime": 1.0,
        "size": 128,
        "ext": "jpg",
        "source_type": "img",
        "source_name": "documents",
    }


def test_successful_upsert_keeps_failed_callback_for_later_retry(tmp_path, monkeypatch):
    store = MagicMock()
    _setup_runtime(
        store,
        index_root=tmp_path,
        event_hooks={"enabled": True, "hooks": [{"name": "cds", "events": ["document.indexed"]}]},
    )
    monkeypatch.setattr(
        "flow_index_vault.drain_due",
        lambda outbox, **kwargs: {"accepted": 0, "retry_pending": 1, "redrive_required": 0},
        raising=False,
    )

    with patch("flow_index_vault.get_run_logger", return_value=MagicMock()):
        process_doc_task.fn(_doc())

    store.upsert_nodes.assert_called_once()
    assert HookOutbox(tmp_path).due(limit=1)[0].event["doc_id"] == "documents::000hF"


def test_process_doc_task_records_safe_hook_outcomes_without_failing_index(tmp_path, monkeypatch):
    store = MagicMock()
    logger = MagicMock()
    _setup_runtime(store, index_root=tmp_path)
    monkeypatch.setattr(
        "flow_index_vault.drain_due",
        lambda outbox, **kwargs: {"accepted": 0, "retry_pending": 1, "redrive_required": 0},
        raising=False,
    )

    with patch("flow_index_vault.get_run_logger", return_value=logger):
        process_doc_task.fn(_doc())

    store.upsert_nodes.assert_called_once()
    assert _RUNTIME["_warnings"] == ["hook delivery pending=1 redrive_required=0"]
    logger.warning.assert_called_once_with("hook delivery pending=1 redrive_required=0")


def test_process_doc_task_does_not_queue_when_upsert_fails(tmp_path, monkeypatch):
    store = MagicMock()
    store.upsert_nodes.side_effect = RuntimeError("upsert failed")
    _setup_runtime(
        store,
        index_root=tmp_path,
        event_hooks={"enabled": True, "hooks": [{"name": "cds", "events": ["document.indexed"]}]},
    )
    queue_event = MagicMock()
    monkeypatch.setattr("flow_index_vault.queue_event", queue_event, raising=False)

    with patch("flow_index_vault.get_run_logger", return_value=MagicMock()):
        with pytest.raises(RuntimeError, match="upsert failed"):
            process_doc_task.fn(_doc())

    queue_event.assert_not_called()
