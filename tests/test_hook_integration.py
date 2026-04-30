from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.node_parser import SentenceSplitter

from extractors import ExtractionResult
from flow_index_vault import _RUNTIME, process_doc_task
from sources.base import SourceRecord


class _FakeSource:
    name = "documents"

    def __init__(self, text):
        self.text = text

    def extract(self, record):
        return ExtractionResult.from_text(self.text)


def _setup_runtime(store, text="OCR text from image"):
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
            "config": {"event_hooks": {"enabled": True}},
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


def test_process_doc_task_dispatches_document_indexed_after_successful_upsert():
    store = MagicMock()
    _setup_runtime(store)

    with patch("flow_index_vault.get_run_logger", return_value=MagicMock()):
        with patch("flow_index_vault.dispatch_event", create=True, return_value=[]) as dispatch:
            process_doc_task.fn(_doc())

    store.upsert_nodes.assert_called_once()
    dispatch.assert_called_once()
    event_hooks_config, event = dispatch.call_args.args
    assert event_hooks_config == {"enabled": True}
    assert event["event"] == "document.indexed"
    assert event["doc_id"] == "documents::000hF"
    assert event["rel_path"] == "email-attachments/gunther/photo@000hF@.jpg"
    assert event["text"] == "OCR text from image"
    assert event["chunks"][0]["loc"] == "img:c:0"


def test_process_doc_task_records_hook_warnings_without_failing_index():
    store = MagicMock()
    logger = MagicMock()
    _setup_runtime(store)

    with patch("flow_index_vault.get_run_logger", return_value=logger):
        with patch("flow_index_vault.dispatch_event", create=True, return_value=["hook h failed: boom"]):
            process_doc_task.fn(_doc())

    store.upsert_nodes.assert_called_once()
    assert _RUNTIME["_warnings"] == ["hook h failed: boom"]
    logger.warning.assert_called_once_with("hook h failed: boom")


def test_process_doc_task_does_not_dispatch_when_upsert_fails():
    store = MagicMock()
    store.upsert_nodes.side_effect = RuntimeError("upsert failed")
    _setup_runtime(store)

    with patch("flow_index_vault.get_run_logger", return_value=MagicMock()):
        with patch("flow_index_vault.dispatch_event", create=True, return_value=[]) as dispatch:
            with pytest.raises(RuntimeError, match="upsert failed"):
                process_doc_task.fn(_doc())

    dispatch.assert_not_called()
