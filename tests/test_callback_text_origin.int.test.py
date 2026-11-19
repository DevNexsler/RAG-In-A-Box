# document.indexed carries text_origin; the stored record does not.
#
# Consumers (Comm-Data-Store) label attachment text by how it was produced: a
# vision model's image description can be wrong and must not read like the
# file's own words. The origin rides on the callback only. Storing it would add
# a metadata sub-field, and a new sub-field widens the whole Lance table
# (_evolve_metadata_schema) on the first sweep after deploy.
#
# Run with: pytest tests/test_callback_text_origin.int.test.py -v

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("prefect")
pytest.importorskip("llama_index")
pytest.importorskip("PIL")

from llama_index.core.node_parser import SentenceSplitter

from doc_id_store import DocIDStore
from extractors import extract_text
from flow_index_vault import _RUNTIME
from lancedb_store import LanceDBStore
from providers.embed.base import EmbedProvider
from providers.ocr.base import OCRProvider

_test_logger = logging.getLogger("callback-text-origin-test")


class MockEmbedProvider(EmbedProvider):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 768 for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [0.1] * 768


class DescribingVision(OCRProvider):
    def extract(self, file_path, page=None) -> str:
        return "Scanned notice: rent increase effective March 1."

    def describe(self, file_path) -> str:
        return "A water-stained ceiling above a kitchen sink."


@pytest.fixture(autouse=True)
def _mock_prefect_logger():
    with patch("flow_index_vault.get_run_logger", return_value=_test_logger):
        yield


@pytest.fixture
def runtime(tmp_path):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    _RUNTIME.clear()
    store = LanceDBStore(str(index_dir), "test_chunks")
    doc_id_store = DocIDStore(index_dir / "doc_registry.db")
    _RUNTIME.update({
        "store": store,
        "doc_id_store": doc_id_store,
        "embed_provider": MockEmbedProvider(),
        "splitter": SentenceSplitter(chunk_size=500, chunk_overlap=50),
        "ocr_provider": DescribingVision(),
        "llm_generator": None,
        "taxonomy_store": None,
        "semantic_splitter": None,
        "semantic_threshold": 0,
        "degraded_lock": __import__("threading").Lock(),
        "config": {"enrichment": {"enabled": False}},
    })
    try:
        yield store
    finally:
        doc_id_store.close()
        _RUNTIME.clear()


def _register(path: Path, ext: str, source_type: str, source_name: str = "documents") -> dict:
    from sources.base import SourceRecord

    ns_id = f"{source_name}::{path.stem}"
    rec = SourceRecord(
        doc_id=path.stem, natural_key=path.name, source_type=source_type,
        mtime=path.stat().st_mtime, size=path.stat().st_size,
        metadata={"ext": ext, "abs_path": str(path)},
    )
    _RUNTIME.setdefault("source_records_by_ns_doc_id", {})[ns_id] = rec

    class _Source:
        name = source_name

        def extract(self, record):
            return extract_text(
                file_path=record.metadata["abs_path"], ext=record.metadata["ext"],
                ocr_provider=_RUNTIME["ocr_provider"],
            )

        def scan(self):
            return iter([rec])

        def close(self):
            pass

    _RUNTIME.setdefault("sources_by_name", {})[source_name] = _Source()
    return {
        "doc_id": ns_id, "rel_path": rec.natural_key, "abs_path": str(path),
        "mtime": rec.mtime, "size": rec.size, "ext": ext,
        "source_type": source_type, "source_name": source_name,
    }


def _index(doc: dict) -> list[dict]:
    from extractors import begin_degradation_capture
    from flow_index_vault import process_doc_task

    events: list[dict] = []
    begin_degradation_capture()
    with patch(
        "flow_index_vault._dispatch_document_indexed_event",
        side_effect=lambda config, event, logger: events.append(event),
    ):
        process_doc_task.fn(doc)
    return events


def test_described_image_callback_says_vision_and_store_is_unchanged(runtime, tmp_path):
    from PIL import Image

    image = tmp_path / "ceiling.jpg"
    Image.new("RGB", (64, 48), color=(120, 130, 140)).save(image, format="JPEG")
    doc = _register(image, "jpg", "img")

    (event,) = _index(doc)
    assert event["event"] == "document.indexed"
    assert event["metadata"]["text_origin"] == "vision"
    for chunk in runtime.get_doc_chunks(doc["doc_id"]):
        assert "text_origin" not in chunk.extra_metadata


def test_scanned_pdf_callback_says_ocr(runtime, tmp_path):
    import pymupdf

    pdf = tmp_path / "notice.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(str(pdf))
    document.close()
    doc = _register(pdf, "pdf", "pdf")

    (event,) = _index(doc)
    assert "rent increase" in event["text"]
    assert event["metadata"]["text_origin"] == "ocr"


def test_markdown_callback_says_text_layer(runtime, tmp_path):
    note = tmp_path / "repair-log.md"
    note.write_text("# Repair log\n\nBoiler serviced on Tuesday.")
    doc = _register(note, "md", "md")

    (event,) = _index(doc)
    assert event["metadata"]["text_origin"] == "text_layer"
