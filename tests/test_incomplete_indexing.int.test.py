# Indexed-but-incomplete document tests (#0584)
#
# A document whose primary-content extractor failed (vision describe timeout,
# transcription failure, page OCR failure) used to land in the index looking
# exactly like a fully-extracted document: the only trace of the failure was a
# WARNING in the flow log. Search consumers could not tell "image described"
# from "image never read", and the enricher happily produced topics from the
# EXIF header ("image, photo, MPO format") which reads like real content.
#
# These tests assert on the record a consumer actually reads back out of
# LanceDB, not on internal flow state.
#
# Run with: pytest tests/test_incomplete_indexing.int.test.py -v

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


_test_logger = logging.getLogger("incomplete-indexing-test")


class MockEmbedProvider(EmbedProvider):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 768 for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [0.1] * 768


class TimingOutVisionProvider(OCRProvider):
    """Vision provider that behaves like the production timeout: describe()
    raises, so no visual content is ever obtained for the image."""

    def __init__(self):
        self.describe_calls = 0

    def extract(self, file_path, page=None) -> str:
        raise TimeoutError("timed out")

    def describe(self, file_path) -> str:
        self.describe_calls += 1
        raise TimeoutError("timed out")


class DescribingVisionProvider(OCRProvider):
    def extract(self, file_path, page=None) -> str:
        return "Cracked bathroom tile above the tub, water staining on the wall."

    def describe(self, file_path) -> str:
        return self.extract(file_path)


class UnstableEnrichmentGenerator:
    """Stand-in for the enrichment LLM that returns different labels on every
    call — the production symptom (same image, different doc_type/topics on all
    7 passes). If enrichment is called for a doc with no extracted content, the
    two runs in these tests disagree and the assertion fails."""

    def __init__(self):
        self.calls = 0

    def generate(self, *args, **kwargs) -> str:
        self.calls += 1
        return (
            '{"summary": "An image file.", "doc_type": ["img", "image", "IMAGE"],'
            f' "topics": ["image", "photo", "MPO format", "variant {self.calls}"],'
            ' "entities_people": [], "entities_places": [], "entities_orgs": [],'
            ' "entities_dates": [], "keywords": [], "key_facts": [],'
            ' "suggested_tags": [], "suggested_folder": "", "importance": 0.5}'
        )


@pytest.fixture(autouse=True)
def _mock_prefect_logger():
    with patch("flow_index_vault.get_run_logger", return_value=_test_logger):
        yield


def _write_jpeg(path: Path) -> Path:
    from PIL import Image

    Image.new("RGB", (640, 480), color=(120, 130, 140)).save(path, format="JPEG")
    return path


def _setup_runtime(index_dir: Path, ocr_provider, llm_generator=None):
    _RUNTIME.clear()
    store = LanceDBStore(str(index_dir), "test_chunks")
    doc_id_store = DocIDStore(index_dir / "doc_registry.db")
    _RUNTIME["store"] = store
    _RUNTIME["doc_id_store"] = doc_id_store
    _RUNTIME["embed_provider"] = MockEmbedProvider()
    _RUNTIME["splitter"] = SentenceSplitter(chunk_size=500, chunk_overlap=50)
    _RUNTIME["ocr_provider"] = ocr_provider
    _RUNTIME["llm_generator"] = llm_generator
    _RUNTIME["taxonomy_store"] = None
    _RUNTIME["semantic_splitter"] = None
    _RUNTIME["semantic_threshold"] = 0
    _RUNTIME["degraded_lock"] = __import__("threading").Lock()
    _RUNTIME["config"] = {"enrichment": {"enabled": bool(llm_generator)}}
    return store, doc_id_store


def _teardown_runtime():
    doc_id_store = _RUNTIME.get("doc_id_store")
    if doc_id_store:
        doc_id_store.close()
    _RUNTIME.clear()


def _image_doc(image_path: Path, ocr_provider, source_name: str = "documents") -> dict:
    """Register an image record processed through the real image extractor."""
    from sources.base import SourceRecord

    doc_id = image_path.stem
    ns_id = f"{source_name}::{doc_id}"
    rec = SourceRecord(
        doc_id=doc_id,
        natural_key=image_path.name,
        source_type="img",
        mtime=image_path.stat().st_mtime,
        size=image_path.stat().st_size,
        metadata={"ext": "jpg", "abs_path": str(image_path)},
    )
    _RUNTIME.setdefault("source_records_by_ns_doc_id", {})[ns_id] = rec

    class _ImageSource:
        name = source_name

        def extract(self, record):
            return extract_text(
                file_path=record.metadata["abs_path"],
                ext=record.metadata["ext"],
                ocr_provider=ocr_provider,
            )

        def scan(self):
            return iter([rec])

        def close(self):
            pass

    _RUNTIME.setdefault("sources_by_name", {})[source_name] = _ImageSource()

    return {
        "doc_id": ns_id,
        "rel_path": rec.natural_key,
        "abs_path": str(image_path),
        "mtime": rec.mtime,
        "size": rec.size,
        "ext": "jpg",
        "source_type": "img",
        "source_name": source_name,
    }


def _index_once(doc: dict) -> None:
    from extractors import begin_degradation_capture
    from flow_index_vault import process_doc_task

    begin_degradation_capture()
    process_doc_task.fn(doc)


def test_failed_image_description_is_consumer_visible_and_stable(tmp_path):
    """A doc whose vision describe timed out must be readable as incomplete
    from the stored record, must not carry LLM-fabricated topics, and must
    produce byte-identical metadata when the unchanged file is indexed again."""
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    image = _write_jpeg(tmp_path / "maintenance-photo.jpg")
    vision = TimingOutVisionProvider()
    enricher = UnstableEnrichmentGenerator()

    store, _ = _setup_runtime(index_dir, vision, llm_generator=enricher)
    try:
        doc = _image_doc(image, vision)
        _index_once(doc)
        first = store.get_doc_chunks(doc["doc_id"])
        assert first, "degraded image doc produced no indexed chunks"
        first_meta = [dict(c.extra_metadata) for c in first]

        # 1. The failure is legible in the record a search consumer reads.
        for chunk in first:
            assert chunk.extra_metadata["content_status"] == "missing"
            assert "ocr_describe_failed" in chunk.extra_metadata[
                "content_failure_reasons"
            ]

        # 2. No fabricated topics from the file's own metadata.
        for chunk in first:
            assert chunk.enr_topics == ""
            assert chunk.enr_doc_type == ""
        assert enricher.calls == 0, "enriched a document with no extracted content"

        # 3. The run summary can count it without tracing individual doc ids.
        assert _RUNTIME["indexed_incomplete"] == {doc["doc_id"]}

        # 4. Re-indexing the unchanged file yields identical enrichment output.
        _index_once(doc)
        second = store.get_doc_chunks(doc["doc_id"])
        assert [dict(c.extra_metadata) for c in second] == first_meta
        assert enricher.calls == 0
    finally:
        _teardown_runtime()


def test_successful_image_description_is_recorded_as_complete(tmp_path):
    """The happy path must stay distinguishable: a described image is complete
    and is still enriched."""
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    image = _write_jpeg(tmp_path / "described-photo.jpg")
    vision = DescribingVisionProvider()
    enricher = UnstableEnrichmentGenerator()

    store, _ = _setup_runtime(index_dir, vision, llm_generator=enricher)
    try:
        doc = _image_doc(image, vision)
        _index_once(doc)
        chunks = store.get_doc_chunks(doc["doc_id"])
        assert chunks
        for chunk in chunks:
            assert chunk.extra_metadata["content_status"] == "complete"
            # Empty extra-metadata values are not stored back out.
            assert chunk.extra_metadata.get("content_failure_reasons", "") == ""
        assert enricher.calls == 1
        assert chunks[0].enr_topics != ""
    finally:
        _teardown_runtime()
