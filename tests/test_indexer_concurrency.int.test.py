# Concurrent document processing tests
#
# Verifies that parallelizing the per-doc loop in index_vault_flow does not
# break error isolation, upsert idempotency, FTS deferred rebuild, or the
# serial baseline (concurrency=1). These are regression guards for adding
# ThreadPoolExecutor concurrency without introducing races on shared state.
#
# Run with: pytest tests/test_indexer_concurrency.int.test.py -v

import logging
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("prefect")
pytest.importorskip("llama_index")

from llama_index.core.node_parser import SentenceSplitter

from doc_id_store import DocIDStore
from flow_index_vault import _RUNTIME, process_doc_task
from lancedb_store import LanceDBStore
from providers.embed.base import EmbedProvider


_test_logger = logging.getLogger("concurrency-test")


class MockEmbedProvider(EmbedProvider):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 768 for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [0.1] * 768


class FailOnDocEmbedProvider(EmbedProvider):
    """Embed provider that raises RuntimeError for docs whose first chunk
    contains a known poison token. Lets us exercise error isolation."""

    def __init__(self, poison: str = "POISON"):
        self.poison = poison

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        for t in texts:
            if self.poison in t:
                raise RuntimeError(f"embed failed: poison in chunk")
        return [[0.1] * 768 for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [0.1] * 768


@pytest.fixture(autouse=True)
def _mock_prefect_logger():
    with patch("flow_index_vault.get_run_logger", return_value=_test_logger):
        yield


@pytest.fixture
def index_dir(tmp_path):
    d = tmp_path / "index"
    d.mkdir()
    return d


def _setup_runtime(index_dir: Path, embed=None):
    _RUNTIME.clear()
    store = LanceDBStore(str(index_dir), "test_chunks")
    doc_id_store = DocIDStore(index_dir / "doc_registry.db")
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
    _RUNTIME["store"] = store
    _RUNTIME["doc_id_store"] = doc_id_store
    _RUNTIME["embed_provider"] = embed or MockEmbedProvider()
    _RUNTIME["splitter"] = splitter
    _RUNTIME["ocr_provider"] = None
    _RUNTIME["llm_generator"] = None
    _RUNTIME["taxonomy_store"] = None
    _RUNTIME["semantic_splitter"] = None
    _RUNTIME["semantic_threshold"] = 0
    _RUNTIME["config"] = {}
    return store, doc_id_store


def _teardown_runtime():
    doc_id_store = _RUNTIME.get("doc_id_store")
    if doc_id_store:
        doc_id_store.close()
    _RUNTIME.clear()


def _fake_record(doc_id: str, text: str, source_name: str = "documents") -> dict:
    """Build a doc dict compatible with process_doc_task, bypassing filesystem."""
    from sources.base import SourceRecord
    from extractors import ExtractionResult

    ns_id = f"{source_name}::{doc_id}"
    rec = SourceRecord(
        doc_id=doc_id,
        natural_key=f"{doc_id}.txt",
        source_type="txt",
        mtime=1.0,
        size=len(text),
        metadata={"ext": "txt", "abs_path": f"/fake/{doc_id}.txt", "text": text},
    )
    _RUNTIME.setdefault("source_records_by_ns_doc_id", {})[ns_id] = rec

    # Minimal Source that returns the cached text via extract()
    class _FakeSource:
        name = source_name
        def extract(self, record):
            return ExtractionResult.from_text(record.metadata["text"])
        def scan(self):
            return iter([rec])
        def close(self):
            pass
    _RUNTIME.setdefault("sources_by_name", {})[source_name] = _FakeSource()

    return {
        "doc_id": ns_id,
        "rel_path": rec.natural_key,
        "abs_path": rec.metadata["abs_path"],
        "mtime": rec.mtime,
        "size": rec.size,
        "ext": "txt",
        "source_type": "txt",
        "source_name": source_name,
    }


# ---------------------------------------------------------------------------
# 1. Serial baseline: concurrency=1 must be byte-for-byte identical to the
#    pre-refactor for-loop. This is the rollback safety net.
# ---------------------------------------------------------------------------


class TestSerialBaseline:
    def test_concurrency_1_preserves_order_and_result(self, index_dir):
        from flow_index_vault import _process_docs

        store, _ = _setup_runtime(index_dir)
        try:
            docs = [_fake_record(f"id{i:02d}", f"body text for doc {i}") for i in range(5)]
            failed = _process_docs(docs, concurrency=1)
            assert failed == []
            ids = sorted(store.list_doc_ids())
            assert ids == sorted(d["doc_id"] for d in docs)
        finally:
            _teardown_runtime()


# ---------------------------------------------------------------------------
# 2. Error isolation: one bad doc must not sink the others; failed_docs
#    contains only the failing id; other docs produce chunks.
# ---------------------------------------------------------------------------


class TestErrorIsolation:
    @pytest.mark.parametrize("concurrency", [1, 4])
    def test_one_doc_fails_others_succeed(self, index_dir, concurrency):
        from flow_index_vault import _process_docs

        store, _ = _setup_runtime(index_dir, embed=FailOnDocEmbedProvider("POISON"))
        try:
            docs = [
                _fake_record("idok1", "clean text one"),
                _fake_record("idbad", "POISON body triggers embed failure"),
                _fake_record("idok2", "clean text two"),
                _fake_record("idok3", "clean text three"),
            ]
            failed = _process_docs(docs, concurrency=concurrency)
            assert failed == ["documents::idbad"]
            indexed = set(store.list_doc_ids())
            assert "documents::idbad" not in indexed
            assert {"documents::idok1", "documents::idok2", "documents::idok3"} <= indexed
        finally:
            _teardown_runtime()


# ---------------------------------------------------------------------------
# 3. Concurrent upsert on the same doc_id: must not explode the chunk count
#    or leave the table in a split state. Last writer wins; chunks for the
#    doc form one coherent set.
# ---------------------------------------------------------------------------


class TestConcurrentUpsertSameDocID:
    def test_two_threads_same_doc_id_one_chunk_set(self, index_dir):
        from flow_index_vault import _process_docs

        store, _ = _setup_runtime(index_dir)
        try:
            # Same doc_id, different text bodies
            d_v1 = _fake_record("iddup", "body version ONE")
            d_v2 = _fake_record("iddup", "body version TWO")
            # Call twice via the parallel path
            _process_docs([d_v1, d_v2], concurrency=2)

            # Exactly one doc_id present, and its chunks match EITHER v1 or v2 body
            indexed = set(store.list_doc_ids())
            assert indexed == {"documents::iddup"}
            chunks = store.get_doc_chunks("documents::iddup")
            assert len(chunks) >= 1
            # All chunks must belong to a single version (no half-and-half)
            texts = {c.text.split("\n\n", 1)[-1] if "\n\n" in c.text else c.text for c in chunks}
            # We tolerate either version; we just forbid a mixed set.
            assert len(texts) == 1, f"Mixed versions indicate torn upsert: {texts}"
        finally:
            _teardown_runtime()


# ---------------------------------------------------------------------------
# 4. DocIDStore concurrent register: 20 threads registering distinct IDs
#    must all persist. SQLITE_BUSY must NOT propagate as an exception.
# ---------------------------------------------------------------------------


class TestDocIDStoreConcurrentRegister:
    def test_20_threads_register_distinct_ids(self, index_dir):
        store = DocIDStore(index_dir / "doc_registry.db")
        try:
            errors: list[Exception] = []
            def _worker(n: int):
                try:
                    store.register(f"did{n:03d}", f"path/{n}.txt", source_name="documents")
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=_worker, args=(i,)) for i in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert errors == [], f"Concurrent register raised: {errors}"
            # All 20 should resolve
            for i in range(20):
                assert store.lookup_path(f"did{i:03d}") == f"path/{i}.txt"
        finally:
            store.close()


# ---------------------------------------------------------------------------
# 5. FTS rebuild is deferred to end-of-flow, not per-doc. Concurrency must
#    not invoke create_fts_index() N times — once at flow finalize only.
#    This is implicit in _process_docs not touching FTS.
# ---------------------------------------------------------------------------


class TestFTSRebuildDeferred:
    def test_process_docs_does_not_rebuild_fts(self, index_dir):
        from flow_index_vault import _process_docs

        store, _ = _setup_runtime(index_dir)
        call_counter = {"n": 0}
        orig = store.create_fts_index
        def _counting(*a, **kw):
            call_counter["n"] += 1
            return orig(*a, **kw)
        store.create_fts_index = _counting  # type: ignore[method-assign]
        try:
            docs = [_fake_record(f"id{i:02d}", f"body {i}") for i in range(6)]
            _process_docs(docs, concurrency=3)
            assert call_counter["n"] == 0, (
                f"_process_docs called create_fts_index {call_counter['n']} times; "
                "FTS rebuild must be deferred to end-of-flow"
            )
        finally:
            _teardown_runtime()
