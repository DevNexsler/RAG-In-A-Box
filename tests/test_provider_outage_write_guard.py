"""A provider outage must never overwrite a good index entry with an empty one.

#0619: five litellm-proxy connection-refused bursts in 14h. For every document
in flight the describe/enrichment call failed, the flow logged the degradation
— and then upserted chunks built from nothing over the previously enriched row.
54 docs were silently demoted to bare chunks while the run reported
`Completed()`.

The rule under test: never replace content with the ABSENCE of content because
the provider was down. A transient degradation (ConnectError, ReadTimeout, 5xx)
on an already-indexed document keeps the indexed version and defers the rewrite
to the degraded-ledger self-heal. Documents that were never indexed are still
written — a partial row beats no row — and permanent failures still overwrite,
because those say something true about the document.
"""

import json

import httpx
import pytest
from unittest.mock import MagicMock, patch

import flow_index_vault as fiv
from extractors import ExtractionResult, begin_degradation_capture, note_degradation
from lancedb_store import LanceDBStore


class _MockEmbed:
    def embed_texts(self, texts):
        return [[0.1] * 768 for _ in texts]

    def embed_query(self, q):
        return [0.1] * 768


class _GoodLLM:
    """Enriches normally."""

    def generate(self, prompt, max_tokens=512):
        return json.dumps({
            "summary": "Photo of bags left by the bins, reported as a theft risk.",
            "doc_type": ["image", "message"],
            "topics": ["property", "theft", "security"],
        })


class _DownLLM:
    """The provider container is being recreated — the socket refuses/stalls."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def generate(self, prompt, max_tokens=512):
        raise self.exc


class _BrokenLLM:
    """Reachable, but its answer is unusable — permanent, says something real."""

    def generate(self, prompt, max_tokens=512):
        return json.dumps({"topics": ["whatever"]})  # no summary/doc_type


@pytest.fixture
def runtime(tmp_path):
    logger_patch = patch("flow_index_vault.get_run_logger", return_value=MagicMock())
    logger_patch.start()
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    store = LanceDBStore(tmp_path / "index", "chunks")
    from llama_index.core.node_parser import SentenceSplitter

    fiv._RUNTIME.clear()
    fiv._RUNTIME.update({
        "store": store,
        "embed_provider": _MockEmbed(),
        "splitter": SentenceSplitter(chunk_size=512, chunk_overlap=20),
        "config": {
            "dedupe": {"enabled": False},
            "enrichment": {"max_input_chars": 4000},
            "pdf": {},
        },
    })
    yield docs_root, store
    logger_patch.stop()
    fiv._RUNTIME.clear()


def _write_doc(docs_root, rel: str, body: str) -> dict:
    path = docs_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return {
        "doc_id": "documents::002qy",
        "rel_path": rel,
        "abs_path": str(path),
        "mtime": path.stat().st_mtime,
        "size": path.stat().st_size,
        "ext": "md",
        "source_name": "documents",
    }


def _index(doc: dict, generator) -> None:
    fiv._RUNTIME["llm_generator"] = generator
    begin_degradation_capture()
    fiv.process_doc_task.fn(doc)


def _stored_row(store: LanceDBStore, doc_id: str) -> dict:
    rows = (
        store._vs.table.to_lance()
        .to_table(columns=["doc_id", "text", "metadata"])
        .to_pylist()
    )
    rows = [r for r in rows if r["doc_id"] == doc_id]
    assert rows, f"{doc_id} is not in the index"
    return rows[0]


def _stored_metadata(store: LanceDBStore, doc_id: str) -> dict:
    return _stored_row(store, doc_id)["metadata"]


@pytest.mark.parametrize("outage", [
    httpx.ConnectError("[Errno 111] Connection refused"),
    httpx.ReadTimeout("timed out"),
])
def test_transient_outage_keeps_the_enriched_row(runtime, outage):
    docs_root, store = runtime
    doc = _write_doc(docs_root, "quo-attachments/annie/theft.md", "Bags left by the bins.")

    _index(doc, _GoodLLM())
    good = _stored_metadata(store, doc["doc_id"])
    assert good["enr_summary"].startswith("Photo of bags")

    _index(doc, _DownLLM(outage))

    after = _stored_metadata(store, doc["doc_id"])
    assert after["enr_summary"] == good["enr_summary"], (
        "a provider outage overwrote the enriched row with an empty one"
    )
    assert after["enr_topics"] == good["enr_topics"]


def test_transient_outage_is_counted_for_the_run_summary(runtime):
    docs_root, store = runtime
    doc = _write_doc(docs_root, "quo-attachments/annie/theft.md", "Bags left by the bins.")

    _index(doc, _GoodLLM())
    _index(doc, _DownLLM(httpx.ConnectError("[Errno 111] Connection refused")))

    assert doc["doc_id"] in fiv._RUNTIME.get("provider_unavailable", set())


def test_transient_outage_on_any_provider_stage_keeps_the_row(runtime):
    """The guard is about transience, not about which provider failed:
    ocr_describe_failed must protect the row exactly like enrichment_failed."""
    docs_root, store = runtime
    doc = _write_doc(
        docs_root,
        "quo-attachments/annie/theft.md",
        "Bags left by the bins. A large pile of black bags sits by the gate.",
    )

    _index(doc, _GoodLLM())
    assert "large pile" in _stored_row(store, doc["doc_id"])["text"]

    def _degraded_extract(**kwargs):
        # Vision came back empty because the endpoint refused; only the caption
        # survives. Enrichment itself is healthy this time.
        note_degradation("ocr_describe_failed", transient=True)
        return ExtractionResult.from_text("Bags left by the bins.")

    with patch("flow_index_vault.extract_text", side_effect=_degraded_extract):
        _index(doc, _GoodLLM())

    assert "large pile" in _stored_row(store, doc["doc_id"])["text"], (
        "an OCR/vision outage truncated the indexed content"
    )


def test_first_index_still_writes_during_an_outage(runtime):
    """Nothing to protect: a partial row beats no row, and the degraded ledger
    re-processes it on a later run."""
    docs_root, store = runtime
    doc = _write_doc(docs_root, "quo-attachments/annie/theft.md", "Bags left by the bins.")

    _index(doc, _DownLLM(httpx.ConnectError("[Errno 111] Connection refused")))

    stored = _stored_metadata(store, doc["doc_id"])
    assert stored["enr_summary"] == ""


def test_permanent_enrichment_failure_still_rewrites(runtime):
    """A reachable provider that cannot enrich this document is information
    about the document — it must not freeze the row forever."""
    docs_root, store = runtime
    doc = _write_doc(docs_root, "quo-attachments/annie/theft.md", "Bags left by the bins.")

    _index(doc, _GoodLLM())
    doc["mtime"] += 1
    _index(doc, _BrokenLLM())

    assert _stored_metadata(store, doc["doc_id"])["enr_summary"] == ""
