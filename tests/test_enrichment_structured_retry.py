"""What actually lands in the index when a structured enrichment call misfires.

#1097: the enrichment client judged a first structured response by the
provider's token accounting instead of by the payload. That was wrong in both
directions — complete answers whose `usage.completion_tokens` overshot
`max_tokens` were discarded and re-requested (a second full LLM call per
document), while answers that parsed but omitted a required enrichment field
were returned unchecked and written degraded with no retry at all.

These tests drive the real LiteLLM client over stubbed HTTP responses through
the real indexing task, then read the row back out of LanceDB — the assertion
is on stored metadata, not on a mock.
"""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

import flow_index_vault as fiv
from core import enrichment_telemetry
from extractors import begin_degradation_capture, collect_degradations
from lancedb_store import LanceDBStore
from providers.llm.litellm_llm import LiteLLMGenerator


class _MockEmbed:
    def embed_texts(self, texts):
        return [[0.1] * 768 for _ in texts]

    def embed_query(self, q):
        return [0.1] * 768


def _enrichment_payload(**overrides) -> str:
    payload = {
        "summary": "Boiler service visit scheduled for the Ashfield site.",
        "doc_type": ["report"],
        "topics": ["maintenance"],
        "keywords": ["boiler", "inspection"],
    }
    payload.update(overrides)
    return json.dumps(payload)


def _response(content: str, *, completion_tokens: int, finish_reason: str = "stop"):
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-litellm",
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {"content": content, "reasoning_content": ""},
                }
            ],
            "usage": {"completion_tokens": completion_tokens},
        },
        request=httpx.Request("POST", "http://litellm.local/v1/chat/completions"),
    )


@pytest.fixture
def runtime(tmp_path):
    logger_patch = patch("flow_index_vault.get_run_logger", return_value=MagicMock())
    logger_patch.start()
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    store = LanceDBStore(tmp_path / "index", "chunks")
    from llama_index.core.node_parser import SentenceSplitter

    enrichment_telemetry.reset()
    fiv._RUNTIME.clear()
    fiv._RUNTIME.update({
        "store": store,
        "embed_provider": _MockEmbed(),
        "splitter": SentenceSplitter(chunk_size=512, chunk_overlap=20),
        "config": {
            "dedupe": {"enabled": False},
            "enrichment": {"max_input_chars": 4000, "max_output_tokens": 5000},
            "pdf": {},
        },
    })
    yield docs_root, store
    logger_patch.stop()
    fiv._RUNTIME.clear()
    enrichment_telemetry.reset()


def _write_doc(docs_root) -> dict:
    path = docs_root / "quo-attachments/annie/boiler.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Boiler service visit booked for Tuesday at the Ashfield site.")
    return {
        "doc_id": "documents::002qy",
        "rel_path": "quo-attachments/annie/boiler.md",
        "abs_path": str(path),
        "mtime": path.stat().st_mtime,
        "size": path.stat().st_size,
        "ext": "md",
        "source_name": "documents",
    }


def _index_with_responses(doc: dict, responses: list) -> list:
    """Index one document against a scripted sequence of LLM HTTP responses."""
    generator = LiteLLMGenerator(
        model="ollama-deepseek-v4-pro",
        base_url="http://litellm.local/v1",
        api_key="secret-key",
    )
    fiv._RUNTIME["llm_generator"] = generator
    begin_degradation_capture()
    with patch(
        "providers.llm.litellm_llm.httpx.post", side_effect=responses
    ) as post:
        fiv.process_doc_task.fn(doc)
    return post.call_args_list


def _stored_metadata(store: LanceDBStore, doc_id: str) -> dict:
    rows = (
        store._vs.table.to_lance()
        .to_table(columns=["doc_id", "metadata"])
        .to_pylist()
    )
    rows = [r for r in rows if r["doc_id"] == doc_id]
    assert rows, f"{doc_id} is not in the index"
    return rows[0]["metadata"]


def test_malformed_first_response_then_valid_retry_stores_required_metadata(runtime):
    """The reasoning blow-up path: the first response really is truncated, the
    reasoning-disabled retry is good, and the retry's metadata is what lands."""
    docs_root, store = runtime
    doc = _write_doc(docs_root)

    calls = _index_with_responses(doc, [
        _response("", completion_tokens=5000, finish_reason="length"),
        _response(_enrichment_payload(), completion_tokens=430),
    ])

    assert len(calls) == 2
    assert calls[1].kwargs["json"]["reasoning_effort"] == "none"
    stored = _stored_metadata(store, doc["doc_id"])
    assert stored["enr_doc_type"] == "report"
    assert stored["enr_summary"].startswith("Boiler service visit")
    assert not collect_degradations()


def test_double_failure_stores_a_degraded_row_without_required_fields(runtime):
    """Both responses omit doc_type. The row is still written (a partial row
    beats no row) but carries no enrichment, and the degradation is permanent —
    the model answered in full, so re-queueing it forever would be wrong."""
    docs_root, store = runtime
    doc = _write_doc(docs_root)

    calls = _index_with_responses(doc, [
        _response(_enrichment_payload(doc_type=[]), completion_tokens=430),
        _response(_enrichment_payload(doc_type=[]), completion_tokens=441),
    ])

    assert len(calls) == 2
    stored = _stored_metadata(store, doc["doc_id"])
    assert stored["enr_doc_type"] == ""
    assert stored["enr_summary"] == ""
    degradations = collect_degradations()
    assert [d.reason for d in degradations] == ["enrichment_failed"]
    assert not any(d.transient for d in degradations), (
        "a model that answered in full is not a provider outage"
    )


def test_a_complete_response_that_overshot_the_budget_costs_one_call(runtime):
    """The regression this ticket is named for: 272 of 274 production responses
    billed above max_tokens were complete and schema-valid, and every one was
    thrown away and re-requested."""
    docs_root, store = runtime
    doc = _write_doc(docs_root)

    calls = _index_with_responses(doc, [
        _response(_enrichment_payload(), completion_tokens=6201),
    ])

    assert len(calls) == 1, "a usable structured response was re-requested"
    assert _stored_metadata(store, doc["doc_id"])["enr_doc_type"] == "report"


def test_run_telemetry_reports_validity_retries_and_degraded_writes(runtime):
    """Acceptance: the run summary answers first-pass validity, retry count and
    degraded indexed writes without grepping per-document warnings."""
    docs_root, store = runtime
    doc = _write_doc(docs_root)

    _index_with_responses(doc, [_response(_enrichment_payload(), completion_tokens=6201)])
    _index_with_responses(doc, [
        _response("", completion_tokens=5000, finish_reason="length"),
        _response(_enrichment_payload(), completion_tokens=430),
    ])

    stats = fiv._enrichment_run_telemetry(1)
    assert stats == {
        "attempts": 2,
        "first_pass_usable": 1,
        "retries": 1,
        "retries_recovered": 1,
        "degraded_writes": 1,
    }


def test_run_telemetry_reaches_index_metadata(runtime, tmp_path):
    docs_root, store = runtime
    doc = _write_doc(docs_root)
    _index_with_responses(doc, [_response(_enrichment_payload(), completion_tokens=6201)])

    fiv.write_index_metadata_task.fn(
        tmp_path, 1, 3, None, None, enrichment=fiv._enrichment_run_telemetry(0)
    )

    meta = json.loads((tmp_path / "index_metadata.json").read_text())
    assert meta["enrichment"]["attempts"] == 1
    assert meta["enrichment"]["first_pass_usable"] == 1
    assert meta["enrichment"]["degraded_writes"] == 0
