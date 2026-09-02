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
import logging
from unittest.mock import MagicMock, patch

import httpx
import pytest

import flow_index_vault as fiv
from core import enrichment_telemetry
from core.hook_outbox import HookOutbox
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
        "entities_people": [],
        "entities_places": ["Ashfield site"],
        "entities_orgs": [],
        "entities_dates": ["Tuesday"],
        "topics": ["maintenance"],
        "keywords": ["boiler", "inspection"],
        "key_facts": ["Boiler service visit is booked for Tuesday."],
        "suggested_tags": ["maintenance"],
        "suggested_folder": "Properties/Ashfield",
        "importance": 0.6,
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
            "index_root": str(tmp_path / "index"),
            "event_hooks": {
                "enabled": True,
                "hooks": [{"name": "test-sink", "events": ["document.indexed"]}],
            },
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
    with (
        patch("providers.llm.litellm_llm.httpx.post", side_effect=responses) as post,
        patch(
            "flow_index_vault.drain_due",
            return_value={"accepted": 0, "retry_pending": 1, "redrive_required": 0},
        ),
    ):
        fiv.process_doc_task.fn(doc)
    return post.call_args_list


def _stored_metadata(store: LanceDBStore, doc_id: str) -> dict:
    rows = _stored_rows(store, doc_id)
    assert rows, f"{doc_id} is not in the index"
    return rows[0]["metadata"]


def _stored_rows(store: LanceDBStore, doc_id: str) -> list[dict]:
    if store._vs._table is None:
        return []
    rows = (
        store._vs.table.to_lance()
        .to_table(columns=["doc_id", "metadata"])
        .to_pylist()
    )
    return [r for r in rows if r["doc_id"] == doc_id]


def _queued_events(index_root) -> list[dict]:
    return [delivery.event for delivery in HookOutbox(index_root).due(limit=20)]


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


def test_double_contract_failure_stores_no_row_or_success_callback(runtime):
    """Exhausted structured retries are explicit failures, never index success."""
    docs_root, store = runtime
    doc = _write_doc(docs_root)

    calls = _index_with_responses(doc, [
        _response(_enrichment_payload(doc_type=[]), completion_tokens=430),
        _response(_enrichment_payload(doc_type=[]), completion_tokens=441),
    ])

    assert len(calls) == 2
    assert not _stored_rows(store, doc["doc_id"])
    assert not _queued_events(fiv._RUNTIME["config"]["index_root"])
    degradations = collect_degradations()
    assert [d.reason for d in degradations] == ["enrichment_failed"]
    assert not any(d.transient for d in degradations), (
        "a model that answered in full is not a provider outage"
    )


def test_incomplete_schema_token_response_retries_before_lance_and_callback(runtime):
    """#1918: parseable JSON may still violate the complete output contract."""
    docs_root, store = runtime
    doc = _write_doc(docs_root)
    malformed = _enrichment_payload(
        key_facts=["importance", "suggested_tags", "suggested_folder"]
    )
    malformed_payload = json.loads(malformed)
    for field in ("importance", "suggested_tags", "suggested_folder"):
        malformed_payload.pop(field)

    calls = _index_with_responses(doc, [
        _response(json.dumps(malformed_payload), completion_tokens=497),
        _response(_enrichment_payload(), completion_tokens=430),
    ])

    assert len(calls) == 2
    assert calls[1].kwargs["json"]["reasoning_effort"] == "none"
    stored = _stored_metadata(store, doc["doc_id"])
    assert json.loads(stored["enr_key_facts"]) == [
        "Boiler service visit is booked for Tuesday."
    ]
    events = _queued_events(fiv._RUNTIME["config"]["index_root"])
    assert [event["doc_id"] for event in events] == [doc["doc_id"]]
    assert json.loads(events[0]["metadata"]["enr_key_facts"]) == [
        "Boiler service visit is booked for Tuesday."
    ]


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


def test_qwen_validation_retry_records_unusable_budget_first_pass(runtime, caplog):
    """A discarded first answer must still reach the telemetry seam.

    qwen-bulk's in-request JSON validation retried a budget-truncated answer
    inside the request helper and returned only the repaired one. The outer
    observer then scored the document as a clean first pass and never emitted
    the ``finish_reason=length`` warning #1151's outcome check counts.
    """
    docs_root, store = runtime
    doc = _write_doc(docs_root)
    generator = LiteLLMGenerator(
        model="qwen-bulk",
        base_url="http://litellm.local/v1",
        api_key="secret-key",
    )
    fiv._RUNTIME["llm_generator"] = generator
    begin_degradation_capture()

    with caplog.at_level(logging.WARNING, logger="providers.llm.litellm_llm"):
        with patch(
            "providers.llm.litellm_llm.httpx.post",
            side_effect=[
                _response("", completion_tokens=5000, finish_reason="length"),
                _response(_enrichment_payload(), completion_tokens=430),
            ],
        ) as post:
            fiv.process_doc_task.fn(doc)

    assert post.call_count == 2
    assert "LiteLLM structured response was not usable" in caplog.text
    assert "finish_reason=length" in caplog.text
    assert fiv._enrichment_run_telemetry(0) == {
        "attempts": 1,
        "first_pass_usable": 0,
        "retries": 1,
        "retries_recovered": 1,
        "degraded_writes": 0,
    }
    assert _stored_metadata(store, doc["doc_id"])["enr_doc_type"] == "report"


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
