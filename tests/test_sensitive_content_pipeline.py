"""Credential-bearing source text must stop before external providers and LanceDB."""

import json
from unittest.mock import MagicMock

import pytest
from llama_index.core.node_parser import SentenceSplitter

import flow_index_vault as fiv
from communication_context import (
    CommunicationItem,
    CommunicationMessage,
    ContextEnvelope,
    format_context_envelope_for_prompt,
)
from extractors import ExtractionResult, begin_degradation_capture
from lancedb_store import LanceDBStore
from sources.base import SourceRecord


class _Source:
    name = "comm_messages"

    def extract(self, record):
        return ExtractionResult.from_text(
            record.metadata["_text"],
            frontmatter={
                key: value
                for key, value in record.metadata.items()
                if key != "_text"
            },
        )


class _CapturingLLM:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, max_tokens=512):
        self.prompts.append(prompt)
        return json.dumps(
            {
                "summary": "A communication message.",
                "doc_type": ["message"],
                "topics": ["operations"],
            }
        )


class _CapturingEmbed:
    def __init__(self):
        self.texts = []

    def embed_texts(self, texts):
        self.texts.extend(texts)
        return [[0.1] * 768 for _ in texts]

    def embed_query(self, query):
        return [0.1] * 768


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(fiv, "get_run_logger", lambda: MagicMock())
    store = LanceDBStore(tmp_path / "index", "chunks")
    llm = _CapturingLLM()
    embed = _CapturingEmbed()
    fiv._RUNTIME.clear()
    fiv._RUNTIME.update(
        {
            "store": store,
            "embed_provider": embed,
            "splitter": SentenceSplitter(chunk_size=512, chunk_overlap=20),
            "config": {"enrichment": {"max_input_chars": 4000}},
            "sources_by_name": {"comm_messages": _Source()},
            "llm_generator": llm,
        }
    )
    yield store, llm, embed
    fiv._RUNTIME.clear()


def _index_message(body, pipeline, *, sender="Pat", message_id="message-1"):
    store, llm, embed = pipeline
    doc_id = f"comm_messages::zoho_cliq/{message_id}"
    record = SourceRecord(
        doc_id=f"zoho_cliq/{message_id}",
        source_type="pg_message",
        natural_key=f"zoho_cliq/{message_id}",
        mtime=1.0,
        size=len(body.encode()),
        metadata={
            "_text": body,
            "source": "zoho_cliq",
            "source_message_id": message_id,
            "sender": sender,
            "sent_at": "2026-08-12T05:34:12Z",
        },
    )
    fiv._RUNTIME["source_records_by_ns_doc_id"] = {doc_id: record}
    begin_degradation_capture()
    fiv.process_doc_task.fn(
        {
            "doc_id": doc_id,
            "rel_path": f"postgres/comm_messages/{message_id}",
            "mtime": 1.0,
            "size": len(body.encode()),
            "source_type": "pg_message",
            "source_name": "comm_messages",
        }
    )
    return doc_id, store, llm, embed


def _stored_rows(store, doc_id):
    if not store.contains_doc_id(doc_id):
        return []
    return [
        row
        for row in store._vs.table.to_lance().to_table().to_pylist()
        if row["doc_id"] == doc_id
    ]


def test_credentials_are_redacted_before_provider_and_storage_boundaries(pipeline):
    credentials = (
        "AES-PRERELEASE:synthetic-live-token-0123456789abcdef:expires=2099-01-01T00:00:00Z",
        "ghp_0123456789abcdefghijklmnopqrstuvwxyzAB",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJzeW50aGV0aWMifQ.synthetic-signature-value",
    )
    body = "Use temporary credentials for the migration: " + " ".join(credentials)

    doc_id, store, llm, embed = _index_message(body, pipeline)
    rows = _stored_rows(store, doc_id)
    provider_payloads = llm.prompts + embed.texts
    stored_payload = json.dumps(rows, default=str)

    assert rows
    assert all(secret not in payload for secret in credentials for payload in provider_payloads)
    assert all(secret not in stored_payload for secret in credentials)
    assert "[REDACTED CREDENTIAL]" in stored_payload


def test_system_credential_message_is_quarantined_before_providers(pipeline):
    credential = (
        "AES-PRERELEASE:synthetic-system-token-0123456789abcdef:"
        "expires=2099-01-01T00:00:00Z"
    )

    doc_id, store, llm, embed = _index_message(
        credential,
        pipeline,
        sender="System",
        message_id="system-message",
    )

    assert llm.prompts == []
    assert embed.texts == []
    assert _stored_rows(store, doc_id) == []


def test_normal_communication_still_enriches_embeds_and_is_stored(pipeline):
    body = "Permit inspection moved to Thursday at 10:30 AM."

    doc_id, store, llm, embed = _index_message(body, pipeline, message_id="normal")
    rows = _stored_rows(store, doc_id)

    assert llm.prompts and body in llm.prompts[0]
    assert embed.texts and any(body in text for text in embed.texts)
    assert rows and any(body in row["text"] for row in rows)


def test_neighboring_context_credentials_are_redacted_when_rendered():
    secret = "ghp_0123456789abcdefghijklmnopqrstuvwxyzAB"
    envelope = ContextEnvelope(
        primary_item=CommunicationItem(doc_id="documents::attachment"),
        same_channel_before=[
            CommunicationMessage(
                message_id="before",
                sender="System",
                sent_at="2026-08-12T05:34:12Z",
                text=f"Temporary token: {secret}",
            )
        ],
    )

    rendered = format_context_envelope_for_prompt(envelope)

    assert secret not in rendered
    assert "[REDACTED CREDENTIAL]" in rendered
