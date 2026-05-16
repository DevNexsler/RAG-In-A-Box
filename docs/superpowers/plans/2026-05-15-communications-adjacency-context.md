# Communications Adjacency Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add scalable index-time communication context enrichment so messages and attachments get same-channel/thread context metadata without hardcoded property logic.

**Architecture:** Introduce a source-agnostic `CommunicationContextProvider` that returns labeled context envelopes. The first backend reads same-channel/thread windows from source records and attachment sidecars; future LanceGraph support swaps only this provider backend. Enrichment stays explicit about primary content vs nearby candidate context and stores atomic/context metadata separately in LanceDB metadata.

**Tech Stack:** Python 3, pytest, LanceDB metadata fields, existing source abstraction, existing LLM enrichment pipeline, GitNexus impact checks.

---

## File Structure

- Create: `communication_context.py`
  - Dataclasses for communication items/messages/envelopes.
  - Source-window provider.
  - Sidecar parsing helpers for attachment provenance.
  - Formatting helper for LLM prompts.
- Modify: `doc_enrichment.py`
  - Add context-specific enrichment fields.
  - Add optional `context_text` / `context_envelope_text` input.
  - Update prompt to label nearby messages as candidate context.
  - Parse and normalize new atomic/context metadata fields.
- Modify: `flow_index_vault.py`
  - Build provider runtime once per index run.
  - Convert each source record/doc into `CommunicationItem` when possible.
  - Request context envelope before enrichment.
  - Store provenance/context metadata on `doc_meta`.
- Modify: `sources/postgres.py`
  - Preserve all configured metadata columns as before.
  - No source-specific context logic here unless needed for stable message identity helpers.
- Modify: `config.yaml.example`, `config.local.yaml.example`, `config.vps.yaml.example`
  - Add optional `communication_context` defaults.
- Test: `tests/test_communication_context.py`
  - Unit tests for sidecars, same-channel windows, batch grouping, and boundary safety.
- Test: `tests/test_doc_enrichment.py` or existing enrichment tests
  - Prompt/parser behavior for context-aware enrichment.
- Test: `tests/test_scan.py`
  - Index integration with fake context provider/runtime metadata.
- Test: `tests/test_config.py` or `tests/test_config_sources.py`
  - Config defaults/validation if config loader gets explicit defaults.

Implementation must preserve unrelated existing dirty files. Stage only files touched for this feature.

## Task 1: Add Communication Context Model And Sidecar Parsing

**Files:**
- Create: `communication_context.py`
- Test: `tests/test_communication_context.py`

- [ ] **Step 1: Run GitNexus impact check for new module integration targets**

Run before editing any existing symbol:

```bash
python - <<'PY'
print("No existing symbol modified in this step; new module only.")
PY
```

Expected: no existing symbol impact required for the new module.

- [ ] **Step 2: Write failing tests for sidecar parsing**

Create `tests/test_communication_context.py` with tests like:

```python
from pathlib import Path

from communication_context import communication_item_from_sidecar


def test_sidecar_attachment_becomes_communication_item(tmp_path: Path):
    media = tmp_path / "2026-04-22T14-56-22Z__msg4442__mm0@000kM@.jpg"
    media.write_bytes(b"fake")
    sidecar = tmp_path / "2026-04-22T14-56-22Z__msg4442__mm0@000kD@.json"
    sidecar.write_text(
        """
        {
          "source": "zoho_cliq",
          "message": {
            "message_id": "4442",
            "source_message_id": "1776869782220_21353330717388",
            "sent_at": "2026-04-22T14:56:22.220Z",
            "from": {"name": "Joycelyn Smith"}
          },
          "channel": {
            "source_channel_id": "2242125288797599446",
            "channel_type": "conversation"
          },
          "media": {
            "media_index": 0,
            "media_type": "image/jpeg",
            "original_filename": "IMG_2133.HEIC"
          }
        }
        """
    )

    item = communication_item_from_sidecar(media, sidecar)

    assert item.origin_source == "zoho_cliq"
    assert item.message_id == "4442"
    assert item.source_message_id == "1776869782220_21353330717388"
    assert item.channel_id == "2242125288797599446"
    assert item.sender == "Joycelyn Smith"
    assert item.attachment_index == "0"
    assert item.batch_key.startswith("zoho_cliq:2242125288797599446:")
```

- [ ] **Step 3: Run test to verify it fails**

Run:

```bash
PYTHONPATH=. pytest tests/test_communication_context.py::test_sidecar_attachment_becomes_communication_item -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'communication_context'`.

- [ ] **Step 4: Implement dataclasses and sidecar parser**

Create `communication_context.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CommunicationMessage:
    message_id: str = ""
    source_message_id: str = ""
    sender: str = ""
    sent_at: str = ""
    text: str = ""


@dataclass(frozen=True)
class CommunicationItem:
    doc_id: str
    rel_path: str = ""
    source_name: str = ""
    source_type: str = ""
    origin_source: str = ""
    message_id: str = ""
    source_message_id: str = ""
    channel_id: str = ""
    channel_name: str = ""
    thread_id: str = ""
    sender: str = ""
    sent_at: str = ""
    batch_key: str = ""
    attachment_index: str = ""
    sidecar_path: str = ""
    primary_text: str = ""


@dataclass(frozen=True)
class ContextEnvelope:
    primary_item: CommunicationItem
    same_channel_before: list[CommunicationMessage] = field(default_factory=list)
    same_channel_after: list[CommunicationMessage] = field(default_factory=list)
    nearest_nonempty_before: CommunicationMessage | None = None
    nearest_nonempty_after: CommunicationMessage | None = None
    same_batch: list[CommunicationItem] = field(default_factory=list)


def communication_item_from_sidecar(media_path: Path, sidecar_path: Path) -> CommunicationItem:
    payload = json.loads(sidecar_path.read_text())
    message = payload.get("message") or {}
    channel = payload.get("channel") or {}
    media = payload.get("media") or {}
    sender = ((message.get("from") or {}).get("name") or "").strip()
    origin_source = str(payload.get("source") or "").strip()
    sent_at = str(message.get("sent_at") or "").strip()
    channel_id = str(channel.get("source_channel_id") or "").strip()
    message_id = str(message.get("message_id") or "").strip()
    source_message_id = str(message.get("source_message_id") or "").strip()
    batch_key = _batch_key(origin_source, channel_id, sent_at)
    return CommunicationItem(
        doc_id="",
        rel_path=str(media_path),
        origin_source=origin_source,
        message_id=message_id,
        source_message_id=source_message_id,
        channel_id=channel_id,
        thread_id=str(channel.get("thread_id") or ""),
        sender=sender,
        sent_at=sent_at,
        batch_key=batch_key,
        attachment_index=str(media.get("media_index") or "0"),
        sidecar_path=str(sidecar_path),
    )


def _batch_key(origin_source: str, channel_id: str, sent_at: str) -> str:
    rounded = _round_timestamp_to_second(sent_at)
    return f"{origin_source}:{channel_id}:{rounded}" if origin_source or channel_id or rounded else ""


def _round_timestamp_to_second(value: str) -> str:
    if not value:
        return ""
    normalized = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
```

- [ ] **Step 5: Run test to verify it passes**

Run:

```bash
PYTHONPATH=. pytest tests/test_communication_context.py::test_sidecar_attachment_becomes_communication_item -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add communication_context.py tests/test_communication_context.py
git commit -m "feat: add communication context model"
```

## Task 2: Add Source-Window Context Provider

**Files:**
- Modify: `communication_context.py`
- Test: `tests/test_communication_context.py`

- [ ] **Step 1: Run GitNexus impact analysis**

Run:

```bash
# GitNexus MCP: impact(target="CommunicationItem", direction="upstream")
# If GitNexus cannot resolve new symbols yet, run npx gitnexus analyze after Task 1 commit, then retry.
```

Expected: low/no upstream callers beyond tests at this stage.

- [ ] **Step 2: Write failing tests for channel-scoped context**

Append tests:

```python
from communication_context import (
    CommunicationItem,
    CommunicationMessage,
    SourceWindowContextProvider,
)


def test_provider_uses_same_channel_only():
    target = CommunicationItem(
        doc_id="attachment-1",
        origin_source="zoho_cliq",
        channel_id="chan-a",
        sent_at="2026-04-22T14:56:22Z",
    )
    messages = [
        CommunicationMessage(message_id="1", source_message_id="a1", sent_at="2026-04-22T14:55:00Z", text="Unit E"),
        CommunicationMessage(message_id="2", source_message_id="b1", sent_at="2026-04-22T14:55:30Z", text="Other channel"),
        CommunicationMessage(message_id="3", source_message_id="a2", sent_at="2026-04-22T14:57:00Z", text="Also this"),
    ]
    message_channels = {"1": "chan-a", "2": "chan-b", "3": "chan-a"}

    provider = SourceWindowContextProvider.from_messages(
        messages,
        message_channels=message_channels,
        window_before=5,
        window_after=5,
    )

    envelope = provider.get_context_envelope(target)

    assert [m.message_id for m in envelope.same_channel_before] == ["1"]
    assert [m.message_id for m in envelope.same_channel_after] == ["3"]
```

- [ ] **Step 3: Run test to verify it fails**

Run:

```bash
PYTHONPATH=. pytest tests/test_communication_context.py::test_provider_uses_same_channel_only -v
```

Expected: FAIL because `SourceWindowContextProvider` does not exist.

- [ ] **Step 4: Implement provider**

Add:

```python
class SourceWindowContextProvider:
    def __init__(
        self,
        messages_by_scope: dict[tuple[str, str, str], list[CommunicationMessage]],
        *,
        window_before: int = 5,
        window_after: int = 5,
    ) -> None:
        self._messages_by_scope = messages_by_scope
        self._window_before = window_before
        self._window_after = window_after

    @classmethod
    def from_messages(
        cls,
        messages: list[CommunicationMessage],
        *,
        message_channels: dict[str, str],
        message_sources: dict[str, str] | None = None,
        message_threads: dict[str, str] | None = None,
        window_before: int = 5,
        window_after: int = 5,
    ) -> "SourceWindowContextProvider":
        grouped: dict[tuple[str, str, str], list[CommunicationMessage]] = {}
        for msg in messages:
            channel_id = message_channels.get(msg.message_id, "")
            origin_source = (message_sources or {}).get(msg.message_id, "")
            thread_id = (message_threads or {}).get(msg.message_id, "")
            grouped.setdefault((origin_source, channel_id, thread_id), []).append(msg)
        for group in grouped.values():
            group.sort(key=lambda m: m.sent_at)
        return cls(grouped, window_before=window_before, window_after=window_after)

    def get_context_envelope(self, item: CommunicationItem) -> ContextEnvelope:
        scope = (item.origin_source, item.channel_id, item.thread_id)
        messages = self._messages_by_scope.get(scope)
        if not messages and item.origin_source:
            messages = self._messages_by_scope.get(("", item.channel_id, item.thread_id))
        messages = messages or []
        before = [m for m in messages if m.sent_at <= item.sent_at][-self._window_before:]
        after = [m for m in messages if m.sent_at > item.sent_at][:self._window_after]
        nearest_before = next((m for m in reversed(before) if m.text.strip()), None)
        nearest_after = next((m for m in after if m.text.strip()), None)
        return ContextEnvelope(
            primary_item=item,
            same_channel_before=before,
            same_channel_after=after,
            nearest_nonempty_before=nearest_before,
            nearest_nonempty_after=nearest_after,
        )
```

- [ ] **Step 5: Add tests for empty body and ordering**

Add:

```python
def test_provider_tracks_nearest_nonempty_messages():
    target = CommunicationItem(doc_id="a", channel_id="c", sent_at="2026-01-01T10:00:00Z")
    messages = [
        CommunicationMessage(message_id="1", sent_at="2026-01-01T09:59:00Z", text="Building 54"),
        CommunicationMessage(message_id="2", sent_at="2026-01-01T09:59:30Z", text=""),
        CommunicationMessage(message_id="3", sent_at="2026-01-01T10:00:30Z", text=""),
        CommunicationMessage(message_id="4", sent_at="2026-01-01T10:01:00Z", text="Unit E"),
    ]
    provider = SourceWindowContextProvider.from_messages(
        messages,
        message_channels={m.message_id: "c" for m in messages},
    )

    envelope = provider.get_context_envelope(target)

    assert envelope.nearest_nonempty_before.message_id == "1"
    assert envelope.nearest_nonempty_after.message_id == "4"
```

- [ ] **Step 6: Run provider tests**

Run:

```bash
PYTHONPATH=. pytest tests/test_communication_context.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add communication_context.py tests/test_communication_context.py
git commit -m "feat: add source-window communication context provider"
```

## Task 3: Make Enrichment Context-Aware

**Files:**
- Modify: `doc_enrichment.py`
- Test: `tests/test_doc_enrichment.py`

- [ ] **Step 1: Run GitNexus impact analysis**

Run:

```bash
# GitNexus MCP: impact(target="enrich_document", direction="upstream")
# GitNexus MCP: context(name="enrich_document")
```

Expected: direct callers include `flow_index_vault.py:process_doc_task` and tests. If risk is HIGH/CRITICAL, stop and report before edits.

- [ ] **Step 2: Write failing parser test for context fields**

Create or update `tests/test_doc_enrichment.py`:

```python
from doc_enrichment import parse_enrichment_response


def test_parse_context_enrichment_fields():
    parsed = parse_enrichment_response(
        """
        {
          "summary": "Photo of vehicle parts.",
          "doc_type": ["image"],
          "entities_people": [],
          "entities_places": [],
          "entities_orgs": [],
          "entities_dates": [],
          "topics": ["vehicle"],
          "keywords": ["car parts"],
          "key_facts": [],
          "suggested_tags": ["maintenance"],
          "suggested_folder": "Housing/Maintenance",
          "importance": 0.5,
          "atomic_entities_places": [],
          "context_entities_places": ["54 S Broad Main Unit E"],
          "context_topics": ["basement storage"],
          "context_key_facts": ["Nearby message says the photos are from Unit E."],
          "context_relationship": "batch_label",
          "context_confidence": "high",
          "context_source_message_ids": ["4434"],
          "context_warning": ""
        }
        """
    )

    assert parsed["enr_context_entities_places"] == "54 S Broad Main Unit E"
    assert parsed["enr_context_relationship"] == "batch_label"
    assert parsed["enr_context_confidence"] == "high"
    assert parsed["enr_context_source_message_ids"] == "4434"
```

- [ ] **Step 3: Run test to verify it fails**

Run:

```bash
PYTHONPATH=. pytest tests/test_doc_enrichment.py::test_parse_context_enrichment_fields -v
```

Expected: FAIL because new fields are not normalized.

- [ ] **Step 4: Add enrichment fields**

Update `doc_enrichment.py`:

```python
_CONTEXT_KEYS_RAW = (
    "atomic_entities_people",
    "atomic_entities_places",
    "atomic_entities_orgs",
    "atomic_entities_dates",
    "atomic_topics",
    "context_entities_people",
    "context_entities_places",
    "context_entities_orgs",
    "context_entities_dates",
    "context_topics",
    "context_key_facts",
    "context_relationship",
    "context_confidence",
    "context_source_message_ids",
    "context_warning",
)

ENRICHMENT_FIELDS = tuple(f"enr_{k}" for k in (*_ENRICHMENT_KEYS_RAW, *_CONTEXT_KEYS_RAW))
```

Update `_normalize_enrichment()` to normalize context list fields with existing `_normalize_list()` and JSON-list behavior for key facts if needed.

- [ ] **Step 5: Add optional context prompt input**

Change `enrich_document()` signature:

```python
def enrich_document(
    text: str,
    title: str,
    source_type: str,
    generator: "LLMGenerator",
    max_input_chars: int = 4000,
    max_output_tokens: int = 512,
    taxonomy_store: "TaxonomyStore | None" = None,
    context_text: str = "",
) -> dict[str, str]:
```

Use a separate prompt template when `context_text` is non-empty, labeling primary item and nearby context candidates exactly as the design doc requires.

- [ ] **Step 6: Add prompt test**

Add fake generator test:

```python
class CapturingGenerator:
    def __init__(self):
        self.prompt = ""

    def generate(self, prompt, max_tokens):
        self.prompt = prompt
        return '{"summary":"ok","doc_type":[],"entities_people":[],"entities_places":[],"entities_orgs":[],"entities_dates":[],"topics":[],"keywords":[],"key_facts":[],"suggested_tags":[],"suggested_folder":"","importance":0.5,"context_relationship":"nearby_ambiguous","context_confidence":"ambiguous"}'


def test_enrich_document_labels_nearby_context_candidates():
    generator = CapturingGenerator()

    enrich_document(
        "image notes",
        "photo.jpg",
        "img",
        generator,
        context_text="[before] Unit E",
    )

    assert "PRIMARY ITEM" in generator.prompt
    assert "NEARBY SAME-CHANNEL CONTEXT CANDIDATES" in generator.prompt
    assert "may or may not describe the primary item" in generator.prompt
```

- [ ] **Step 7: Run enrichment tests**

Run:

```bash
PYTHONPATH=. pytest tests/test_doc_enrichment.py tests/test_benchmark_scoring.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```bash
git add doc_enrichment.py tests/test_doc_enrichment.py
git commit -m "feat: add context-aware enrichment fields"
```

## Task 4: Integrate Context Provider Into Indexing

**Files:**
- Modify: `flow_index_vault.py`
- Modify: `communication_context.py`
- Test: `tests/test_scan.py`

- [ ] **Step 1: Run GitNexus impact analysis**

Run:

```bash
# GitNexus MCP: impact(target="process_doc_task", direction="upstream")
# GitNexus MCP: context(name="process_doc_task")
```

Expected: likely MEDIUM or higher because indexing flow is central. If HIGH/CRITICAL, report before editing.

- [ ] **Step 2: Write failing integration test**

Add test in `tests/test_scan.py`:

```python
def test_process_doc_task_passes_context_to_enrichment(monkeypatch):
    from flow_index_vault import process_doc_task, _RUNTIME
    from sources.base import SourceRecord
    from extractors import ExtractionResult
    from communication_context import CommunicationItem, ContextEnvelope, CommunicationMessage

    captured = {}

    class FakeSource:
        name = "documents"
        def extract(self, record):
            return ExtractionResult.from_text("image description", frontmatter={})

    class FakeProvider:
        def get_context_envelope(self, item):
            return ContextEnvelope(
                primary_item=item,
                same_channel_before=[
                    CommunicationMessage(message_id="m1", sent_at="2026-01-01T10:00:00Z", text="This is for Unit E")
                ],
            )

    class FakeStore:
        def upsert_nodes(self, nodes):
            captured["metadata"] = nodes[0].metadata

    class FakeEmbed:
        def embed_texts(self, texts):
            return [[0.1] * 768 for _ in texts]

    def fake_enrich_document(*args, **kwargs):
        captured["context_text"] = kwargs.get("context_text", "")
        return {
            "enr_summary": "summary",
            "enr_doc_type": "image",
            "enr_entities_people": "",
            "enr_entities_places": "",
            "enr_entities_orgs": "",
            "enr_entities_dates": "",
            "enr_topics": "",
            "enr_keywords": "",
            "enr_key_facts": "",
            "enr_suggested_tags": "",
            "enr_suggested_folder": "",
            "enr_importance": "0.5",
            "enr_context_entities_places": "Unit E",
            "enr_context_relationship": "batch_label",
            "enr_context_confidence": "high",
        }

    monkeypatch.setattr("flow_index_vault.enrich_document", fake_enrich_document)
    _RUNTIME.clear()
    _RUNTIME.update({
        "store": FakeStore(),
        "embed_provider": FakeEmbed(),
        "splitter": _FakeSplitter(),
        "config": {},
        "sources_by_name": {"documents": FakeSource()},
        "source_records_by_ns_doc_id": {
            "documents::photo": SourceRecord(
                doc_id="photo",
                source_type="img",
                natural_key="photo.jpg",
                mtime=1.0,
                size=10,
                metadata={"source": "zoho_cliq", "channel_id": "chan", "sent_at": "2026-01-01T10:00:01Z"},
            )
        },
        "communication_context_provider": FakeProvider(),
        "llm_generator": object(),
    })

    process_doc_task.fn({
        "doc_id": "documents::photo",
        "rel_path": "photo.jpg",
        "mtime": 1.0,
        "size": 10,
        "source_type": "img",
        "source_name": "documents",
    })

    assert "This is for Unit E" in captured["context_text"]
    assert captured["metadata"]["enr_context_entities_places"] == "Unit E"
    assert captured["metadata"]["enr_context_confidence"] == "high"
```

- [ ] **Step 3: Run test to verify it fails**

Run:

```bash
PYTHONPATH=. pytest tests/test_scan.py::test_process_doc_task_passes_context_to_enrichment -v
```

Expected: FAIL because context provider is not wired.

- [ ] **Step 4: Add helper to build CommunicationItem from doc/source record**

In `communication_context.py`, add:

```python
def communication_item_from_record(doc: dict, metadata: dict, primary_text: str = "") -> CommunicationItem | None:
    origin_source = str(metadata.get("source") or metadata.get("origin_source") or "").strip()
    channel_id = str(metadata.get("channel_id") or metadata.get("source_channel_id") or metadata.get("channel") or "").strip()
    sent_at = str(metadata.get("sent_at") or "").strip()
    source_message_id = str(metadata.get("source_message_id") or "").strip()
    message_id = str(metadata.get("message_id") or "").strip()
    if not any((origin_source, channel_id, sent_at, source_message_id, message_id)):
        return None
    return CommunicationItem(
        doc_id=str(doc.get("doc_id") or ""),
        rel_path=str(doc.get("rel_path") or ""),
        source_name=str(doc.get("source_name") or ""),
        source_type=str(doc.get("source_type") or ""),
        origin_source=origin_source,
        message_id=message_id,
        source_message_id=source_message_id,
        channel_id=channel_id,
        channel_name=str(metadata.get("channel_name") or metadata.get("channel") or ""),
        thread_id=str(metadata.get("thread_id") or ""),
        sender=str(metadata.get("sender") or ""),
        sent_at=sent_at,
        batch_key=str(metadata.get("batch_key") or ""),
        attachment_index=str(metadata.get("attachment_index") or ""),
        sidecar_path=str(metadata.get("sidecar_path") or ""),
        primary_text=primary_text,
    )
```

- [ ] **Step 5: Wire provider in process_doc_task**

In `process_doc_task`, after extraction and before `enrich_document()`:

```python
source_metadata = getattr(source_record, "metadata", {}) if source_record is not None else {}
comm_item = communication_item_from_record(doc, source_metadata, result.full_text)
context_text = ""
context_meta = {}
provider = _RUNTIME.get("communication_context_provider")
if comm_item is not None and provider is not None:
    envelope = provider.get_context_envelope(comm_item)
    context_text = format_context_envelope_for_prompt(envelope)
    context_meta = envelope_metadata(envelope)
```

Pass `context_text=context_text` into `enrich_document()`.

Merge `context_meta` into `doc_meta` before enrichment output, without overwriting existing core fields.

- [ ] **Step 6: Add prompt formatting helper**

In `communication_context.py`:

```python
def format_context_envelope_for_prompt(envelope: ContextEnvelope) -> str:
    lines = []
    for label, messages in (
        ("BEFORE", envelope.same_channel_before),
        ("AFTER", envelope.same_channel_after),
    ):
        for msg in messages:
            if msg.text.strip():
                lines.append(f"[{label} {msg.sent_at} message_id={msg.message_id}] {msg.sender}: {msg.text}")
    return "\n".join(lines)
```

- [ ] **Step 7: Run integration test**

Run:

```bash
PYTHONPATH=. pytest tests/test_scan.py::test_process_doc_task_passes_context_to_enrichment -v
```

Expected: PASS.

- [ ] **Step 8: Run focused indexing tests**

Run:

```bash
PYTHONPATH=. pytest tests/test_scan.py tests/test_store.py::test_extra_metadata_in_vector_search -v
```

Expected: PASS.

- [ ] **Step 9: Commit**

Run:

```bash
git add communication_context.py flow_index_vault.py tests/test_scan.py
git commit -m "feat: enrich indexed docs with communication context"
```

## Task 5: Build Runtime Provider From Scanned Communication Records

**Files:**
- Modify: `flow_index_vault.py`
- Modify: `communication_context.py`
- Test: `tests/test_scan.py`

- [ ] **Step 1: Run GitNexus impact analysis**

Run:

```bash
# GitNexus MCP: impact(target="index_vault_flow", direction="upstream")
```

Expected: central flow impact. If HIGH/CRITICAL, report before editing.

- [ ] **Step 2: Write failing runtime build test**

Add test with fake scanned records:

```python
def test_index_flow_builds_context_provider_from_scanned_records(monkeypatch, tmp_path):
    # Use a narrow test around a helper if possible:
    # build_context_provider_from_records(records, source_records_by_ns_doc_id, config)
    from communication_context import build_context_provider_from_records
    from sources.base import SourceRecord

    records = [
        {"doc_id": "comm::1", "source_name": "comm", "source_type": "pg_message", "rel_path": "zoho/1"},
        {"doc_id": "comm::2", "source_name": "comm", "source_type": "pg_message", "rel_path": "zoho/2"},
    ]
    source_records = {
        "comm::1": SourceRecord("1", "pg_message", "zoho/1", 1.0, 10, {
            "_text": "Unit E",
            "source": "zoho_cliq",
            "source_message_id": "s1",
            "message_id": "1",
            "channel_id": "chan",
            "sent_at": "2026-01-01T10:00:00Z",
            "sender": "A",
        }),
        "comm::2": SourceRecord("2", "pg_message", "zoho/2", 2.0, 10, {
            "_text": "Also this",
            "source": "zoho_cliq",
            "source_message_id": "s2",
            "message_id": "2",
            "channel_id": "chan",
            "sent_at": "2026-01-01T10:00:10Z",
            "sender": "A",
        }),
    }

    provider = build_context_provider_from_records(records, source_records, {})
    envelope = provider.get_context_envelope(
        communication_item_from_record(
            {"doc_id": "comm::2", "source_name": "comm", "source_type": "pg_message"},
            source_records["comm::2"].metadata,
        )
    )

    assert envelope.nearest_nonempty_before.text == "Unit E"
```

- [ ] **Step 3: Run test to verify it fails**

Run:

```bash
PYTHONPATH=. pytest tests/test_scan.py::test_index_flow_builds_context_provider_from_scanned_records -v
```

Expected: FAIL because helper does not exist.

- [ ] **Step 4: Implement provider builder**

In `communication_context.py`, add `build_context_provider_from_records()` that:
- Iterates scanned docs and source records.
- Uses `communication_item_from_record()` for identity.
- Converts record `_text` to `CommunicationMessage`.
- Groups by source/channel/thread.
- Respects config defaults from `communication_context`.

- [ ] **Step 5: Wire provider into `index_vault_flow`**

After `source_records_by_ns_doc_id` is populated:

```python
from communication_context import build_context_provider_from_records

_RUNTIME["communication_context_provider"] = build_context_provider_from_records(
    scanned,
    source_records_by_ns_doc_id,
    config.get("communication_context", {}),
)
```

If disabled in config, set provider to `None`.

- [ ] **Step 6: Run focused tests**

Run:

```bash
PYTHONPATH=. pytest tests/test_scan.py::test_index_flow_builds_context_provider_from_scanned_records tests/test_scan.py::test_process_doc_task_passes_context_to_enrichment -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add communication_context.py flow_index_vault.py tests/test_scan.py
git commit -m "feat: build communication context provider during indexing"
```

## Task 6: Add Config Defaults And Examples

**Files:**
- Modify: `core/config.py`
- Modify: `config.yaml.example`
- Modify: `config.local.yaml.example`
- Modify: `config.vps.yaml.example`
- Test: `tests/test_config.py` or `tests/test_config_sources.py`

- [ ] **Step 1: Run GitNexus impact analysis**

Run:

```bash
# GitNexus MCP: impact(target="load_config", direction="upstream")
```

Expected: config loading is shared. If HIGH/CRITICAL, report before editing.

- [ ] **Step 2: Write failing config default test**

Add to `tests/test_config_sources.py`:

```python
def test_communication_context_defaults(tmp_path):
    cfg_path = _write_config(tmp_path, {
        "documents_root": str(tmp_path / "vault"),
    })

    cfg = load_config(str(cfg_path))

    assert cfg["communication_context"]["enabled"] is True
    assert cfg["communication_context"]["window_before"] == 5
    assert cfg["communication_context"]["window_after"] == 5
    assert cfg["communication_context"]["same_channel_only"] is True
```

- [ ] **Step 3: Run test to verify it fails**

Run:

```bash
PYTHONPATH=. pytest tests/test_config_sources.py::test_communication_context_defaults -v
```

Expected: FAIL because defaults are missing.

- [ ] **Step 4: Add config validation/defaults**

In `load_config()`:

```python
comm_ctx_cfg = raw.get("communication_context")
if comm_ctx_cfg is None:
    comm_ctx_cfg = {}
elif not isinstance(comm_ctx_cfg, dict):
    raise ValueError("communication_context must be a mapping")

comm_ctx = {
    "enabled": True,
    "window_before": 5,
    "window_after": 5,
    "max_time_window_minutes": 15,
    "same_channel_only": True,
    "include_batch": True,
    **comm_ctx_cfg,
}
if not isinstance(comm_ctx["enabled"], bool):
    raise ValueError("communication_context.enabled must be a boolean")
for key in ("window_before", "window_after", "max_time_window_minutes"):
    if not isinstance(comm_ctx[key], int) or comm_ctx[key] < 0:
        raise ValueError(f"communication_context.{key} must be a non-negative integer")
if comm_ctx["same_channel_only"] is not True:
    raise ValueError("communication_context.same_channel_only must remain true")
raw["communication_context"] = comm_ctx
```

- [ ] **Step 5: Update example configs**

Add:

```yaml
communication_context:
  enabled: true
  window_before: 5
  window_after: 5
  max_time_window_minutes: 15
  same_channel_only: true
  include_batch: true
```

- [ ] **Step 6: Run config tests**

Run:

```bash
PYTHONPATH=. pytest tests/test_config.py tests/test_config_sources.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add core/config.py config.yaml.example config.local.yaml.example config.vps.yaml.example tests/test_config_sources.py
git commit -m "feat: add communication context configuration"
```

## Task 7: Preserve Search Contract And Add Metadata Filter Test

**Files:**
- Modify: `tests/test_mcp_contract.py` or `tests/test_store.py`
- No production code unless tests expose missing metadata passthrough.

- [ ] **Step 1: Run GitNexus impact analysis if production code changes are needed**

If only tests change, no symbol impact required. If `_compact_hit_to_dict` or `lancedb_store.py` changes, run:

```bash
# GitNexus MCP: impact(target="_compact_hit_to_dict", direction="upstream")
# GitNexus MCP: impact(target="_extract_extra_metadata", direction="upstream")
```

- [ ] **Step 2: Write metadata passthrough/filter test**

Add to `tests/test_store.py`:

```python
def test_context_metadata_filter_and_passthrough():
    vec = [0.1] * 768
    store = LanceDBStore(tempfile.mkdtemp(), "chunks")
    store.upsert_nodes([
        _make_node_with_meta(
            "photo.jpg",
            "img:c:0",
            "photo of garage",
            vec,
            source_type="img",
            enr_context_entities_places="54 S Broad Main Unit E",
            enr_context_confidence="high",
        )
    ])
    hits = store.vector_search(vec, top_k=5, where="lower(metadata.enr_context_confidence) = 'high'")

    assert hits
    assert hits[0].extra_metadata["enr_context_entities_places"] == "54 S Broad Main Unit E"
```

- [ ] **Step 3: Run test**

Run:

```bash
PYTHONPATH=. pytest tests/test_store.py::test_context_metadata_filter_and_passthrough -v
```

Expected: PASS if dynamic metadata works. If it fails because enrichment fields are core fields, adjust expected direct attributes or `_CORE_META_KEYS` deliberately.

- [ ] **Step 4: Commit**

Run:

```bash
git add tests/test_store.py lancedb_store.py core/storage.py mcp_server.py
git commit -m "test: cover communication context metadata search"
```

Only include production files if actually changed.

## Task 8: End-To-End Verification And Index Safety

**Files:**
- No new production files expected.

- [ ] **Step 1: Run GitNexus detect changes**

Run:

```bash
# GitNexus MCP: detect_changes(scope="all")
```

Expected: changed symbols match planned areas: `communication_context.py`, `doc_enrichment.py`, `flow_index_vault.py`, `core/config.py`, tests/config examples.

- [ ] **Step 2: Run focused test suite**

Run:

```bash
PYTHONPATH=. pytest \
  tests/test_communication_context.py \
  tests/test_doc_enrichment.py \
  tests/test_scan.py \
  tests/test_config.py \
  tests/test_config_sources.py \
  tests/test_store.py \
  tests/test_mcp_contract.py \
  -v
```

Expected: PASS.

- [ ] **Step 3: Run required existing health tests**

Run:

```bash
PYTHONPATH=. pytest tests/test_config.py tests/test_prefect_server.py -v
```

Expected: PASS.

- [ ] **Step 4: Re-run GitNexus analyze after final commit**

Check `.gitnexus/meta.json` first:

```bash
jq '.stats.embeddings // 0' .gitnexus/meta.json
npx gitnexus analyze
```

If embeddings count is nonzero, use:

```bash
npx gitnexus analyze --embeddings
```

- [ ] **Step 5: Final commit if any verification-only edits remain**

Run:

```bash
git status --short
git add <only feature files>
git commit -m "feat: add communication adjacency context enrichment"
```

Only commit if there are uncommitted feature changes.

## Implementation Notes

- Keep context enrichment enabled by default, but make it safe: same source and same channel/thread only.
- Nearby context must be labeled as candidate context in LLM prompts.
- Do not infer property/building identity with regex-specific rules.
- Do not add LanceGraph as a dependency in this implementation.
- Keep provider interface stable so LanceGraph can replace source-window traversal later.
- Existing dirty worktree files must not be reverted or included unless they are part of this feature.

