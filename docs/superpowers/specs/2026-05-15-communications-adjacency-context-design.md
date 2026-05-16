# Communications Adjacency Context Enrichment Design

Date: 2026-05-15
Status: Draft for review

## Summary

Add index-time communication context enrichment for messages and attachments.

Communication artifacts often need nearby same-channel context to be understood. A photo, video, or short text message may be ambiguous alone, while the immediately preceding or following messages explain the building, unit, claim, repair, sender intent, or upload batch. The system should enrich those artifacts with nearby context by default, while making it clear that nearby context is candidate context, not guaranteed truth.

The first backend will build adjacency context directly from source data. The design keeps this behind a provider interface so LanceGraph can replace the traversal backend later without reworking enrichment, metadata, or search contracts.

## Goals

- Enrich communication-derived messages and attachments with bounded same-channel context at indexing time.
- Keep retrieval lean by storing useful context-derived metadata on each indexed record.
- Avoid hardcoded property, building, person, or claim logic.
- Prevent blind context contamination by asking the LLM to judge whether nearby messages apply.
- Preserve separate atomic and context-derived metadata so provenance remains auditable.
- Keep the design compatible with a future LanceGraph backend.

## Non-Goals

- Add a production graph database dependency now.
- Wait for LanceGraph before fixing attachment provenance.
- Treat all nearby messages as authoritative context.
- Merge atomic attachment content and neighboring message context into one indistinguishable text blob.
- Build legal/evidence workflow UI in this phase.

## Core Concept

Each indexed communication artifact gets a context envelope.

The envelope contains:
- `primary_item`: the message or attachment being indexed.
- `same_channel_before`: bounded previous messages from the same source/channel/thread.
- `same_channel_after`: bounded next messages from the same source/channel/thread.
- `nearest_nonempty_before`: closest previous non-empty text message.
- `nearest_nonempty_after`: closest following non-empty text message.
- `same_batch`: sibling attachments/messages from the same upload burst when available.
- `source_scope`: source, channel/thread, sender, timestamp, and batch identifiers used to constrain context.

The envelope is source-agnostic. Zoho Cliq, SMS, email, Slack, and future sources can all produce the same shape.

## Architecture

### CommunicationContextProvider

Add a provider interface:

```python
class CommunicationContextProvider:
    def get_context_envelope(self, item: CommunicationItem) -> ContextEnvelope:
        ...
```

The indexer asks this provider for context before enrichment.

Initial backend:
- `SourceWindowContextProvider`
- Uses source data and sidecars to gather same-channel/thread before/after context.
- Does not require a graph database.

Future backend:
- `LanceGraphContextProvider`
- Uses LanceGraph/Cypher over Lance-backed node and edge tables.
- Returns the same `ContextEnvelope` shape.

The enrichment pipeline should depend only on `CommunicationContextProvider`, not on how context is found.

### Graph-Shaped Model

Even before using LanceGraph, model identities as if they were graph nodes and edges.

Node types:
- `Message`
- `Attachment`
- `Channel`
- `Thread`
- `Sender`
- `Batch`

Edge types:
- `HAS_ATTACHMENT`
- `SOURCE_MESSAGE`
- `IN_CHANNEL`
- `IN_THREAD`
- `SENT_BY`
- `NEXT_IN_CHANNEL`
- `PREVIOUS_IN_CHANNEL`
- `IN_UPLOAD_BATCH`

The source-window backend can compute these relationships procedurally now. LanceGraph can persist/query them later.

## Context Boundaries

Context must be scoped by source and channel/thread.

Required filters:
- Same source, such as `zoho_cliq`, `sms`, `email`, or future source value.
- Same channel, conversation, thread, or email thread identifier when available.
- Same tenant/account/source namespace when available.

Default bounds:
- Previous messages: 5
- Following messages: 5
- Time window: configurable, default 15 minutes for chat/SMS-like sources.
- Batch window: configurable, default same source/channel plus same timestamp burst or source batch key.

The provider must never pull messages from a different channel just because timestamps are close.

## LLM Enrichment Contract

The enrichment prompt must label context explicitly.

Prompt structure:

```text
PRIMARY ITEM
This is the item being indexed. Its content is authoritative for itself.

NEARBY SAME-CHANNEL CONTEXT CANDIDATES
These messages are nearby in the same channel/thread. They may describe the primary item, but they may also belong to another conversation topic. Use them only if they appear relevant.

TASK
Extract atomic metadata from the primary item.
Extract context-derived metadata only when nearby messages appear relevant.
If nearby context is ambiguous, conflicting, or unrelated, say so.
```

The LLM output should include separate fields:
- `atomic_entities_people`
- `atomic_entities_places`
- `atomic_entities_orgs`
- `atomic_entities_dates`
- `atomic_topics`
- `context_entities_people`
- `context_entities_places`
- `context_entities_orgs`
- `context_entities_dates`
- `context_topics`
- `context_key_facts`
- `context_relationship`
- `context_confidence`
- `context_source_message_ids`
- `context_warning`

Recommended `context_relationship` values:
- `direct`
- `batch_label`
- `nearby_relevant`
- `nearby_ambiguous`
- `conflicting`
- `none`

Recommended `context_confidence` values:
- `high`
- `medium`
- `low`
- `ambiguous`

## Indexed Metadata

Store context metadata as normal LanceDB metadata fields so search can filter and rank without retrieval-time expansion.

Core provenance fields:
- `origin_source`
- `source_message_id`
- `message_id`
- `channel_id`
- `channel_name`
- `thread_id`
- `sender`
- `sent_at`
- `batch_key`
- `attachment_index`
- `sidecar_path`

Context fields:
- `context_relationship`
- `context_confidence`
- `context_source_message_ids`
- `context_entities_people`
- `context_entities_places`
- `context_entities_orgs`
- `context_entities_dates`
- `context_topics`
- `context_key_facts`
- `context_warning`

Atomic and context-derived fields must remain separate. Search can expose both. Evidence workflows can require non-empty context fields or a minimum confidence.

## Data Flow

For each indexed source record:
1. Source emits or discovers communication identity metadata.
2. Indexer builds `CommunicationItem`.
3. `CommunicationContextProvider` returns a bounded same-channel context envelope.
4. Extractor produces primary item text, such as OCR, image description, video notes, or message body.
5. Enrichment runs with primary item plus labeled context candidates.
6. Enrichment output is split into atomic and context-derived metadata.
7. LanceDB stores primary chunks, embeddings, and metadata.
8. Search filters can use context metadata directly.

## Handling Attachments

Attachment sidecars should be parsed when present.

For sidecar-backed attachments, derive:
- source
- source message ID
- local message ID
- sender
- sent timestamp
- channel/source channel ID
- media index
- original filename
- batch key

Batch key should be source-agnostic:

```text
<origin_source>:<channel_or_thread_id>:<rounded_sent_at_or_source_batch_id>
```

If a source has an explicit batch/upload ID, prefer it over time rounding.

## Handling Email

Email should use thread-aware context rather than plain channel adjacency.

Recommended scope:
- same mailbox/source
- same normalized thread ID or message thread headers
- previous/following messages by sent timestamp within that thread
- attachments linked to the source email

Email should not use unrelated mailbox messages near the same timestamp.

## Error Handling

If context lookup fails:
- Continue indexing primary item.
- Set `context_relationship = "none"`.
- Set `context_confidence = "low"`.
- Add warning to index metadata.

If context exists but appears conflicting:
- Keep the primary item indexed.
- Set `context_relationship = "conflicting"`.
- Set `context_confidence = "ambiguous"`.
- Store `context_warning`.

If sidecar metadata is missing or malformed:
- Fall back to filesystem metadata.
- Do not infer channel/thread from path unless source adapter explicitly supports it.

## Migration Path To LanceGraph

This design does not block LanceGraph adoption.

Stable contract:
- `CommunicationContextProvider`
- `ContextEnvelope`
- indexed metadata fields
- enrichment prompt semantics

Replaceable part:
- how context candidates are discovered.

When LanceGraph is production-ready:
1. Store message, attachment, channel, sender, thread, and batch tables as Lance-backed graph nodes/edges.
2. Implement `LanceGraphContextProvider`.
3. Use Cypher traversal to produce the same `ContextEnvelope`.
4. Keep enrichment and LanceDB search metadata unchanged.

No search API or enrichment output rewrite should be required if the provider contract stays stable.

## Testing Strategy

Unit tests:
- Provider never crosses channel/thread boundaries.
- Provider returns previous/next messages from the same channel only.
- Provider includes same-batch attachments.
- Provider handles empty-body media messages.
- Provider degrades cleanly when sidecar is missing.

Enrichment tests:
- Nearby message naming a building creates context-derived place metadata.
- Unrelated nearby message results in `context_relationship = "none"` or `nearby_ambiguous`.
- Conflicting nearby message produces `conflicting` or `nearby_ambiguous`, not a direct attribution.
- Atomic and context-derived metadata stay separate.

Search tests:
- Context-derived building/unit fields are filterable through metadata filters.
- Visual-only media hits can still appear, but do not look like direct evidence without context confidence.
- Same-channel unrelated media is not promoted as high-confidence evidence.

Acceptance cases:
- A photo batch with a nearby same-channel label like "this is for Unit E" receives context metadata for that unit.
- A broad multi-property channel list does not automatically assign every nearby attachment to each property in the list.
- A media item adjacent to a different building label is not attributed to the queried building.

## Recommended First Implementation

Implement only the source-window provider first.

Why:
- Fixes current bug.
- Adds no unstable graph dependency.
- Preserves future LanceGraph path.
- Keeps retrieval lean through index-time metadata.

Avoid:
- Adding LanceGraph as required dependency now.
- Combining context and primary item into an unlabeled text blob.
- Property-specific rules.
- Search-time-only context expansion as the main fix.
