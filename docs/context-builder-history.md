# Exact conversation history in context_builder

Doc Organizer owns reusable conversation retrieval. The SOR packet builder owns
ticket identity recovery, selection of the evidence window, and mutation gates.
Semantic hits and a recent activity count cannot substitute for conversation text.

## Request

Existing arguments remain supported. Three optional arguments are additive:

- `history_since`: timezone-aware ISO-8601 lower bound from source evidence.
  Omit to retrieve newest messages regardless of age. Future bounds are rejected.
- `history_limit`: 1–100 messages per page; default 50.
- `history_cursor`: opaque `cds.conversation.next_cursor`. Keep the same normalized
  email/phone and `history_since` on subsequent calls. Page size may change.

Example: `context_builder(phone="2025550123", include=["cds"], history_limit=50)`.
Phone normalization remains the existing North American normalization contract.
Names and lead IDs alone do not establish conversation identity.

## Evidence and coverage

`cds.conversation.messages` contains exact-identifier messages, newest first by
`sent_at`, then ID. Both inbound and outbound messages are included. Matching uses
CDS participant links plus verified Quo and Zoho raw-recipient lanes for outbound
messages whose recipient link is missing. Raw payloads are not returned. Names,
shared channels and inferred aliases never broaden the match.

Each message carries CDS ID, provider source/message ID, timestamp, direction,
stored sender name, subject and body. Missing sender names stay missing.
Bodies are capped at 4,000 characters; clipping is explicit per message and page.

- `has_more` and `next_cursor` expose pagination rather than silently dropping rows.
- `through` fixes an event-time upper bound across pages; this is not a database
  snapshot and does not guarantee coverage of later-ingested historical data.
- `window_exhausted` means this page ends the requested identity/time window.
- `coverage_complete` is true only for an unclipped, exhausted first page.
  Continuation pages never certify earlier pages. Clients must accumulate every
  page and inspect every truncation flag before treating a paginated window as read.
- Errors and missing identifiers cannot claim coverage. Skipped sources can omit
  `conversation`; consumers must treat missing coverage as unknown, not complete.

Coverage applies only to stored CDS messages matching the supplied identifiers,
not to other aliases, un-ingested communications, or an entire business matter.
If bodies are clipped, obtain original source text before relying on omitted text.
An empty or fully read window is not proof a task is complete or superseded.

`inbound_count_30d` retains its 30-day meaning and deduplicates participant matches.
`latest_inbound_at` now searches all recorded time, excluding future messages.
Derived outbound timing uses the newer valid caller/CDS inbound timestamp.

## Rollout

This change does not migrate SOR callers or deploy the service. Retain the existing
SOR history fallback until this API is merged, deployed, and tested through the
live consumer. Then switch SOR retrieval to this contract while retaining its
identity recovery and evidence-completeness mutation gates. Do not maintain two
permanent conversation builders.

## Verification

The regression reproduces a question and withdrawal older than 30 days. PostgreSQL
tests also cover outbound raw recipients, shared-channel exclusion, duplicate
participants, timestamp-tie pagination, malformed recipient JSON, clipping and
bound SQL parameters. All fixtures are synthetic.

Set `DOC_HISTORY_TEST_DSN` to an isolated PostgreSQL database named
`doc_history_test` and run `pytest -q tests/test_cds_history.int.test.py`.
Tests create temporary tables only; without this variable they skip explicitly.
Run `make gate-fast` with the same variable to include them in the repository gate.
