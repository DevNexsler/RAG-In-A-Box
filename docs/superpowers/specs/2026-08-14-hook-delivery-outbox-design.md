# Hook Delivery Outbox Design

## Goal

Make `document.indexed` delivery durable, semantically acknowledged, safe to retry, and auditable without exposing hook secrets or document content.

## Root Cause

Current document processing writes Lance data, then synchronously invokes a best-effort HTTP hook. `hooks/http.py` treats every HTTP 2xx as success and discards its response body. `hooks/dispatcher.py` converts failures into warnings. Nothing durable owns delivery after the indexing task returns.

CDS legitimately returns HTTP 200 for non-success outcomes such as `no_match`, `ambiguous`, and `correlation_mismatch`; RAG therefore records none of those states and never retries them. This is a delivery-boundary defect, not an indexing defect. `/health` disk status does not gate hook dispatch, so disk pressure is a separate operational incident.

## Design

Add a SQLite-backed per-index-root hook delivery outbox. Every enabled matching hook receives one persisted delivery before any network call. A delivery contains only opaque operational metadata plus serialized event data: event ID, event name, document ID, hook name, payload, attempts, next-attempt timestamp, terminal state, and safe error/outcome fields. It never contains a secret. Logs and spans must never include payload text, chunks, or headers.

After persistence, pipeline attempts due deliveries immediately. Server scheduler drains due rows after restarts. Successful delivery removes its row only after both transport and configured semantic acceptance succeed. The CDS hook declares `accepted_statuses: [updated, duplicate]`; generic hooks without that setting retain ordinary 2xx acceptance.

Transient transport errors, 5xx responses, and unexpected status bodies remain durable and retry with bounded exponential delay. A deterministic, non-retryable semantic outcome becomes a visible terminal redrive state. A forced targeted re-index creates a fresh delivery after repair, providing the existing operational redrive path without a new MCP tool.

Each `document.indexed` event gets an immutable UUID `event_id`. Per-attempt telemetry contains only event ID, document ID, hook name, attempt number, transport/result class, HTTP status when present, and retry/terminal state. Existing tracing JSONL remains opt-in.

## Scope

- New focused outbox module. Do not bend `IndexRequestQueue` into a generic event queue; its schema and revision semantics model indexing work, not external delivery.
- Preserve indexing success when downstream hook unavailable.
- Add hook config validation/contract for `accepted_statuses`.
- Drain delivery outbox from existing in-process index scheduler.
- Add unit and integration tests for red-green regressions, persistence, semantic 2xx rejection, retry, and safe telemetry.
- Add operator notes: disk high-water needs capacity reclamation or explicit policy decision; code must not mask it by raising the threshold.

## Out of Scope

- Reclaiming production disk or changing its high-water policy.
- Modifying CDS response semantics or data schema.
- New MCP tools. Forced targeted indexing already provides a controlled redrive route.

## Security and Reliability Constraints

- Secrets remain environment-resolved at send time; never persist secret values.
- Payload content may exist only in index-root outbox storage, which uses existing index-root access controls; delete on accepted delivery.
- Retry capped at five automatic attempts using exponential delays of 1, 5, 30, 120, and 600 seconds. Then mark terminal `redrive_required`.
- `updated` and `duplicate` are accepted CDS outcomes. `no_match`, `ambiguous`, and `correlation_mismatch` are terminal redrive states; other unexpected semantic outcomes retry until cap.
- Default generic hooks accept any 2xx unless `accepted_statuses` explicitly narrows contract.

## Verification

Tests must demonstrate red then green for:

1. HTTP 200 plus CDS `no_match` is not accepted.
2. A timeout persists delivery and later retry succeeds.
3. Reopening outbox preserves a due delivery.
4. A terminal semantic state never leaks payload content or secret values through warnings/tracing.
5. Indexing remains successful while failed delivery stays durable.
6. Existing hook and targeted-index tests remain green.

Run focused tests during tasks, then `make gate-fast`. Baseline currently has unrelated reproducible Lance repair failure in `tests/test_store.py::test_store_open_repairs_fragment_that_overclaims_its_columns`; record separately unless this work changes it.
