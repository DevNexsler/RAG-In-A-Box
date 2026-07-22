# Attachment Conversation Context Refresh

Date: 2026-07-22

## Goal

Make attachment retrieval use nearby message context already stored in communication sidecars. Guarantee bounded context from both sides when available. Repair existing indexed attachments without repeating media extraction, OCR, transcription, or description calls.

## Context selection

- Scope candidates to same communication source/channel/thread rules already enforced by provider.
- Keep attachment message text as primary context.
- Select nearest nonempty message before and nearest nonempty message after attachment when each falls within `max_time_window_minutes` (default 15).
- Treat selected nearest messages as guaranteed. Adaptive time-distance pruning may remove optional window messages, but never qualifying guaranteed messages.
- Reject candidates outside absolute time window. This prevents unrelated context bleed such as 45-minute-later messages.
- Preserve chronological ordering and deduplicate source IDs.
- Parse explicit `nearest_nonempty_before` and `nearest_nonempty_after` sidecar fields. Fall back to window lists only for old sidecars.

## Search text

- Represent conversation context as one standardized `[Conversation context]` suffix.
- Refresh operation removes existing standardized suffix before appending current context. Repeated runs remain idempotent.
- Update one anchor chunk per attachment, normally final chunk. Anchor contains existing chunk text plus current context suffix.
- Regenerate only anchor embedding. Preserve all other chunks, metadata, media-derived text, and vectors.
- Empty context removes stale standardized suffix and re-embeds anchor.

## Backfill

- Command supports dry-run, source-type filter, explicit document IDs, and row limit.
- Read existing Lance rows and stored sidecars. Do not open or process original media.
- Skip documents whose anchor text already matches desired text.
- Commit updates per document. Report scanned, eligible, changed, skipped, and failed counts.
- Start production backfill with 5 attachments. Run live context query checks. Expand only after correct target appears. Suggested progression: 5, 25, 100, then remaining corpus in bounded batches.

## Ongoing indexing

- Normal attachment indexing embeds context during initial indexing.
- Full communication sweep repairs sidecars first.
- Existing indexed attachments whose sidecars changed take context-only refresh path.
- New, missing, or degraded attachments retain full processing path.
- Future-message arrival remains eventually consistent through sweep repair; no cross-service message-arrival callback exists in this repository.

## Verification

- Unit tests prove bounded before/after guarantee and explicit nearest-field round-trip.
- Unit/integration tests prove context refresh changes searchable text and embedding without invoking media extraction.
- Focused retrieval tests prove before and after terms find intended attachment.
- Run `make gate-fast` before deployment.
- Production rollout records representative before/after query results after each batch.

## Failure handling

- Stop expansion on embedding error, row-update error, wrong-document retrieval, or lost existing text.
- Re-running same batch is safe because desired anchor text is deterministic.
- Existing vectors remain intact for documents not successfully updated.
