# Note: switch comm-data-store message reads to canonical rows

**Date:** 2026-06-12
**From:** Comm-Data-Store migration 029 (`cross_delivery_origin_linking`)

## What changed upstream

Comm-Data-Store now links cross-delivery duplicates: when one platform event
fans out as multiple physical emails (e.g. TenantCloud sends the same tenant
reply to both the `Notification` and `LIVE-WATCH` mailboxes, each with its own
Message-ID), the extra copies are still stored but get
`messages.canonical_message_id` pointed at the earliest copy. A new view,
`messages_canonical`, returns exactly one row per real-world event
(`SELECT * FROM messages WHERE canonical_message_id IS NULL`).

Backfill already linked 80 historical duplicates (all TenantCloud
LIVE-WATCH/Notification pairs); new duplicates link automatically at ingest.

## What this repo should do

The `comm_messages` postgres source in `config.yaml` reads `FROM messages m`
directly, so it indexes every duplicate delivery as a separate document, and
each one gets its own LLM digest pass downstream.

One-line fix when convenient — in the `pg_message` table spec, either:

- `FROM messages_canonical m` (drop-in; the view has all columns), or
- keep `FROM messages m` and add `AND m.canonical_message_id IS NULL`
  to the WHERE clause.

## Cleanup (optional)

Documents already indexed from duplicate rows will not disappear on their own
after the query change — they just stop being re-indexed. To purge them, the
duplicate ids are queryable upstream:

```sql
SELECT source, source_message_id
FROM messages
WHERE canonical_message_id IS NOT NULL;
```

(`id_template` for pg_message is `{source}/{source_message_id}`.)

This note can be deleted once the query is updated.
