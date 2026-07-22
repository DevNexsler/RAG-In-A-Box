# Attachment Conversation Context Implementation Plan

Date: 2026-07-22

1. Add failing provider tests for nearest nonempty before and after within absolute time bound; confirm optional candidates may still be pruned.
2. Add failing sidecar parser tests for explicit nearest fields and legacy fallback.
3. Run GitNexus impact analysis for every existing symbol to modify. Stop and warn before HIGH/CRITICAL edits.
4. Implement bounded symmetric selection and sidecar round-trip. Run focused context tests.
5. Add failing tests for idempotent search-text replacement and context-only Lance refresh.
6. Implement reusable refresh service plus bounded CLI. Reuse stored sidecar text and embedding provider; never call media extraction.
7. Change repaired-sidecar sweep handling so existing indexed attachments use refresh service while new/degraded items retain full indexing.
8. Run focused unit/integration suites, then `make gate-fast`.
9. Run GitNexus change detection and inspect every affected flow before commit.
10. Check production writer state. Avoid concurrent full index writer during deployment/backfill.
11. Deploy verified code. Dry-run production batch, refresh 5 attachments, run live queries for terms from before and after messages.
12. If results correct, repeat at 25, 100, then bounded remaining batches. Stop expansion on any verification failure.
