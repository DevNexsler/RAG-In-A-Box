# Staging Gate, E2E Suite, and Traceability Stack — Design

**Date:** 2026-07-04
**Status:** Approved design, pending implementation plan

## Goal

A release gate with a defined guarantee: **if `make gate` passes, the system works in production.** Concretely:

1. A disposable **staging environment** (docker-compose) running the exact production image, isolated and memory-capped so it can never OOM the host.
2. An **e2e suite** that drives the staging container from the outside — all 20 MCP tools over real MCP protocol, all REST endpoints, all extractor pipelines (text, PDF, image/OCR, audio, video), hook delivery, and fault recovery.
3. A **strictly ordered, fail-fast tier pipeline** where cheap deterministic tiers act as spend-gates for the live tier: fake providers first to rule out code issues, then real API calls (full workload, including Mac Mini OCR and comm-store Postgres) only on code that already passes.
4. **Traceability**, two meanings, both in scope:
   - Runtime: OpenTelemetry SDK spans (no backend/UI) following each document through scan → extract/OCR → enrich → embed → store → search, so a failed run names the document, stage, and provider call.
   - Coverage: an auto-collected tool-coverage matrix; the gate fails if any MCP tool has zero e2e coverage, making "full pass" a defined claim.

## Current state (verified 2026-07-04)

- ~69 test files: 56 unit, 9 `.int.test.py` integration, 1 mocked e2e (`test_production_readiness.e2e.test.py`), 3 `_live` tests requiring real keys. Only pytest marker: `live`. Tier selection is filename convention only; no runner script, no Makefile, no CI.
- `docker-compose.yml` defines a single production service (`doc-organizer`), no memory limits, no staging/test profile. Running the full suite inside the prod container has previously OOM-restarted the host.
- `test_mcp_contract.py` comprehensively validates MCP schemas/aliases/response shapes (in-process, no container).
- Traceability today: only `LLMTraceRecorder` (`providers/llm/trace_recorder.py`) — JSONL request/response capture for LLM enrichment calls to `.evals/llm-traces/`. No correlation IDs, no OTEL, no structured logging.
- External deps: OpenRouter (embeddings, LLM, audio/video transcription), DeepInfra (rerank), DeepSeek OCR2 on the Mac Mini (`http://192.168.68.70:8790`), Ollama vision, comm-store PostgreSQL (`COMM_DATA_STORE_DSN`), HTTP event hooks.

## Design

### 1. Staging environment — `docker-compose.staging.yml`

Three services on an isolated network, distinct ports from prod, torn down with `down -v` after every run (enforced by trap in the gate runner):

| Service | Details |
|---------|---------|
| `doc-organizer-staging` | Built from the **same Dockerfile as production**. Hard memory limit **4GB**. Throwaway named volumes for `/data` (documents, index, traces). `config.staging.yaml` mounted read-only; all provider base URLs point at `provider-sim`. `tracing.enabled: true`. |
| `provider-sim` | One small FastAPI container emulating the HTTP dialects the production code already speaks: OpenRouter chat completions + embeddings + audio/video transcription, DeepInfra rerank, DeepSeek OCR2, Ollama (embeddings + vision). Deterministic responses: embedding vectors derived from content hash (stable search ranking across runs); canned transcripts/OCR text per fixture. Fault injection via `X-Sim-Fault: 429 \| timeout \| garbage` request header. Also hosts a **webhook sink** (records received `document.indexed` POSTs for assertion). |
| `comm-postgres` | `postgres:16-alpine`, seeded at startup with the comm-store schema and fixture messages, so `sor_query`/`sor_schema` run against real SQL in staging. |

`config.staging.yaml` is committed (no secrets — the simulator accepts any API key).

### 2. The gate — `make gate`

Strictly ordered; each tier only runs if the previous passed; the run aborts on first tier failure.

| # | Tier | Marker | What runs | Cost |
|---|------|--------|-----------|------|
| 1 | static | — | `ruff check` + `pytest --collect-only` sanity | free, seconds |
| 2 | unit | `unit` | existing unit files, no network | free, ~1 min |
| 3 | integration | `integration` | existing `.int.test` files | free |
| 4 | staging-e2e | `e2e` | compose up staging stack → e2e suite from host → collect trace artifacts → compose down -v | free, deterministic |
| 5 | live | `live` | preflight, then live suite with real providers, full workload | 💰 real money |

- **Markers**: `unit`, `integration`, `e2e`, `live` registered in `pyproject.toml`. Existing filename conventions stay; a conftest guard fails collection if a file's marker disagrees with its filename convention (e.g. `.int.test.py` without `integration`), so the convention is enforced rather than drifting.
- **`make gate-fast`**: tiers 1–3 only, for inner-loop development.
- **Live preflight** (tier 5 entry check, hard-fails the gate rather than skipping): required API keys present (`OPENROUTER_API_KEY`, `DEEPINFRA_API_KEY`); Mac Mini DeepSeek OCR endpoint reachable **and not mid-indexing** (checked via the indexer heartbeat); comm-store Postgres reachable via `COMM_DATA_STORE_DSN`. No silent skips: a missing dependency is a gate failure with a named cause.
- **Live tier scope** (full workload, per decision): real OpenRouter embeddings/LLM/transcription, real DeepInfra rerank, a real (small) OCR job on the Mac Mini, real read-only comm-store queries. Fixture set kept deliberately small to bound spend; the gate report shows call/token counts per provider.
- Runner is a shell/Python script invoked by `make`; structured so dropping it into GitHub Actions later is configuration, not rework. CI itself is out of scope.

### 3. E2E suite — `tests/e2e/`

Drives the staging container exactly like a production client — nothing in-process:

- **MCP client over HTTP** for all 20 tools; plain `httpx` for REST endpoints (`/api/index/document`, health, etc.).
- **Scenario flow**: seed fixture documents → trigger indexing → poll until indexed → exercise the full tool surface → assert on results and on trace artifacts.
- **Fixtures** (`tests/fixtures/e2e/`): plain text, markdown, a small PDF, an image (OCR path), a short real audio clip (~5–10 s), and a short real video clip (~5 s). Media files are real recordings ("live data as needed"), small enough to commit.
- **Tool coverage**: search & retrieval (`file_search` incl. filters/sort=recent/return modes, `file_get_chunk`, `file_get_doc_chunks`, `file_recent`), browsing (`file_list_documents`, `file_status`, `file_facets`, `file_folders`), audit (`file_audit_log`), indexing control (`file_index_document`, `file_index_update`), all 7 taxonomy tools, and `sor_query`/`sor_schema` against `comm-postgres`.
- **Hook delivery**: assert the `document.indexed` webhook arrived at the simulator's sink with correct payload.
- **Fault scenarios**: simulator returns 429/timeout on provider calls → assert `core/resilience.py` retry/backoff recovers and the document still indexes; garbage response → assert degraded-path behavior (skip ledger / degraded ledger entries) rather than crash.

**Tool-coverage matrix**: the e2e MCP client is a thin wrapper that records every tool invocation per test. After the e2e tier, the gate compares the invoked set against the tool list discovered from the running server (not a hardcoded list — new tools are automatically required to gain coverage). **Any tool with zero coverage fails the gate.** The matrix (tool → covering tests) is written into the gate report.

The live tier reuses the same scenario suite where it can (same fixtures, real providers, plus the existing `_live` tests), so fake-vs-live behavioral differences surface as diffable results.

### 4. Traceability — OpenTelemetry SDK, no backend

- New `core/tracing.py`: OTEL SDK setup + a small custom `SpanExporter` (~30 lines) writing spans as JSONL to `/data/traces/` (host-mounted during gate runs). **Off by default**; enabled via `tracing.enabled: true` in config (staging/test configs enable it; prod can opt in later).
- `opentelemetry-instrumentation-httpx`: every outbound provider HTTP call auto-tagged with URL, status, latency — no code changes at call sites.
- Manual spans on the six pipeline stages: **scan, extract/OCR, enrich, embed, store, search**, attributed with `doc_id`, `rel_path`, `source`, chunk counts. Async context propagation (parent/child nesting) is automatic via OTEL contextvars.
- `LLMTraceRecorder` is kept unchanged in behavior (full request/response payloads are too heavy for span attributes) but its JSONL records gain `trace_id`/`span_id` fields so payloads join to spans.
- No collector, no Jaeger, no new services. If a UI is ever wanted, it's an exporter/config change.
- Per CLAUDE.md, `gitnexus_impact` must be run on each pipeline-stage function before instrumenting it.

### 5. Gate report

`scripts/gate_report.py` reads pytest results (junit XML per tier), span JSONL, `LLMTraceRecorder` output, and the coverage matrix → writes `.evals/gate-runs/<timestamp>/report.md`:

- Tier-by-tier pass/fail with durations.
- Tool-coverage matrix (N/N tools, per-tool covering tests).
- Per-document stage timelines from spans — a failed e2e run names the document, the stage, and the failing provider call.
- Live-tier provider call/token counts (spend visibility).

## Error handling

- Staging teardown (`compose down -v`) runs unconditionally via trap, including on Ctrl-C and tier failure.
- The 4GB memory limit on the staging container bounds worst-case host impact; the suite never runs inside the prod container.
- Live preflight failures and provider errors fail the gate with a named cause; nothing is silently skipped.
- Simulator write/latency behavior is deterministic; any flaky e2e test is a bug by definition, not a retry candidate.

## Testing the test infrastructure

- `provider-sim` gets its own contract tests: response shapes validated against recorded real-provider responses (seeded from existing `.evals/llm-traces/` captures where available), so the fakes can't silently drift from reality.
- The file span exporter and the conftest marker guard get unit tests.

## Out of scope (v1)

- CI (GitHub Actions) — the gate is structured to drop in later.
- OTEL collector / Jaeger / any tracing UI.
- structlog / logging migration.
- Prefect flow coverage.
- Load/performance testing beyond the durations already captured in the report.

## Decisions log

| Decision | Choice |
|----------|--------|
| Traceability meaning | Both runtime tracing and coverage matrix |
| Provider strategy | Hybrid, strictly staged: fakes first as spend-gate, live tier always runs after |
| Live tier scope | Full workload — real Mac Mini OCR and real comm-store Postgres included; hard preflight, no silent skips |
| Gate runner | Local `make gate` first; CI later |
| Tracing implementation | OTEL SDK + file exporter, no backend (upgrade path preserved) |
| Media pipelines | In scope — real small audio/video fixtures; simulated transcription in e2e, real transcription in live tier |
