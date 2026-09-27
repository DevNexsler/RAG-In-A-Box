# Extended regression checks

Scope approved September 26, 2026: add RAG quality evaluations, property/mutation
checks, load/soak tests, and deployed-revision smoke checks. Existing agreed
boundaries remain indexing/queue/storage, failed-source retirement, enrichment,
Factbook identity, CDS delivery, and provider failures/health.

## Commands and evidence

Run commands from a checkout with Python dependencies installed. Each command
returns nonzero for a failed check. Override `QUALITY_RUN_DIR` to preserve reports
outside a worktree; default is `.evals/quality`.

| Command | Boundary and evidence |
| --- | --- |
| `make test-quality` | Real LanceDB/FTS and hybrid retrieval against `tests/fixtures/quality/rag-v1.json`; per-query recall, reciprocal rank, NDCG, source text fidelity, and exact enrichment repair judgments. Report includes corpus SHA-256. |
| `make test-properties` | Durable queue operation histories: 16 reproducible seeds × 160 enqueue/ACK/fail/reopen operations, table/source isolation, forced-work preservation, plus callback replay and legacy queue upgrade. Failure includes seed and history. |
| `make test-mutation` | Curated safety guards removed in disposable tracked-file copies; baseline must pass first. JSON distinguishes killed, survived, invalid, timeout, baseline failure, and test infrastructure error. JUnit/log artifacts identify failing assertions. |
| `make test-soak` | Ten rounds of 16 documents, four persistent concurrent writers; commit replay, fresh storage readers, local HTTP 503→accepted delivery. Measures write p95, resident-memory growth after first-round warmup, retained Lance manifests, disk bytes, and drained backlogs. |
| `make smoke-deployed EXPECTED_REVISION=<commit>` | Read-only Docker source-hash comparison plus `/health`, `/health/providers`, OOM/container state, pending SQLite queues, and terminal callbacks requiring redrive. `DEPLOYED_CONTAINER` defaults to `doc-organizer`. |

Small evaluator, property, smoke-assessment, and soak regressions participate in
normal unit/integration tiers (`make gate-fast`). Full curated mutation campaign
and the default 160-write soak are mandatory release tiers, after integration
and before staging/live. A failure stops later tiers. Use gate `--only mutation`
or `--only soak` for a standalone run with `result.json` and `report.md`.
Long-duration soak remains an explicit command; no nightly job is installed by
these targets. Stage and live tiers remain required before release.

A longer, bounded local soak:

```bash
.venv/bin/python scripts/pipeline_soak.py --rounds 100 --documents 16 --workers 4 \
  --deadline 1800 --max-p95-ms 5000 --max-rss-growth-mib 128 \
  --output /tmp/rag-soak-report.json
```

Soak owns a temporary index and loopback callback receiver. Parent process removes
its temporary index after reaping worker on completion, deadline, SIGINT, or SIGTERM. Callback retry timestamps advance between attempts so delay/backoff does not
dominate the test. Write p95 excludes callback transmission. No production index,
configured LLM, or external API participates. Budgets are explicit regression
limits, not production service-level objectives; review hardware and workload when
changing them. Storage-byte samples are observational; retained-version budget is
four manifests per logical write (each logical write includes two actual upserts).

## What these checks establish

Corpus documents and judgments are synthetic and reviewed in Git. Provider double
uses stable lexical feature hashes. This protects retrieval integration, filters,
provider-outage fallback, and deterministic postprocessing. It does **not** measure
semantic embedding quality, real model intent preservation, or generated-answer
faithfulness. Keep existing real-provider enrichment benchmarks/live tests for
those; enlarge versioned corpus with independently reviewed labels when adding
capabilities. Each case must meet its threshold; strong cases cannot average away
an identity/filter failure. Duplicate chunks consume retrieval slots.

Generated histories use a separate logical model and fixed seeds, with no added
runtime dependency. They do not shrink failing traces automatically. Fixed replay
regressions protect issues discovered by these histories. Mutations cover queue
revision/incarnation, callback incarnation, card grounding, failed-source
retirement, Factbook response correlation, exclusive provider probes, and the
PR-specific faults mapped in [PR_TEST_COVERAGE.md](PR_TEST_COVERAGE.md). This is
a targeted sensitivity gate, not a whole-repository mutation score.

Subsystem soak validates persistence and replay under concurrent local load. It
does not establish end-to-end HTTP throughput or shared GPU/provider capacity.
Existing hermetic staging E2E and real-provider live tiers cover those boundaries
functionally; a separate controlled production-like load run is still needed for
capacity planning. Real provider latency remains visible in live JUnit timings.

Deployment smoke checks source files inside an immutable container ID against an
explicit Git commit, then verifies the named container still identifies that ID.
This is filesystem evidence, not proof of Python bytecode already loaded in a
process. Use normal container replacement for deployment. Missing queue databases,
missing source, malformed/failed probes, or unknown Docker health fail closed.
Terminal callback backlog must be zero. Default pending budget is zero; direct CLI accepts `--max-pending` for a documented
operational allowance. Report omits source content, environment and credentials.
No restart, indexing request, or deployment occurs.

## Queue replay fixes discovered by new tests

SQLite may reuse deleted row IDs, and revision counters restart for newly enqueued
work. An old ACK could therefore delete a later request or callback. Both queues
now persist a random incarnation token and require it together with ID/revision
for state changes. Coalescing preserves incarnation; a fresh insertion gets a new
one. Additive schema upgrade assigns tokens to pending historical rows in an
immediate transaction. Public request/delivery snapshots carry the token.

Deploy by replacing all queue worker processes together. Older running binaries
ignore the new guard; rolling mixed-version workers cannot provide this guarantee.
Schema upgrade preserves pending contents, attempts, and revision numbers.

To evaluate the same reviewed judgments with configured real embeddings, run:

```bash
.venv/bin/python scripts/rag_quality.py --live-config config_test.yaml --allow-paid \
  --output /tmp/rag-quality-live.json
```

This mode requires the existing live preflight and explicit spend opt-in, uses the
same provider for document and query vectors, and records provider/model identity.
Normal cases must have healthy retrieval diagnostics; only the explicit outage
case may degrade to keyword fallback. The live tier also runs `test_rag_quality_live.py` against `config_test.yaml` after
its normal preflight. These are small relevance judgments, not a comprehensive
semantic benchmark or generated-answer evaluation.

## September 26 baseline finding

A 100-round run (1,600 logical writes, four persistent workers) completed every
write and callback retry without loss, with p95 write latency 2.20 seconds. RSS
grew 297.8 MiB, exceeding the unchanged 128 MiB growth budget. The same harness
against merged baseline `2ad899d` grew 299.4 MiB (p95 2.39 seconds), establishing
that this growth predates the incarnation fixes. These runs used the normal
512 MiB index / 128 MiB metadata cache caps; they do not prove an unbounded leak.

Forty-round diagnostic comparisons found reader reuse did not materially reduce
growth; smaller cache caps did. Production cache settings were not changed.
Treat long-soak RSS failure as a performance investigation trigger. Do not infer
production capacity from the small default run or relax its threshold merely to
get a passing report. Follow-up should establish memory plateau under production
cache/maintenance settings and representative concurrent provider load.
