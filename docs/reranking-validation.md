# qwen-bulk candidate qualification — 2026-10-31

The user accepted observed local latency and requested a thirty-second test
budget. Qwen now defaults to, and is capped at, **30 seconds for the entire
rerank operation**, including semaphore wait and all batches. Compose explicitly
overrides old mounted ten-second configs. A shorter configured budget remains
supported. Health probing stays capped at five seconds; explicit DeepInfra
retains its previous timeout behavior. These measurements precede merge and
deployment; final release status is recorded in the PR.

The real MCP streamable-HTTP test seeds a temporary LanceDB/FTS corpus, runs the
production `file_search` consumer and sends real gateway inference requests.
Retrieval vectors are deterministic so this test isolates reranking. Gateway
credentials come from the existing deployment environment in memory; none are
written to test configuration. Every observed request asserts `qwen-bulk`,
top-level priority 0, thinking disabled, zero retries, disabled fallbacks and a
positive remaining coordinator deadline. No paid provider is selected.

## Thirty-second factory-path live E2E

All four cases passed without an adapter monkeypatch. Six-check live preflight
passed before inference; the test client uses 45-second HTTP and MCP deadlines.
Every case preserved the paid-tenant answer document ID/snippet at rank one and
sent exactly one occurrence score per retrieved candidate.

| Corpus | MCP elapsed | Rerank elapsed | Result |
|---|---:|---:|---|
| 3 ordinary passages | 0.922s | 0.863s | Correct answer; healthy |
| 70 digit-dense passages | 10.026s | 9.958s | Correct answer; healthy |
| 80 digit-dense passages | 14.088s | 14.007s | Correct answer; healthy |
| 90 digit-dense passages | 9.680s | 9.613s | Correct answer; healthy |

## Repeated diagnostic measurements

Before changing the default, an isolated harness repeatedly called the same real
MCP consumer using the committed adapter. Its diagnostic profile overrode only
the adapter object's timeout to 30 seconds. Warmups are excluded. Requests were
sequential at the MCP level, with the normal two concurrent batch slots. The
benchmark stopped when the user accepted latency; it did not reach its planned
100 samples per dense profile.

| Profile | Samples | Successful reranks | MCP p50 | MCP p95 | MCP p99 |
|---|---:|---:|---:|---:|---:|
| 90 dense, 30s diagnostic | 59 | 59 | 9.654s | 17.022s | 20.573s |
| 90 dense, original 10s | 60 | 0 | 10.021s | 10.033s | 10.042s |
| 3 ordinary, 10s control | 12 | 12 | 1.042s | 2.648s | 2.648s |

These are nearest-rank empirical response percentiles on a fixed synthetic
corpus, not production SLAs. At 59 samples, p99 is simply the largest observed
value. Every ten-second dense sample was censored by timeout and returned honest
cosine degradation; those response percentiles are not inference completion
percentiles. Thirty-second dense rerank-only p95/p99 were 16.985s/20.556s.

Read-only coordinator inspection showed that a ten-second remaining deadline
reserves time across three local backends, leaving roughly 3.3 seconds for the
first attempt. A thirty-second deadline gives that attempt roughly ten seconds.
Thus changing the deadline can change local retry behavior as well as allowing
more total time. This does not prove which portion of latency is queue, prefill
or generation. Priority 0 survives caller-knob merging, including numeric zero;
no repeated packet capture or gateway modification was needed.

The implementation retains 32 passages per batch, two process-wide requests,
the digit-aware aggregate 64k limit and conservative 16k request envelopes.
Truncation may discard relevant evidence beyond passage heads. Further labeled
real-corpus quality and production-load latency evaluation remain future work.
Timeout, malformed output and cancellation regressions preserve atomic scoring,
honest degradation and cancellation behavior under the larger cap.

## Deterministic qualification

- Focused provider/search/deployment/health contracts: 77 passed.
- Static lint and collection passed (2,921 collected); unit suite: 2,443 passed.
- Final-image staging: 74 E2E passed, with 22/22 MCP tools covered/traced and
  attachment-path audit passed. Injected 32-second hangs degrade at the shared
  30-second cap, then recover through the same MCP consumer.
- Fast gate passed: 248 integration tests passed, 16 skipped because isolated
  PostgreSQL history test credentials were not configured.

The first full release run passed all six deterministic tiers, including all
264 integration cases with disposable PostgreSQL, 42/42 killed mutations,
160-write soak and 74 staging cases. Live preflight first refused an active
production indexer. Its guarded retry completed 137/140 live cases; it exposed
the synchronous scorer's nested-event-loop incompatibility and a remaining
hardcoded DeepInfra health test against the spent account.

The synchronous scorer now uses a separate event loop thread when called from
an active async consumer, with the same absolute deadline and inherited
cancellation context. Real-loopback/Lance regressions cover both caller modes;
the health test requires a valid score from the configured scorer. Final
release gates and post-deployment smoke are recorded in the PR; earlier failed
run artifacts remain available and are not relabeled as passes.

The worktree is registered and incomplete pending merge and verification.
Disposable tests use temporary indexes and isolated compose resources. The
interrupted benchmark's exact owned scratch was removed after verifying no
benchmark process remained; evidence and its diagnostic script were archived
outside the worktree. No Factbook database or source is touched.
