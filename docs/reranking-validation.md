# qwen-bulk candidate qualification — 2026-10-30

**Deployment blocked: large-pool live latency qualification fails.** This change
is a draft PR. No merge or deployment is authorized by this evidence.

The real MCP streamable-HTTP test seeds a temporary LanceDB/FTS corpus, runs the
production `file_search` consumer and sends real gateway inference requests.
Retrieval vectors are deterministic so this test isolates reranking. Gateway
credentials come from the existing deployment environment in memory; none are
written to test configuration. Every observed request asserts `qwen-bulk`,
top-level priority 0, thinking disabled, zero retries, disabled fallbacks and a
positive remaining coordinator deadline. No paid provider is selected.

Final real-gateway results:

| Corpus | MCP elapsed | Rerank elapsed | Result |
|---|---:|---:|---|
| 3 ordinary passages | 0.983s | 0.921s | Correct paid-tenant document ID/snippet; healthy |
| 70 digit-dense passages | 10.075s | 10.002s | Deadline; cosine fallback, degraded |
| 80 digit-dense passages | 10.070s | 10.002s | Deadline; cosine fallback, degraded |
| 90 digit-dense passages | 10.069s | 10.002s | Deadline; cosine fallback, degraded |

The large-pool tests intentionally require successful scoring and therefore
remain failing qualification tests (`1 passed, 3 failed`). They must pass before
release; a healthy cosine fallback is not successful reranker qualification.

Read-only coordinator logs showed local attempts timing out around 3.3s: the
coordinator reserves remaining time across its three local candidates. Smaller
prompt/batch trials (32/16/8 passages, lower byte budgets and bounded increased
concurrency) also missed the shared ten-second deadline. Those experiments do
not justify reducing evidence further or increasing load in the shipped default.
The candidate retains 32 passages, two process-wide concurrent requests, the
aggregate 64k limit and a stricter 16k request envelope. No gateway changes,
backend pinning, paid fallback or timeout increase are included.

Remaining work before release: achieve successful 70/80/90 digit-dense MCP runs
inside the existing ten-second rerank budget, assess relevance beyond passage
heads on a labeled real corpus, then run the full seven-tier release gate and
post-deployment smoke. Default provider selection changes on service recreation
because compose overrides legacy mounted DeepInfra configuration. Explicit paid
rollback remains an operator choice, described in [reranking.md](reranking.md).

Deterministic qualification completed:

- Static lint and collection: pass (2,916 collected).
- Final unit suite: 2,438 passed.
- Integration gate: 248 passed, 16 skipped (isolated PostgreSQL history cases).
- Final targeted provider/search/health contracts: 74 passed.
- Final-image staging gate: 74 passed, including all five Qwen failure/recovery
  cases; MCP tool coverage and attachment-path audits passed.

An earlier staging image built during the smaller-prompt experiment failed one
media-presence assertion. The committed 16k envelope passed the same 31-test
prefix; Qwen and simulated DeepInfra returned identical top-ten paths on its
same corpus. The full final-image staging gate then passed. No assertion was
weakened to obtain that result.

The complete release gate (mutation, soak and unrelated paid-provider live
tiers) was not run. These passing deterministic gates do not override the
failing large-pool live qualification above.

The implementation worktree is registered and remains incomplete pending merge
and verification. Disposable tests use temporary filesystem indexes and an
isolated compose project; no Factbook database or source is touched.
