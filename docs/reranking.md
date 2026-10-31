# Local search reranking

Large-pool live qualification currently blocks deployment. See
[candidate validation evidence](reranking-validation.md).

`search.reranker.provider` defaults to `qwen-bulk`, the existing local LiteLLM
alias. Requests use top-level `priority: 0`, `temperature: 0`,
`chat_template_kwargs: {enable_thinking: false}`, `num_retries: 0` and
`disable_fallbacks: true`. The adapter never selects `qwen-fast` or another
provider. LiteLLM must retain its local-only definition of `qwen-bulk`.

Set `LITELLM_API_KEY` (preferred) or `LITELLM_MASTER_KEY` in the service
environment. `LITELLM_BASE_URL` overrides `search.reranker.base_url`; default is
`http://host.docker.internal:4000/v1`. Credentials are never stored in prompts.
Compose forwards these settings to the app; indexing workers inherit its
environment. `file_status` probes the selected adapter with the same credentials
and validates its output, with a five-second budget.

Compose defaults `DOC_ORGANIZER_RERANKER_PROVIDER=qwen-bulk`. This deliberately
overrides old mounted `config.yaml` files that still select DeepInfra. It also
ignores their legacy reranker model field: this adapter's alias is fixed.
Outside compose, an explicit provider in YAML still applies. For a deliberate
paid rollback, set `DOC_ORGANIZER_RERANKER_PROVIDER=deepinfra`, configure the
legacy model/base URL and `DEEPINFRA_API_KEY`, and restart the service. There is
no automatic paid fallback. Unknown providers fail configuration explicitly.

Each operation retains the existing digit-aware aggregate 64k estimate and
4,000-character passage heads. Query heads are capped at 2,000 characters.
Batches contain at most 32 occurrences; passage heads are further shortened
uniformly until serialized UTF-8 request bytes plus completion allowance and
512 framing tokens fit 16k. Counting bytes conservatively covers digit-dense,
Unicode and escaped text, plus the fixed-ID schema. Relevant evidence beyond a
head can be lost; tune only with retrieval-quality and latency evidence.
The 64k pool envelope also includes those chat/schema/output allowances across
all batches; each batch receives at most 64k divided by the number of batches.

Two requests per process may run concurrently, shared across fresh search
adapters. One ten-second maximum covers preprocessing, semaphore wait and all
batches. Each gateway request receives the remaining budget in
`metadata.inference_deadline_seconds`; this prevents the coordinator's long
background default from applying. Requests are cancelled and awaited on a
failed batch, timeout or cancellation. MCP dispatch signals cancellation into
the synchronous search worker. Cancellation propagates without a degraded
success response.

Strict JSON maps attach scores to unique occurrence IDs, not array positions,
document IDs or text. Missing, unknown and duplicate IDs, invalid scores and
truncated generations reject the entire operation before mutating any hit.
Original document IDs and duplicate occurrences survive; ties remain stable.
Successful scoring retains the existing normalized 60/40 relevance/retrieval
blend. Failure retains cosine fallback and
`diagnostics.reranker_applied=false`, `diagnostics.degraded=true`.
Endpoint circuit state is shared with other callers of the same gateway URL.
Only a fully validated operation recovers it; malformed HTTP 200 cannot.

Deterministic regressions:

```sh
python -m pytest tests/test_qwen_bulk_reranker.py tests/contracts/test_qwen_search.int.test.py -q
make gate-fast
```

Opt-in live MCP/real-gateway regression (only reranking uses a live provider;
retrieval uses an isolated temporary LanceDB corpus and fixed vectors):

```sh
RUN_QWEN_RERANK_LIVE=1 python -m pytest tests/test_qwen_bulk_live.py -q -s
```

Provide gateway credentials only through environment. The test creates no
database service or Docker volume; its loopback MCP server, store and temporary
indexes use test-owned paths. Pytest can retain diagnostic files; remove owned
scratch paths after inspecting failures. Full release gate and post-deploy smoke
remain required before deployment; a focused live run does not qualify release.
