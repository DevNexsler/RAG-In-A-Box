# Testing — the staging gate

Operator's manual for the five-tier test gate. The short version: run
`make gate-fast` while developing, `make gate` before anything ships.

## The guarantee

A full `make gate` pass means, end to end and in this order:

| # | Tier | What it proves | Cost |
|---|------|----------------|------|
| 1 | `static` | `ruff check .` clean and `pytest --collect-only` succeeds (no import/collection errors anywhere) | free |
| 2 | `unit` | ~1000 fast tests, no network, no external services | free |
| 3 | `integration` | subsystems against local resources (tmp dirs, in-process apps) | free |
| 4 | `staging-e2e` | the real container image, driven from outside over real MCP/REST, against simulated providers — every MCP tool exercised AND traced (see [Traceability](#traceability)) | free |
| 5 | `live` | real providers: OpenRouter, DeepInfra, LiteLLM `ocr`/`vision`, comm-store Postgres | $$ |

The ordering is a **spend gate by design**: each tier only runs if every
tier before it passed (`scripts/gate.py` fails fast). Fakes rule out code
issues before any money is spent — by the time the live tier starts, a
failure there can only mean provider/infra trouble, not a bug the free
tiers could have caught. The live tier is additionally guarded by a
[preflight](#live-tier) that must exit 0 before `pytest -m live` launches.

Every run leaves artifacts in `.evals/gate-runs/<timestamp>/`: junit XML
per tier, `result.json` (machine-readable tier states + overall verdict),
`tool-coverage.json`, collected trace spans, and a rendered `report.md`.

A gate that has never fully passed is not a gate — the five-tier run has
been executed to completion against real providers; keep it that way.

## Quick reference

| Command | Runs |
|---|---|
| `make gate` | all five tiers, fail-fast, artifacts per run |
| `make gate-fast` | tiers 1–3 only (static, unit, integration) — the dev loop |
| `make gate-real` | full gate **+** an opt-in real-API e2e pass (media + enrichment live; additional spend, needs a real key) |
| `make test-unit` | `pytest -m unit -q` directly |
| `make test-integration` | `pytest -m integration -q` directly |
| `make test-e2e` | gate `--only staging-e2e` (brings the compose stack up/down) |
| `make test-e2e-real` | gate `--only e2e-real` — just the real-API e2e stage (SPENDS MONEY) |
| `make test-live` | gate `--only live` (preflight still enforced) |

`make gate` includes the real-provider `live` tier and can spend money. It does
not run the optional `e2e-real` stage. `make gate-real` adds that stage after a
full gate; `make test-e2e-real` runs only that stage. Both optional real-e2e
commands require a real `OPENROUTER_API_KEY`. See "Live tier" and "Cross-repo
test target" below for the deterministic-then-real workflow.

### Tier selection: auto-derived markers

`tests/conftest.py::pytest_collection_modifyitems` assigns exactly one
tier marker per test, derived from file naming — an explicit marker
always wins over the filename rule:

| Convention | Tier |
|---|---|
| `_live` in filename (e.g. `test_extractors_live.py`) | `live` |
| under `tests/e2e/` or `.e2e.test` in filename | `e2e` |
| `.int.test.py` suffix (e.g. `test_mcp_handlers.int.test.py`) | `integration` |
| everything else | `unit` |

So a new test file is a unit test by default; putting it in the right
place/name is all the wiring a tier needs. Markers are registered in
`pyproject.toml` (`[tool.pytest.ini_options] markers`).

## The staging stack

`docker-compose.staging.yml`, project name `doc-organizer-staging`,
fully hermetic — no provider traffic leaves the machine. Three services:

| Service | Host port | Role |
|---|---|---|
| `doc-organizer-staging` | **17788** → 7788 | the real app image, `config.staging.yaml` bind-mounted over `/app/config.yaml`, API key `staging-test-key` |
| `provider-sim` | **19999** → 9999 | one FastAPI app speaking OpenRouter, DeepInfra, DeepSeek-OCR2 and Ollama dialects, plus webhook sink and fault-injection admin |
| `comm-postgres` | (internal) | throwaway Postgres 16 with deterministic fixtures, PGDATA on tmpfs — every `up` reseeds |

Bring it up manually:

```bash
docker compose -f docker-compose.staging.yml up -d --build --wait
# ... run tests / poke around ...
docker compose -f docker-compose.staging.yml down -v
```

`scripts/gate.py` does exactly this around the staging-e2e tier
(`down -v` always runs, even after a partial `up`). No `restart:`
policies anywhere — a crash-looping service fails `up --wait`
immediately instead of flapping.

### How the sim works (`staging/provider_sim/app.py`)

- **Deterministic everything**: embeddings are content-hash-derived unit
  vectors (`fake_embedding`), chat/enrichment/OCR/transcript responses
  embed a sha-12 of the input. Same input → same output, forever; no
  randomness, no wall clock.
- **Webhook sink**: the app's `event_hooks` post `document.indexed`
  events to `POST /hooks/sink`; tests read them back via
  `GET /hooks/received`. `POST /admin/reset` clears sink + armed faults
  (the e2e `sim_reset` fixture does this before and after every test).
- **Fault injection**: see [Fault injection how-to](#fault-injection-how-to).

### config.staging.yaml — the two landmines

1. **`documents_root` and `sources:` are mutually exclusive**
   (`core/config.py` raises "Cannot use both"). The staging config is
   sources-mode; do not add a `documents_root` on top.
2. **The postgres source MUST be named `sor`** —
   `sor_query.resolve_sor_dsn()` looks it up by that exact name. Rename
   it and every `file_sor_*` tool silently loses its DSN.

## Traceability

- **Span JSONL** (`core/tracing.py`): OTEL SDK with a JSONL file
  exporter — no collector, no UI. Off unless `tracing.enabled: true`.
  One file per process (`spans-<pid>.jsonl`), one JSON object per span:
  `name`, `trace_id`, `span_id`, `parent_span_id`, `start_ns`, `end_ns`,
  `status`, `attributes`.
- **Where traces land**: in the container, `tracing.directory` is
  `/data/traces` (the `staging-traces` volume). After the e2e pytest run
  — inside the compose window — `gate.py` copies them out via
  `docker compose cp` to `<run_dir>/traces/`.
- **`mcp.tool.<name>` spans**: every registered MCP tool emits one
  server-side span per invocation, via a single generic wrapper composed
  into `mcp.tool()` at registration time (`mcp_server.py`). New tools
  are instrumented automatically; nothing to hand-write.
- **Two-sided coverage check** (`scripts/check_tool_coverage.py`, runs
  inside the compose window as part of the staging-e2e tier): every tool
  discovered via live `list_tools` must be
  1. **covered** — at least one successful client-side e2e call recorded
     in `.evals/e2e-tool-coverage.jsonl` (written by
     `tests/e2e/client.py::RecordingSession`), and
  2. **traced** — at least one `mcp.tool.<name>` span in the collected
     trace artifacts.

  `uncovered` means no e2e test calls the tool; `untraced` means the
  call happened but traceability is broken. Either fails the tier.
  Zero discovered tools also fails (a broken `list_tools` must never
  greenlight the gate).

### Reading a gate report (`<run_dir>/report.md`)

- **Title** — `PASS` / `FAIL` from the runner's own `result.json`;
  `INCOMPLETE` only when both `result.json` and all junit artifacts are
  absent (a run that died before producing evidence).
- **Tiers table** — one row per tier including `static` (whose state
  exists only in `result.json`; counts render as `-`). Failing test
  names are listed under the table.
- **Tool coverage** — `N/N covered, N/N traced` plus a per-tool
  tests/spans matrix; `uncovered`/`untraced` lists name the offenders.
- **Document timelines** — per-document stage waterfalls reconstructed
  from `process_doc` traces; `⚠` flags any span with ERROR status, with
  a count at the top of the section.
- **Live spend** — per provider/model call counts aggregated from
  `.evals/llm-traces/*.jsonl`, filtered to records after the run start.

## Live tier

`pytest -m live` runs in-process against the real world: OpenRouter, DeepInfra,
the configured LiteLLM proxy, and the comm-store Postgres (localhost:5433).
LiteLLM is the OCR routing authority: alias `ocr` handles extraction, alias
`vision` handles image description, and the proxy owns local/cloud fallback
routing. The first-class OCR factory does not read a LiteLLM credential from
YAML; it reads `LITELLM_API_KEY` first, then `LITELLM_MASTER_KEY`, from the
environment.

`scripts/live_preflight.py` runs all five checks (no early exit — one run
reports every problem) and **only exit code 0 releases the spend**:

| Check | What it verifies | When it fails |
|---|---|---|
| `api_keys` | `OPENROUTER_API_KEY` and `DEEPINFRA_API_KEY` non-empty | set them in `.env` (dotenv is loaded) |
| `config_test` | `config_test.yaml` present in CWD | it's gitignored so fresh worktrees don't have it — copy from the main checkout (the failure message prints the path) |
| `litellm_ocr` | reads the first existing config candidate (CWD/worktree `config.yaml` before the main checkout), authenticates a non-generating `GET {ocr.endpoint}/models`, and confirms both configured aliases | config/endpoint/model alias/credential missing, endpoint unreachable, authentication rejected, malformed response, or either alias absent; credentials belong only in `LITELLM_API_KEY` or `LITELLM_MASTER_KEY` |
| `prod_indexer_idle` | prod container heartbeat (`/data/index/indexer.heartbeat`) older than 120 s | **"prod indexer active — rerun when quiet"**: the prod indexer is mid-run and would contend with the live tier for shared LiteLLM/local inference hardware. Wait for it to finish and re-run; the preflight itself is the poll (it prints the heartbeat age). |
| `comm_postgres` | `COMM_DATA_STORE_DSN` answers `SELECT 1` | DSN unset or Postgres down |

The LiteLLM probe is fail-closed: missing configuration, network/auth errors,
invalid JSON/schema, and missing aliases all fail preflight without generating
model output. Failure messages identify the endpoint/check but never print the
credential.

**Worktree setup**: linked worktrees need their own copies of three
gitignored files from the main checkout — `config_test.yaml`, `.env`,
and `config.yaml` (several live tests call `load_config()` on the real
config for provider settings and the read-only index health checks; its
paths are absolute, so a straight copy is correct — without it the live
tier fails ~35 tests on `FileNotFoundError: config.yaml`). The `.env`
copy must keep the host-side DSN rewrite
(`COMM_DATA_STORE_DSN=postgresql://...@localhost:5433/...`), since live
tests run on the host, not in a container. CWD precedence means a worktree's
copied `config.yaml` is the exact provider configuration preflight validates.
OCR stays disabled in `config_test.yaml`; the dedicated LiteLLM live test owns
all real OCR/vision generation.

`tests/test_litellm_ocr_live.py` creates one large image containing the OCR
sentinel `RAGBOX LIVE OCR 7319`, a red circle, and a blue square. It contains
two non-skipping generation smoke invocations: one through alias `ocr` to
recover the normalized sentinel, and one through alias `vision` to identify
both colors and shapes. Provider retries and the preflight `/models` probe can
add HTTP requests, so these are not a total request count. Missing
infrastructure fails preflight or the tests; the file has no skip fallback.

Live failures come in two flavors: infrastructure flakes (LiteLLM or a local
backend busy, provider rate limits — retry once, document) and genuine failures
(stop, investigate, never weaken the test to pass).

## Adding a new MCP tool

Coverage is **enforced, not opt-in**: the checker discovers tools from
the live `list_tools` endpoint, so the moment your tool is registered it
is REQUIRED to have e2e coverage and spans — otherwise the staging-e2e
tier fails with your tool listed under `uncovered`/`untraced`.

What to add where:

1. Register the tool in `mcp_server.py` with `@mcp.tool()` as usual —
   the `mcp.tool.<name>` span wrapper is applied automatically.
2. Add at least one e2e test in `tests/e2e/test_tools_*.py` that calls
   it through the fixture session (`mcp_session.call_tool_json(...)`) —
   `RecordingSession` records client-side coverage on every successful
   call. Use the `indexed_corpus` fixture if the tool needs data.
3. Run `make test-e2e` — the coverage matrix at the end must show your
   tool with `tests ≥ 1` and `spans ≥ 1`.

## Fault injection how-to

The sim supports three faults: `429` (rate-limit response with
`Retry-After: 0`), `timeout` (delay, then answer normally), `garbage`
(HTTP 200, body `not json {`).

**Single-shot, header mode** — for direct requests to the sim:

```bash
curl -H 'X-Sim-Fault: 429' http://localhost:19999/api/v1/embeddings ...
curl -H 'X-Sim-Fault: timeout' -H 'X-Sim-Fault-Seconds: 5' ...
```

**Armed mode** — for faults that must hit app→sim traffic (the app
doesn't send the header). Arm N failures for a route prefix, then
trigger app work:

```bash
curl -X POST http://localhost:19999/admin/fault \
  -d '{"route_prefix": "/api/v1/embeddings", "fault": "429", "times": 2}'
```

Armed faults decrement per hit and disarm at zero. Unknown fault names
are a 400 (a typo must not silently no-op into a false "retry
recovered"). `POST /admin/reset` clears everything; the e2e suite does
this around every test.

## Known limitations

Honest edges of the current gate — known, documented, not silently
papered over:

- **Single-doc indexing is vector-only until the next sweep.** The
  targeted per-document index path updates the vector store but not the
  FTS index; keyword/hybrid recall for that doc catches up on the next
  full sweep. Decision on closing the gap is documented as pending.
- **The spend section reports latency, not dollars.** LLM trace records
  carry per-call latency; cost-per-token accounting is not wired in.
- **e2e media fixtures bypass the upload API.** `/api/upload` allowlists
  document extensions only (no `.wav`/`.mp4`), so `clip.wav`/`clip.mp4`
  are deposited via `docker compose cp` into `/data/documents` — which
  faithfully emulates the production deposit path (comm hooks writing to
  the shared volume), but means uploads of media are not exercised.
- **`file_folders` undercounts in sources-mode.** Folder aggregation
  predates multi-source configs; counts for non-filesystem sources are
  incomplete.

## Cross-repo test target (CDS)

Comm-Data-Store points its media-enrichment e2e at a **parallel, fully
isolated** copy of the staging stack — never at production. The overlay
`docker-compose.staging.cds.yml` gives it a distinct compose project
(`cds-doc-organizer`), distinct host ports (**27788** app / **29999**
sim), and its own throwaway `cds-*` volumes, so it can run alongside a
`make gate` run without either stack colliding with or tearing down the
other. Nothing it touches reaches prod: prod is a different container, a
different `/data/documents` host directory, different volumes, a
different network, and port 7788. The CDS stack's `/data/documents` is a
throwaway volume that starts **empty** every `up`.

**Two-stage run — deterministic first, real API last.** Matches the
gate's own fakes-then-live philosophy:

```bash
# 1. Deterministic (all providers = sim, free, catches wiring bugs):
docker compose -f docker-compose.staging.yml -f docker-compose.staging.cds.yml up -d --build --wait

# 2. Final real-API pass (media + enrichment = live OpenRouter, SPENDS MONEY) —
#    only after stage 1 is green:
export OPENROUTER_API_KEY=<real key>
STAGING_CONFIG=./config.staging.realmedia.yaml \
  docker compose -f docker-compose.staging.yml -f docker-compose.staging.cds.yml up -d --build --wait

# Tear down (wipes throwaway volumes):
docker compose -f docker-compose.staging.yml -f docker-compose.staging.cds.yml down -v
```

`STAGING_CONFIG` selects the provider wiring for any staging stack (base or
CDS overlay); unset = the all-sim `config.staging.yaml`. The real-media config
(`config.staging.realmedia.yaml`) repoints only `media` and `enrichment` at
real OpenRouter (with an audio fallback chain, since `whisper-1` 500s in
practice); embeddings, OCR, and reranker stay on the sim — deterministic and
free — because they are not the media path under test.

**Real `document.indexed` callback (the prod integration CDS tests).** Both
staging configs carry a second, env-driven event hook `cds-callback` alongside
the sim sink:

```yaml
- name: "cds-callback"
  url: "${CDS_HOOK_URL}"     # unset → silently skipped (gate unaffected)
  events: ["document.indexed"]
```

The app resolves `${CDS_HOOK_URL}` from the container env at delivery time (a
`${VAR}` url that resolves empty is a deliberate no-op — no warning per doc).
The CDS overlay passes `CDS_HOOK_URL` through and adds
`extra_hosts: ["host.docker.internal:host-gateway"]` so the container can reach
a hook on the host. To receive the real callback:

```bash
export CDS_HOOK_URL=http://host.docker.internal:8095/hooks/doc-indexed
docker compose -f docker-compose.staging.yml -f docker-compose.staging.cds.yml up -d --build --wait
# indexing any doc now POSTs document.indexed {doc_id, rel_path, metadata} to :8095
```

The payload is unchanged (`doc_id` + `rel_path` + sanitized `metadata`,
including `enr_*` enrichment fields) — the same event the sim sink and prod's
comm-data-store hook already consume.

**Standing seeded corpus (optional).** The gate e2e always starts from an empty
index, but for manual / CDS testing against a persistent parallel dataset, drop
files into `staging/fixtures/corpus/` and run `scripts/seed_staging.sh` against a
running stack — it deposits each file into `/data/documents` and indexes it. Env
overrides (`SEED_COMPOSE`, `SEED_URL`, `SEED_API_KEY`, `SEED_CORPUS`) point it at
the base stack (`:17788`, default) or the CDS overlay (`:27788`). This is
testing-only data on a throwaway volume — never production. To keep the corpus
across restarts, stop with `down` (NOT `down -v`); `down -v` wipes it.

**Index contract:**
- Endpoint: `POST http://127.0.0.1:27788/api/index/document`, body
  `{"rel_path"|"abs_path"|"doc_id": ..., "source_name"?: "documents", "force"?: false}`.
- Auth: `Authorization: Bearer staging-test-key` on all `/api/*` (only
  `/health` is exempt). This key is a committed non-secret fixed literal —
  **different from prod's real key; do not reuse across the two.**
- The endpoint indexes a file that **must already exist** in
  `/data/documents`. `/api/upload` rejects media extensions, so deposit
  media first (`docker compose ... cp <file> doc-organizer-staging:/data/documents/<name>`),
  then call the index endpoint with that `rel_path`.
- Reads back: the doc is **vector-searchable immediately**; keyword/FTS
  visibility waits for a full sweep (which this stack may never run), so
  assert via semantic search.
