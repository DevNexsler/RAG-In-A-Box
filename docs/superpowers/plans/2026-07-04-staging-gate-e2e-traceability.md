# Staging Gate, E2E Suite, and Traceability Stack — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `make gate` command whose full pass means "this will work in production": ordered tiers (static → unit → integration → staging-e2e → live), a disposable memory-capped staging compose stack with a provider simulator, an e2e suite covering all 20 MCP tools from the outside, and OTEL-SDK span tracing with a tool-coverage matrix in a per-run report.

**Spec:** `docs/superpowers/specs/2026-07-04-staging-gate-e2e-traceability-design.md`

**Architecture:** The staging container runs the unmodified production image; a FastAPI `provider-sim` container speaks the exact HTTP dialects of OpenRouter/DeepInfra/DeepSeek-OCR/Ollama (deterministic, fault-injectable); a seeded Postgres plays comm-store. Tests select into tiers via markers auto-derived from existing filename conventions. Tracing is OTEL SDK + a JSONL file exporter — no collector, no UI.

**Tech Stack:** pytest (markers, anyio), FastAPI (simulator), docker compose, `mcp` Python client (streamable HTTP), `opentelemetry-sdk` + `opentelemetry-instrumentation-httpx`, Makefile + Python gate runner.

**Project rules that apply to every task (from CLAUDE.md):**
- Before modifying any existing function/class: `gitnexus_impact({target: "<symbol>", direction: "upstream"})`; report blast radius; stop and warn on HIGH/CRITICAL.
- Before every commit: `gitnexus_detect_changes({scope: "staged"})` and confirm only expected symbols changed.
- After commits, the PostToolUse hook re-runs `npx gitnexus analyze` automatically.

**Verified facts this plan relies on (from research, 2026-07-04):**
- MCP server: FastMCP `streamable_http_app()` mounted with `/health` + `/api/*` on one Starlette app, port 7788 (`mcp_server.py:2589-2632`, `server.py`).
- REST: `POST /api/upload`, `GET /api/documents`, `GET /api/documents/{doc_id:path}`, `POST /api/search`, `POST /api/index/document` (`api_server.py:299-306`).
- Heartbeat: `{index_root}/indexer.heartbeat`, unix-timestamp text, written by the full-flow subprocess (`flow_index_vault.py:650-660`), staleness `INDEXER_HEARTBEAT_MAX_AGE` default 1800s.
- Provider dialects and the exact response fields parsed: see "Simulator dialect contract" in Task 6.
- Hardcoded provider URLs needing `base_url` config: OpenRouter embeddings (`providers/embed/openrouter_embed.py:79`), OpenRouter LLM (`providers/llm/openrouter_llm.py:165`), DeepInfra rerank (`search_hybrid.py:293`), OpenRouter media (`providers/media/openrouter_media.py:170`). `enrichment.base_url`, `ocr.base_url`, `media.base_url` config keys already exist; embeddings and reranker need new keys.
- Integration files: 8 of 9 `.int.test.py` are local-only; `test_multi_source_flow.int.test.py` requires API keys (already `@pytest.mark.live`) and must be renamed out of the integration tier.
- Existing markers: only `live` in `pyproject.toml`; `addopts = "--import-mode=importlib"`.
- Fixtures: `test_vault/` has md/pdf/png; **no audio/video fixtures exist** — they must be added.
- Transient retry contract: statuses `{408, 425, 429, 500, 502, 503, 504}`, `core/resilience.py:36`.

---

## Phase 1 — Tiers and gate skeleton

### Task 1: Auto-derived pytest markers + tier enforcement

**Files:**
- Modify: `pyproject.toml` (markers section)
- Modify: `tests/conftest.py`
- Rename: `tests/test_multi_source_flow.int.test.py` → `tests/test_multi_source_flow_live.py`
- Rename: `tests/test_production_readiness.e2e.test.py` → `tests/test_production_readiness.int.test.py`
- Test: `tests/test_tier_markers.py`

**Design note (deliberate deviation from spec):** the spec asked for a conftest guard that fails collection on marker/filename disagreement; this plan instead auto-derives markers from filenames, which enforces agreement by construction — there is nothing left to disagree. Two files whose filename convention lies about their tier get renamed here: `test_multi_source_flow` (marked `live`, needs API keys) and `test_production_readiness` (in-process with mock providers — that is this project's definition of *integration*, and the `.e2e.test` name would otherwise drop it out of every tier once the e2e tier becomes path-restricted to `tests/e2e/`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tier_markers.py
"""The tier system: markers are auto-derived from filename conventions."""
import subprocess, sys, json

def _collect(marker):
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", marker, "tests/"],
        capture_output=True, text=True,
    )
    return out.stdout

def test_integration_tier_matches_filenames():
    out = _collect("integration")
    assert ".int.test" in out
    assert "_live" not in out          # live files never in integration tier
    # (covers the renamed test_multi_source_flow_live.py too, via the _live check)

def test_unit_tier_excludes_special_files():
    out = _collect("unit")
    assert ".int.test" not in out
    assert ".e2e.test" not in out
    assert "_live" not in out

def test_live_tier_collects_live_files():
    out = _collect("live")
    assert "_live" in out
```

- [ ] **Step 2: Run it to make sure it fails** — `pytest tests/test_tier_markers.py -v`. Expected: FAIL (markers `integration`/`unit` unknown, collection empty).

- [ ] **Step 3: Register markers in `pyproject.toml`**

```toml
[tool.pytest.ini_options]
markers = [
    "unit: fast, no network, no external services",
    "integration: subsystems with local resources only (tmp dirs, in-process apps)",
    "e2e: drives the staging container from outside (compose stack must be up)",
    "live: real providers, real money (API keys, Mac Mini OCR, comm-store Postgres)",
]
addopts = "--import-mode=importlib"
```

- [ ] **Step 4: Add auto-marker hook to `tests/conftest.py`** (existing sys.path insert stays):

```python
import pytest

def pytest_collection_modifyitems(config, items):
    for item in items:
        fname = item.fspath.basename
        if item.get_closest_marker("live") or "_live" in fname:
            item.add_marker(pytest.mark.live)
        elif ".e2e.test" in fname or str(item.fspath).endswith("tests/e2e") or "/tests/e2e/" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif ".int.test" in fname:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)
```

Order matters: an explicit `@pytest.mark.live` wins over a `.int.test` filename, which is exactly the `test_multi_source_flow` bug.

- [ ] **Step 5: Rename the two mis-tiered files** so filename and tier agree:

```bash
git mv tests/test_multi_source_flow.int.test.py tests/test_multi_source_flow_live.py
git mv tests/test_production_readiness.e2e.test.py tests/test_production_readiness.int.test.py
```

- [ ] **Step 6: Run the tier tests** — `pytest tests/test_tier_markers.py -v`. Expected: 3 PASS. Also sanity: `pytest -m unit --collect-only -q | tail -1` shows ~50+ files; `pytest -m integration --collect-only -q | tail -1` shows exactly 9 files (the 8 local-only `.int.test` files + the renamed `test_production_readiness.int.test.py`).

- [ ] **Step 7: Run the unit tier for real** — `pytest -m unit -x -q`. Expected: PASS (fix any test that breaks purely from marker addition; do not touch failing tests unrelated to markers — report them).

- [ ] **Step 8: Commit** — `git add -A tests/ pyproject.toml && git commit -m "test: tier markers auto-derived from filename conventions"` (run `gitnexus_detect_changes({scope:"staged"})` first; expect tests-only).

### Task 2: Makefile + gate runner

**Files:**
- Create: `Makefile`
- Create: `scripts/gate.py`
- Modify: `requirements.txt` (add `ruff`)
- Modify: `pyproject.toml` (add `[tool.ruff]` baseline config)
- Test: `tests/test_gate_runner.py`

**Ruff baseline policy:** ruff is NOT currently installed and the codebase has never been linted. Do not attempt a repo-wide cleanup inside this plan. Install ruff, then start from a narrow always-defensible rule set that passes today:

```toml
[tool.ruff]
line-length = 120

[tool.ruff.lint]
# Baseline: syntax errors and definite bugs only. Broaden deliberately, later.
select = ["E9", "F63", "F7", "F82", "F401"]
```

Run `ruff check .` once; if `F401` (unused imports) reports pre-existing hits, either autofix them in a dedicated commit (`ruff check --fix --select F401`) or drop `F401` from the baseline — implementer's choice, but the static tier MUST pass on the current codebase before Task 2 completes. The static tier is `ruff check .` **plus** `pytest --collect-only -q` (import/syntax sanity across all tests, per spec).

- [ ] **Step 1: Write failing tests for the runner's tier logic**

```python
# tests/test_gate_runner.py
# NOTE: `scripts` has no __init__.py — this import works via conftest's sys.path
# insert + namespace packages. Do not "fix" by adding __init__.py.
from scripts.gate import TIERS, next_tier_allowed

def test_tier_order():
    assert [t.name for t in TIERS] == ["static", "unit", "integration", "staging-e2e", "live"]

def test_fail_fast():
    results = {"static": True, "unit": False}
    assert next_tier_allowed("integration", results) is False

def test_live_requires_all_prior():
    results = {"static": True, "unit": True, "integration": True, "staging-e2e": True}
    assert next_tier_allowed("live", results) is True
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_gate_runner.py -v`. Expected: FAIL (no scripts.gate module).

- [ ] **Step 3: Implement `scripts/gate.py`**

Core structure (complete the obvious glue; each tier writes junit XML into the run dir):

```python
#!/usr/bin/env python3
"""Gate runner: ordered tiers, fail-fast, artifacts per run.

Usage: python scripts/gate.py [--fast] [--only TIER] [--run-dir DIR]
"""
import dataclasses, subprocess, sys, time
from pathlib import Path

@dataclasses.dataclass(frozen=True)
class Tier:
    name: str
    cmd: list[str]          # command template; {run_dir} substituted
    needs_compose: bool = False

RUN_ROOT = Path(".evals/gate-runs")

TIERS = [
    # static is two commands; gate.py runs them in sequence, both must pass
    Tier("static", ["ruff", "check", "."]),   # + ["python", "-m", "pytest", "--collect-only", "-q"]
    Tier("unit", ["python", "-m", "pytest", "-m", "unit", "-q",
                  "--junitxml={run_dir}/unit.xml"]),
    Tier("integration", ["python", "-m", "pytest", "-m", "integration", "-q",
                         "--junitxml={run_dir}/integration.xml"]),
    Tier("staging-e2e", ["python", "-m", "pytest", "tests/e2e", "-m", "e2e", "-q",
                         "--junitxml={run_dir}/e2e.xml"], needs_compose=True),
    Tier("live", ["python", "-m", "pytest", "-m", "live", "-q",
                  "--junitxml={run_dir}/live.xml"]),
]

def next_tier_allowed(name: str, results: dict[str, bool]) -> bool:
    for t in TIERS:
        if t.name == name:
            return True
        if not results.get(t.name, False):
            return False
    return False
```

The `main()` (same file): create `RUN_ROOT/<YYYYmmdd-HHMMSS>/`, iterate tiers, honor `--fast` (stop after integration), for `needs_compose` tiers wrap in compose up/down (Task 7 provides the compose file; until then `staging-e2e` exits with a clear "compose file missing" error), run live preflight before the live tier (Task 10; until then, hard-fail with "preflight not implemented"), always finish by invoking `scripts/gate_report.py` (Task 11; tolerate absence until then), exit non-zero on first tier failure.

Compose wrapping in `main()` must be exception-safe:

> **Correction (review, Task 2):** `subprocess.run(up, check=True)` belongs INSIDE the
> `try` — otherwise a partially-started stack (up fails midway) never gets `down -v`.
> The implementation in `scripts/gate.py` does this; the snippet below kept the
> original (wrong) placement for the record.

```python
if tier.needs_compose:
    up = ["docker", "compose", "-f", "docker-compose.staging.yml", "up", "-d", "--build", "--wait"]
    down = ["docker", "compose", "-f", "docker-compose.staging.yml", "down", "-v"]
    subprocess.run(up, check=True)
    try:
        ok = run_tier(tier, run_dir)
        collect_staging_traces(run_dir)   # docker cp traces volume → run_dir/traces/
    finally:
        subprocess.run(down, check=False)
```

- [ ] **Step 4: Run tests** — `pytest tests/test_gate_runner.py -v`. Expected: PASS.

- [ ] **Step 5: Create `Makefile`**

```makefile
.PHONY: gate gate-fast test-unit test-integration test-e2e test-live

gate:
	python scripts/gate.py

gate-fast:
	python scripts/gate.py --fast

test-unit:
	python -m pytest -m unit -q

test-integration:
	python -m pytest -m integration -q

test-e2e:
	python scripts/gate.py --only staging-e2e

test-live:
	python scripts/gate.py --only live
```

- [ ] **Step 6: Verify `make gate-fast` runs tiers 1–3** end to end. Expected: static + unit + integration run; PASS (or report pre-existing failures verbatim — do not fix unrelated tests silently).

- [ ] **Step 7: Commit** — `feat: gate runner with ordered fail-fast tiers`.

---

## Phase 2 — Traceability (can land independently)

### Task 3: `core/tracing.py` + JSONL file exporter

**Files:**
- Create: `core/tracing.py`
- Modify: `requirements.txt` (add `opentelemetry-sdk`, `opentelemetry-instrumentation-httpx`)
- Modify: `config.yaml.example` (document `tracing:` section)
- Test: `tests/test_tracing.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tracing.py
import json
from pathlib import Path
from core.tracing import setup_tracing, get_tracer, shutdown_tracing

def test_disabled_is_noop(tmp_path):
    setup_tracing({"tracing": {"enabled": False}}, service_name="t")
    with get_tracer("x").start_as_current_span("s"):
        pass  # must not raise, must not write anything
    assert not list(tmp_path.rglob("*.jsonl"))

def test_spans_written_as_jsonl(tmp_path):
    cfg = {"tracing": {"enabled": True, "directory": str(tmp_path)}}
    setup_tracing(cfg, service_name="test-svc")
    tracer = get_tracer("pipeline")
    with tracer.start_as_current_span("extract", attributes={"doc_id": "d1"}):
        with tracer.start_as_current_span("ocr"):
            pass
    shutdown_tracing()
    files = list(tmp_path.glob("*.jsonl"))
    assert files
    spans = [json.loads(l) for f in files for l in f.read_text().splitlines()]
    by_name = {s["name"]: s for s in spans}
    assert by_name["extract"]["attributes"]["doc_id"] == "d1"
    assert by_name["ocr"]["parent_span_id"] == by_name["extract"]["span_id"]
    assert by_name["ocr"]["trace_id"] == by_name["extract"]["trace_id"]
```

- [ ] **Step 2: Verify failure** — `pytest tests/test_tracing.py -v`. Expected: FAIL (module missing).

- [ ] **Step 3: Implement `core/tracing.py`**

```python
"""OTEL SDK tracing with a JSONL file exporter. No collector, no UI.

Off unless config has tracing.enabled: true. Span JSONL lands in
tracing.directory (default .evals/spans), one file per process start.
"""
import json, os, threading
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

_provider = None
_lock = threading.Lock()

class JsonlSpanExporter(SpanExporter):
    def __init__(self, directory: str):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"spans-{os.getpid()}.jsonl"
        self._flock = threading.Lock()

    def export(self, spans) -> SpanExportResult:
        lines = []
        for s in spans:
            ctx, parent = s.get_span_context(), s.parent
            lines.append(json.dumps({
                "name": s.name,
                "trace_id": format(ctx.trace_id, "032x"),
                "span_id": format(ctx.span_id, "016x"),
                "parent_span_id": format(parent.span_id, "016x") if parent else None,
                "start_ns": s.start_time, "end_ns": s.end_time,
                "status": s.status.status_code.name,
                "attributes": dict(s.attributes or {}),
            }, default=str))
        try:
            with self._flock, self._path.open("a") as f:
                f.write("\n".join(lines) + "\n")
            return SpanExportResult.SUCCESS
        except OSError:
            return SpanExportResult.FAILURE  # never crash the pipeline over tracing

    def shutdown(self):  # noqa: D102
        pass

def setup_tracing(config: dict, service_name: str) -> None:
    global _provider
    tcfg = (config or {}).get("tracing") or {}
    if not tcfg.get("enabled", False):
        return
    with _lock:
        if _provider is not None:
            return
        _provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        exporter = JsonlSpanExporter(tcfg.get("directory", ".evals/spans"))
        _provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(_provider)
        try:
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
            HTTPXClientInstrumentor().instrument()
        except Exception:
            pass  # tracing must never take the service down

def get_tracer(name: str):
    return trace.get_tracer(name)

def shutdown_tracing() -> None:
    global _provider
    if _provider is not None:
        _provider.shutdown()
        _provider = None
```

Note: `SimpleSpanProcessor` (synchronous) is deliberate — spans are few (per-document, not per-token) and it guarantees spans are on disk when a test asserts. If profiling ever shows overhead, switch to `BatchSpanProcessor` + flush in `shutdown_tracing`.

- [ ] **Step 4: Run tests** — `pytest tests/test_tracing.py -v`. Expected: PASS.

- [ ] **Step 5: Add deps to `requirements.txt`** (`opentelemetry-sdk>=1.25`, `opentelemetry-instrumentation-httpx>=0.46b0`) and a commented `tracing:` block to `config.yaml.example`:

```yaml
# tracing:
#   enabled: false          # spans as JSONL; no external services
#   directory: .evals/spans # inside the container use /data/traces
```

- [ ] **Step 6: Commit** — `feat(core): OTEL tracing with JSONL file exporter`.

### Task 4: Instrument the six pipeline stages + join LLMTraceRecorder

**Files:**
- Modify: `server.py` (call `setup_tracing(config, "doc-organizer")` at boot)
- Modify: `flow_index_vault.py` (`scan_vault_task` ~:525, `process_doc_task` ~:890 — the per-doc parent span; inside it, child spans around extract/OCR, enrich, embed, store call sites)
- Modify: `doc_enrichment.py` (`enrich_document`)
- Modify: `lancedb_store.py` (`upsert_nodes`)
- Modify: `search_hybrid.py` (`hybrid_search`)
- Modify: `providers/llm/trace_recorder.py` (`record()` gains current trace/span ids)
- Test: `tests/test_tracing_pipeline.py`

**MANDATORY before edits:** run `gitnexus_impact` upstream on each of: `process_doc_task`, `scan_vault_task`, `enrich_document`, `upsert_nodes`, `hybrid_search`, `LLMTraceRecorder.record`. Report blast radius. These are hot-path symbols; expect non-trivial caller counts. If HIGH/CRITICAL → stop and warn the user before proceeding.

- [ ] **Step 1: Impact analysis (above) — record results in the task notes.**

- [ ] **Step 2: Write failing test** — index one tiny doc through the real flow with mock providers (reuse the wiring pattern from `test_indexing_roundtrip.int.test.py`) with `tracing.enabled: true` pointed at `tmp_path`; assert the span file contains a `process_doc` span with `doc_id` attribute and child spans named `extract`, `enrich`, `embed`, `store.upsert`, all sharing one `trace_id`; then run a search and assert a `search.hybrid` span exists.

- [ ] **Step 3: Verify failure**, then instrument. Pattern — spans are additive wrappers, never control flow:

```python
from core.tracing import get_tracer
_tracer = get_tracer("pipeline")

# inside process_doc_task, wrapping the existing body:
with _tracer.start_as_current_span(
    "process_doc",
    attributes={"doc_id": doc_id, "rel_path": rel_path, "source": source_name},
):
    ...existing body, with child spans around the four stage call sites...
```

`LLMTraceRecorder.record()` addition (keeps signature, adds two fields to the JSONL payload):

```python
from opentelemetry import trace as _otel_trace
ctx = _otel_trace.get_current_span().get_span_context()
if ctx.is_valid:
    payload["trace_id"] = format(ctx.trace_id, "032x")
    payload["span_id"] = format(ctx.span_id, "016x")
```

- [ ] **Step 4: Run** `pytest tests/test_tracing_pipeline.py tests/test_openrouter_trace_capture.py -v`. Expected: PASS (existing trace-capture tests must not break).

- [ ] **Step 5: Run `make gate-fast`** to prove instrumentation broke nothing. Expected: same results as Task 2 Step 6.

- [ ] **Step 6: Commit** — `feat: OTEL spans on pipeline stages; trace ids join LLM trace records`. Pre-commit `gitnexus_detect_changes` must list exactly the six instrumented symbols + trace_recorder.

---

## Phase 3 — Provider simulator

### Task 5: `base_url` overrides for hardcoded provider endpoints

**Files:**
- Modify: `providers/embed/openrouter_embed.py` (~:79 — endpoint from `embeddings.base_url`, default `https://openrouter.ai/api/v1`)
- Modify: `providers/llm/openrouter_llm.py` (~:165 — honor `enrichment.base_url` when provider is openrouter)
- Modify: `search_hybrid.py` (~:293 — endpoint from `search.reranker.base_url`, default `https://api.deepinfra.com`)
- Modify: `providers/media/openrouter_media.py` (~:170 — honor existing `media.base_url`)
- Modify: `config.yaml.example` (document the new keys)
- Test: `tests/test_provider_base_urls.py`

**MANDATORY:** `gitnexus_impact` upstream on the constructor/entry of each modified provider (`OpenRouterEmbedProvider`, `OpenRouterLLMProvider`, the rerank function in `search_hybrid.py`, the media provider class) before editing.

- [ ] **Step 1: Failing tests** — for each provider, instantiate with a config carrying `base_url: "http://sim:9999"` and assert the URL used by the request call (patch `httpx` client, capture the URL) starts with the override; and that omitting the key preserves today's default URL exactly.
- [ ] **Step 2: Verify failure.**
- [ ] **Step 3: Implement** — one pattern everywhere: `self._base_url = (cfg.get("base_url") or DEFAULT).rstrip("/")`, endpoint = `f"{self._base_url}/embeddings"` etc. No behavior change when unset.
- [ ] **Step 4: Run** the new tests + the provider unit tests (`pytest -m unit -k "openrouter or rerank or media" -q`). Expected: PASS.
- [ ] **Step 5: Commit** — `feat(providers): configurable base_url for openrouter embed/llm, deepinfra rerank, media`.

### Task 6: `provider-sim` FastAPI app

**Files:**
- Create: `staging/provider_sim/app.py`
- Create: `staging/provider_sim/Dockerfile`
- Create: `staging/provider_sim/requirements.txt` (`fastapi`, `uvicorn`)
- Test: `tests/test_provider_sim.py` (in-process via `httpx.ASGITransport` — no docker needed)

**Simulator dialect contract** (from code research; each row lists ONLY what production code parses):

| Route | Serves | Must return |
|---|---|---|
| `POST /api/v1/embeddings` | OpenRouter embed | `{"data": [{"index": i, "embedding": [768 floats]} for each input]}` — vector = deterministic hash of the input text (see below). (No separate Ollama `/v1/embeddings` route — staging config uses openrouter embeddings only; YAGNI.) |
| `POST /api/v1/chat/completions` | OpenRouter LLM + media | `{"choices": [{"message": {"content": <str>}}]}`. If request content is a list containing `input_audio`/`video_url` parts → return canned transcript `"[transcript] <fixture marker>"`; if `response_format.type == "json_schema"` → content must be valid JSON matching a minimal enrichment object `{"summary": "...", "topics": [], "entities": []}` (check `doc_enrichment.py` for exact required keys during implementation) |
| `POST /v1/inference/{model}` | DeepInfra rerank | `{"scores": [<float per document>]}` — score = lexical-overlap of query/doc so ordering is meaningful and stable |
| `POST /extract`, `POST /describe` | DeepSeek OCR2 (multipart `file`) | `{"text": "[ocr] <deterministic text derived from file name/bytes-hash>"}` |
| `POST /api/chat` | Ollama vision/LLM | if `"stream": true`: NDJSON lines `{"message": {"content": <piece>}, "done": false}` ending with `{"message": {"content": ""}, "done": true}`; else single JSON `{"message": {"content": <str>}}` |
| `POST /hooks/sink` | webhook sink | 200; append received JSON to in-memory list |
| `GET /hooks/received` | test assertion helper | `{"events": [...]}` all payloads received |
| `POST /admin/reset` | between-test reset | clears sink + fault state |

**Deterministic embeddings** — must produce stable, content-correlated vectors so hybrid search ranks sensibly:

```python
import hashlib, math

def fake_embedding(text: str, dim: int = 768) -> list[float]:
    h = hashlib.sha256(text.lower().encode()).digest()
    vec = [(h[i % 32] / 255.0) - 0.5 for i in range(dim)]
    # mix in token-level signal so similar texts get similar vectors
    for tok in text.lower().split()[:64]:
        th = hashlib.md5(tok.encode()).digest()
        for j in range(dim):
            vec[j] += ((th[j % 16] / 255.0) - 0.5) * 0.1
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]
```

**Fault injection** — ASGI middleware reading `X-Sim-Fault`: `429` → respond 429 with `Retry-After: 0`; `timeout` → `await asyncio.sleep(...)` beyond client read timeout (make the sleep configurable via `X-Sim-Fault-Seconds`, default 10); `garbage` → 200 with body `"not json {"`. Also `POST /admin/fault {"route_prefix": "/api/v1/embeddings", "fault": "429", "times": 2}` to arm faults for requests that production code sends (which can't carry the header) — decrement per hit, then behave normally. This "fail N times then succeed" shape is what proves `call_with_retry` recovers.

- [ ] **Step 1: Write failing contract tests** (`tests/test_provider_sim.py`, in-process ASGI): one test per route asserting the exact minimal shape above; determinism test (`fake_embedding(t) == fake_embedding(t)`, differs for different t, unit norm); fault tests (armed 429 twice → third call 200; NDJSON stream parses line-by-line with final `done: true`).
- [ ] **Step 2: Verify failure**, implement `app.py` (single file, ~250 lines, stdlib + fastapi only).
- [ ] **Step 3: Run** — `pytest tests/test_provider_sim.py -v`. Expected: PASS.
- [ ] **Step 4: Fidelity check against recorded reality**: if `.evals/llm-traces/*.jsonl` exist locally, add one test that loads a real recorded response and asserts the simulator's response for the same route contains every key the recorded one has at the top level of the parsed path (guard with `pytest.mark.skipif` when no traces present).
- [ ] **Step 5: `staging/provider_sim/Dockerfile`** — `python:3.13-slim`, `pip install -r requirements.txt`, `CMD uvicorn app:app --host 0.0.0.0 --port 9999`.
- [ ] **Step 6: Commit** — `feat(staging): provider simulator speaking openrouter/deepinfra/ocr/ollama dialects`.

---

## Phase 4 — Staging stack + e2e suite

### Task 7: `docker-compose.staging.yml` + `config.staging.yaml` + comm-postgres seed

**Files:**
- Create: `docker-compose.staging.yml`
- Create: `config.staging.yaml` (committed; no secrets — sim ignores keys)
- Create: `staging/comm_postgres/init.sql`

- [ ] **Step 1: `config.staging.yaml`** — copy structure from `config.yaml.example`, set: `embeddings.provider: openrouter` + `embeddings.base_url: http://provider-sim:9999/api/v1`; `enrichment.provider: openrouter` + `enrichment.base_url: http://provider-sim:9999/api/v1`; `ocr.provider: deepseek_ocr2` + `ocr.base_url: http://provider-sim:9999`; `media.provider: openrouter` + `media.base_url: http://provider-sim:9999/api/v1`; `search.reranker.base_url: http://provider-sim:9999`; `event_hooks` → one hook to `http://provider-sim:9999/hooks/sink` for `document.indexed`; `tracing: {enabled: true, directory: /data/traces}`; `mcp.port: 7788`; **and the whole file MUST be written in sources-mode, not documents_root-mode.** Two constraints from `core/config.py` make this non-optional: (a) `documents_root` and `sources` cannot coexist — `load_config` hard-errors (`core/config.py:74-78`), so the container would fail to boot at Step 4's smoke test; (b) there is no top-level `sor:` config key in this codebase — the commented block in `config.yaml.example` (~line 165) is a `sources:` list entry. Concretely: define `sources:` with (1) a filesystem source for `/data/documents` carrying its own `scan:` sub-key (top-level `scan:` is not forwarded in sources-mode; `documents_root` is synthesized from the first filesystem source, `core/config.py:167-170`), and (2) a postgres source **named exactly `sor`** (required by `sor_query.py:95` for config-based DSN resolution) with a `tables:` spec matching `staging/comm_postgres/init.sql` (per `sources/postgres.py` TableSpec: `source_type` (required — `TableSpec(**t)` fails loudly without it), `query`, `id_template`, `text_column`, TIMESTAMPTZ mtime column, direction/sender/channel metadata columns). Rationale, precisely: the SOR tools (`sor_schema`/`sor_query`) work off the DSN (config-resolved or `SOR_DSN` env fallback) and live `information_schema` introspection — they do NOT read `tables:`; the `tables:` spec is what lets the *scan flow index comm rows* as documents, which the staging e2e can then find via `file_search` with a comm-source filter.

- [ ] **Step 2: `staging/comm_postgres/init.sql`** — create the comm message table matching `sources/postgres.py` TableSpec expectations (text column, TIMESTAMPTZ mtime column, id column, direction/sender/channel metadata columns — confirm exact names against the commented `sources:` postgres `tables:` entry in `config.yaml.example` (~line 163) during implementation) + insert 5 fixture messages.

- [ ] **Step 3: `docker-compose.staging.yml`**

```yaml
name: doc-organizer-staging
services:
  doc-organizer-staging:
    build: .
    mem_limit: 4g
    ports: ["17788:7788"]
    environment:
      OPENROUTER_API_KEY: sim
      DEEPINFRA_API_KEY: sim
      API_KEY: staging-test-key
      COMM_DATA_STORE_DSN: postgresql://comm:comm@comm-postgres:5432/comm
      SOR_DSN: postgresql://comm:comm@comm-postgres:5432/comm
    volumes:
      - ./config.staging.yaml:/app/config.yaml:ro
      - staging-docs:/data/documents
      - staging-index:/data/index
      - staging-traces:/data/traces
    depends_on:
      provider-sim: {condition: service_started}
      comm-postgres: {condition: service_healthy}
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:7788/health')"]
      interval: 5s
      retries: 20
  provider-sim:
    build: ./staging/provider_sim
    ports: ["19999:9999"]
  comm-postgres:
    image: postgres:16-alpine
    environment: {POSTGRES_USER: comm, POSTGRES_PASSWORD: comm, POSTGRES_DB: comm}
    volumes: ["./staging/comm_postgres/init.sql:/docker-entrypoint-initdb.d/init.sql:ro"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U comm"]
      interval: 3s
      retries: 20
volumes:
  staging-docs: {}
  staging-index: {}
  staging-traces: {}
```

No external networks (unlike prod compose) — fully isolated; distinct host ports (17788/19999).

- [ ] **Step 4: Manual smoke** — `docker compose -f docker-compose.staging.yml up -d --build --wait && curl -s http://localhost:17788/health && curl -s http://localhost:19999/hooks/received && docker compose -f docker-compose.staging.yml down -v`. Expected: health 200, sink `{"events": []}`, clean teardown.
- [ ] **Step 5: Wire `collect_staging_traces` in `scripts/gate.py`** — `docker compose -f docker-compose.staging.yml cp doc-organizer-staging:/data/traces <run_dir>/traces` before teardown.
- [ ] **Step 6: Commit** — `feat(staging): isolated memory-capped staging compose stack`.

### Task 8: E2E suite — fixtures, client wrapper, scenarios

**Files:**
- Create: `tests/e2e/__init__.py`, `tests/e2e/conftest.py`, `tests/e2e/client.py`
- Create: `tests/e2e/test_lifecycle.py`, `tests/e2e/test_tools_browse.py`, `tests/e2e/test_tools_taxonomy.py`, `tests/e2e/test_tools_sor.py`, `tests/e2e/test_hooks_and_faults.py`
- Create fixtures: `tests/fixtures/e2e/` — `note.md`, `report.pdf`, `diagram.png` (copy from `test_vault/`), `clip.wav` (~5s, generate: `python -c` with `wave` module writing a 5s sine tone), `clip.mp4` (~5s, generate once with `ffmpeg -f lavfi -i testsrc=duration=5:size=64x64:rate=5 -f lavfi -i sine=duration=5 -shortest tests/fixtures/e2e/clip.mp4`; commit the binary, ~50KB)

- [ ] **Step 1: `tests/e2e/client.py`** — the coverage-recording MCP client:

```python
"""MCP client wrapper for e2e: real streamable-HTTP transport + tool-coverage recording."""
import json, os
from pathlib import Path
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

E2E_BASE = os.environ.get("E2E_BASE_URL", "http://localhost:17788")
E2E_API_KEY = os.environ.get("E2E_API_KEY", "staging-test-key")
COVERAGE_FILE = Path(os.environ.get("E2E_COVERAGE_FILE", ".evals/e2e-tool-coverage.jsonl"))

class RecordingSession:
    def __init__(self, session: ClientSession, test_name: str):
        self._s, self._test = session, test_name

    async def call_tool(self, name: str, arguments: dict):
        result = await self._s.call_tool(name, arguments)
        COVERAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with COVERAGE_FILE.open("a") as f:
            f.write(json.dumps({"tool": name, "test": self._test}) + "\n")
        return result

    async def list_tools(self):
        return await self._s.list_tools()
```

`tests/e2e/conftest.py`: an `mcp_session` async fixture opening `streamablehttp_client(f"{E2E_BASE}/", headers={"Authorization": f"Bearer {E2E_API_KEY}"})` → `ClientSession` → yields `RecordingSession(session, request.node.name)`; an `api` fixture yielding an authenticated `httpx.AsyncClient(base_url=E2E_BASE)`; a session-scoped autouse fixture that deletes `COVERAGE_FILE` at session start and calls `POST {SIM}/admin/reset` (SIM = `http://localhost:19999`) between tests. **Verify during implementation**: the exact streamable-HTTP mount path (`/` vs `/mcp`) by checking `mcp_server.py:2614-2618` mounts — adjust `E2E_BASE` path accordingly.

- [ ] **Step 2: `test_lifecycle.py`** — the backbone scenario, ordered within one module: upload all 5 fixture docs via `POST /api/upload` → trigger `file_index_document` per doc (or `POST /api/index/document`) → poll `file_status` until idle/indexed (timeout 180s) → assert: `file_search` finds the md by content; search finds OCR text from the png (`"[ocr]"` marker); search finds `"[transcript]"` from clip.wav and clip.mp4 (proves media pipeline end-to-end); `file_get_chunk` + `file_get_doc_chunks` round-trip a search hit's `doc_id`/`loc`; `GET /api/documents/{doc_id}` downloads the original; `file_search` with `sort=recent` and a `filter` variant each return sane shapes.
- [ ] **Step 3: `test_tools_browse.py`** — `file_list_documents` (pagination), `file_recent`, `file_facets`, `file_folders`, `file_status`, `file_audit_log` (asserts the lifecycle docs appear with `created` events), `file_index_update` (pause → status reflects → resume).
- [ ] **Step 4: `test_tools_taxonomy.py`** — all 7: add → get → search → list → update → import (bulk) → delete; assert round-trip consistency.
- [ ] **Step 5: `test_tools_sor.py`** — `sor_schema` lists the seeded table; `sor_query` SELECT returns the 5 fixture rows; write statements rejected (read-only enforced).
- [ ] **Step 6: `test_hooks_and_faults.py`** — (a) after lifecycle indexing, `GET {SIM}/hooks/received` contains `document.indexed` events with `doc_id`, `chunks`; (b) arm `POST {SIM}/admin/fault {"route_prefix": "/api/v1/embeddings", "fault": "429", "times": 2}` → index a new doc → assert it still indexes (retry recovered) and `file_audit_log` shows no error; (c) arm `garbage` on enrichment route → assert doc indexes degraded (present in search, enrichment fields empty/default) rather than failing.
- [ ] **Step 7: Run the tier** — `make test-e2e` (brings the stack up/down via gate runner). Expected: all e2e PASS. Budget: whole tier under ~10 min.
- [ ] **Step 8: Commit** — `test(e2e): full tool-surface suite against staging stack`.

### Task 9: Tool-coverage matrix enforcement

**Files:**
- Create: `scripts/check_tool_coverage.py`
- Modify: `scripts/gate.py` (run the check right after the e2e tier, while the stack is still up)
- Test: `tests/test_check_tool_coverage.py`

- [ ] **Step 1: Failing tests** — `check_coverage(discovered={"a","b"}, covered={"a"})` returns the missing set `{"b"}`; empty missing set → ok; matrix rendering includes tool→tests mapping.
- [ ] **Step 2: Implement** — the script connects to the running staging MCP endpoint, calls `list_tools()` (source of truth — new tools automatically require coverage), reads `E2E_COVERAGE_FILE`, exits 1 with a named list of uncovered tools. `gate.py` calls it inside the compose window; failure fails the staging-e2e tier.
- [ ] **Step 3: Run `make test-e2e`** — expect 20/20 covered, exit 0. Temporarily comment one taxonomy test and re-run to see it fail with the tool named; restore.
- [ ] **Step 4: Commit** — `feat(gate): fail unless every MCP tool has e2e coverage`.

---

## Phase 5 — Live tier, report, docs

### Task 10: Live preflight + live tier wiring

**Files:**
- Create: `scripts/live_preflight.py`
- Modify: `scripts/gate.py` (run preflight before live tier; hard-fail, never skip)
- Test: `tests/test_live_preflight.py` (unit-test the check functions with monkeypatched probes)

Checks (each returns `(ok: bool, reason: str)`; ALL must pass):
1. `OPENROUTER_API_KEY` and `DEEPINFRA_API_KEY` set and non-empty.
2. Mac Mini OCR reachable: `GET {ocr.base_url}` (from the real `config.yaml`) connects within 5s (any HTTP response counts — reachability, not correctness).
3. Production indexer not mid-run: the prod heartbeat lives at `{index_root}/indexer.heartbeat` *inside* the prod container's `doc-organizer-data` volume, so the host can't read it directly. Read it via `docker exec doc-organizer cat /data/index/indexer.heartbeat` (exit code nonzero or missing file ⇒ no run ever ⇒ proceed). Heartbeat age < 120s ⇒ an indexing run is actively writing ⇒ fail with "prod indexer active; rerun later". Age ≥ 120s ⇒ proceed. If the prod container isn't running at all, proceed (nothing can contend for the Mac). Threshold is a named constant, documented in the file.
4. Comm-store Postgres: `psycopg` connect + `SELECT 1` via `COMM_DATA_STORE_DSN` within 5s.

Live tier execution = `pytest -m live` (existing `test_full_e2e_live.py`, `test_e2e_live.py`, `test_extractors_live.py`, renamed `test_multi_source_flow_live.py`) — these run in-process against real providers, which covers real OpenRouter/DeepInfra/OCR/Postgres full-workload per spec. Additionally append the media fixtures to the live path: extend `test_full_e2e_live.py` (or a new `tests/test_media_live.py`) to index `clip.wav` + `clip.mp4` through real transcription and assert the transcript is searchable.

- [ ] Steps: failing unit tests for each check (monkeypatched) → implement → wire into `gate.py` → `pytest tests/test_live_preflight.py -v` PASS → run the real live tier once end-to-end (`make test-live`, requires keys + Mac up; report cost counts from the report) → commit `feat(gate): live tier preflight — hard fail, no silent skips`.

### Task 11: Gate report

**Files:**
- Create: `scripts/gate_report.py`
- Test: `tests/test_gate_report.py` (feed synthetic junit XML + span JSONL + coverage JSONL in tmp dir; assert rendered markdown contains tier table, 20/20 matrix, a per-doc timeline, provider call counts)

Report inputs, all already produced by earlier tasks: `<run_dir>/*.xml` (junit per tier), `<run_dir>/traces/*.jsonl` (spans), `.evals/e2e-tool-coverage.jsonl`, `.evals/llm-traces/*.jsonl` (live tier spend: count records + sum latency per provider/model). Output: `<run_dir>/report.md` with: tier table (pass/fail/duration/test counts), tool-coverage matrix, per-document timeline (group spans by `trace_id`, order by `start_ns`, render stage→duration; flag ERROR status), live spend table. `gate.py` always calls it, even on failure — failed runs need the report most.

- [ ] Steps: failing tests with synthetic inputs → implement (~150 lines, stdlib only) → PASS → run `make gate-fast` and verify a report renders → commit `feat(gate): per-run markdown report with timelines and coverage matrix`.

### Task 12: Documentation + full gate proof

**Files:**
- Create: `docs/TESTING.md` (tier definitions, how to run each, staging stack anatomy, how to read a gate report, how to add a new MCP tool without breaking coverage enforcement, fault-injection how-to)
- Modify: `CLAUDE.md` (short "Testing" section: `make gate-fast` during dev, `make gate` before release; link to docs/TESTING.md)

- [ ] **Step 1: Write docs.**
- [ ] **Step 2: THE PROOF: run `make gate` end-to-end** (all five tiers, real keys, Mac up). Expected: full pass, report generated, live spend visible. If any tier fails, fix before calling this plan done — a gate that has never fully passed is not a gate.
- [ ] **Step 3: Commit** — `docs: testing + gate runbook`.

---

## Execution notes

- **Order:** Tasks 1–2 first (everything else reports through them). Then 3–4 (tracing) and 5–6 (simulator) can proceed in parallel. 7 needs 5+6; 8 needs 7; 9 needs 8; 10–12 last.
- **Test isolation rule (from memory, non-negotiable):** never run the suite inside the prod `doc-organizer` container. Tiers 1–3 run on the host; tier 4 runs against the memory-capped staging stack; tier 5's in-process live tests run on the host with small fixtures.
- **Cost control:** the live tier is small fixtures only; if a task needs repeated live runs while debugging, debug against the staging tier first — that ordering is the entire point of the gate.
- **Pre-existing test failures:** if tiers 2–3 surface failures unrelated to this work, report them to the user; do not fix silently inside this plan.
