# LLM Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a replayable benchmark for enrichment prompts that uses 100 real saved traces, Codex-authored gold labels, weighted scoring, and live model comparison against `openai/gpt-4.1-mini`.

**Architecture:** Add a small benchmarking package under `core/benchmarking/` for case loading, gold-label validation, scoring, replay, and reporting. Keep all sensitive benchmark data under gitignored `.evals/benchmarks/`, expose a thin CLI in `scripts/enrichment_benchmark.py`, and reuse production parsing/normalization logic from `doc_enrichment.py` so benchmark scoring matches runtime behavior.

**Tech Stack:** Python 3.13, pytest, httpx/OpenRouter, existing `doc_enrichment.py` normalization logic, JSON/JSONL, stdlib `argparse`, gitignored `.evals/benchmarks/` runtime data.

---

## Planned File Structure

**Files to Create**

- `core/benchmarking/__init__.py` — package exports for benchmark helpers.
- `core/benchmarking/models.py` — typed dict/dataclass schemas for case records, gold records, run records, and aggregate summaries.
- `core/benchmarking/cases.py` — trace loader, prompt metadata extraction, stratified case selection, case/gold/status file IO.
- `core/benchmarking/scoring.py` — field normalization and weighted scoring rules.
- `core/benchmarking/runner.py` — candidate-model replay orchestration, per-case result persistence, aggregate metrics.
- `core/benchmarking/reporting.py` — JSON/CSV/Markdown summary builders.
- `scripts/enrichment_benchmark.py` — CLI with `prepare-cases`, `show-case`, `status`, `run`, and `report` subcommands.
- `tests/test_benchmark_cases.py`
- `tests/test_benchmark_labeling.py`
- `tests/test_benchmark_scoring.py`
- `tests/test_benchmark_runner.py`
- `tests/test_benchmark_reporting.py`
- `tests/fixtures/benchmarks/sample_traces.jsonl`
- `tests/fixtures/benchmarks/cases/case_0001.json`
- `tests/fixtures/benchmarks/gold/case_0001.json`

**Files to Modify**

- `doc_enrichment.py` — expose one public parse/normalize helper for benchmark code reuse.
- `providers/llm/openrouter_llm.py` — expose one metadata-rich replay path that preserves existing `generate()` behavior.

**Files to Read for Context While Implementing**

- `docs/superpowers/specs/2026-04-21-llm-benchmark-design.md`
- `providers/llm/openrouter_llm.py`
- `doc_enrichment.py`
- `providers/llm/__init__.py`
- `tests/test_openrouter_trace_capture.py`
- `core/config.py`

### Task 1: Expose Reusable Enrichment Parsing Contract

**Files:**
- Modify: `doc_enrichment.py`
- Test: `tests/test_enrichment.py`

- [ ] **Step 1: Write failing parse-helper tests**

Add focused tests in `tests/test_enrichment.py` for a new public helper:

```python
def test_parse_enrichment_response_normalizes_valid_json():
    raw = '{"summary":"x","doc_type":["memo"],"entities_people":[],"entities_places":[],"entities_orgs":[],"entities_dates":[],"topics":["ops"],"keywords":["lease"],"key_facts":["rent due"],"suggested_tags":["housing"],"suggested_folder":"2-Housing","importance":0.7}'
    parsed = parse_enrichment_response(raw)
    assert parsed["enr_doc_type"] == "memo"
    assert parsed["enr_topics"] == "ops"
    assert parsed["enr_importance"] == "0.7"
```

- [ ] **Step 2: Run targeted test to verify failure**

Run: `python3 -m pytest tests/test_enrichment.py -k parse_enrichment_response -q`
Expected: FAIL with `NameError` or import error because helper does not exist yet.

- [ ] **Step 3: Add minimal public helper**

Implement `parse_enrichment_response(raw_response: str) -> dict[str, str]` in `doc_enrichment.py` by composing existing `_extract_json()` and `_normalize_enrichment()`:

```python
def parse_enrichment_response(raw_response: str) -> dict[str, str]:
    parsed = _extract_json(raw_response)
    return _normalize_enrichment(parsed)
```

Update `enrich_document()` to call this helper instead of duplicating parse/normalize steps.

- [ ] **Step 4: Run targeted test to verify pass**

Run: `python3 -m pytest tests/test_enrichment.py -k parse_enrichment_response -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add doc_enrichment.py tests/test_enrichment.py
git commit -m "refactor: expose enrichment parse helper"
```

### Task 2: Build Benchmark Schemas And Case Preparation

**Files:**
- Create: `core/benchmarking/__init__.py`
- Create: `core/benchmarking/models.py`
- Create: `core/benchmarking/cases.py`
- Create: `tests/test_benchmark_cases.py`
- Create: `tests/fixtures/benchmarks/sample_traces.jsonl`
- Modify: `scripts/enrichment_benchmark.py`

- [ ] **Step 1: Write failing case-loader and selector tests**

Add tests covering:

```python
def test_load_trace_rows_reads_saved_prompt_and_output(tmp_path):
    rows = load_trace_rows(fixture_path)
    assert rows[0].prompt.startswith("Extract metadata from this document")
    assert rows[0].baseline_response.startswith("{")

def test_prepare_cases_filters_smoke_rows_and_caps_to_limit(tmp_path):
    result = prepare_cases(trace_dir=fixture_dir, out_dir=tmp_path, limit=2, seed=7)
    assert result.selected_count == 2
    assert "smoke" not in {case["difficulty"] for case in result.cases}
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `python3 -m pytest tests/test_benchmark_cases.py -q`
Expected: FAIL because benchmark modules and fixtures do not exist yet.

- [ ] **Step 3: Implement schema and case preparation**

Create lightweight dataclasses or `TypedDict` shapes in `core/benchmarking/models.py`:

```python
@dataclass
class BenchmarkCase:
    case_id: str
    prompt: str
    baseline_response: str
    title: str
    source_type: str
    category: str
    difficulty: str
    trace_file: str
    trace_line: int
```

Implement in `core/benchmarking/cases.py`:

- trace JSONL loading
- prompt metadata extraction
- synthetic/smoke filtering
- deterministic stratified selection by `seed`
- case file and manifest writing under `.evals/benchmarks/cases/`

Add CLI subcommand skeleton in `scripts/enrichment_benchmark.py`:

```python
prepare = subparsers.add_parser("prepare-cases")
prepare.add_argument("--trace-dir", default=".evals/llm-traces")
prepare.add_argument("--bench-dir", default=".evals/benchmarks")
prepare.add_argument("--limit", type=int, default=100)
prepare.add_argument("--seed", type=int, default=42)
```

- [ ] **Step 4: Run targeted tests to verify pass**

Run: `python3 -m pytest tests/test_benchmark_cases.py -q`
Expected: PASS

- [ ] **Step 5: Smoke-test case preparation CLI**

Run: `python3 scripts/enrichment_benchmark.py prepare-cases --trace-dir tests/fixtures/benchmarks --bench-dir /tmp/llm-bench --limit 2 --seed 7`
Expected: prints `Prepared 2 cases` and creates `/tmp/llm-bench/cases/manifest.json`

- [ ] **Step 6: Commit**

```bash
git add core/benchmarking/__init__.py core/benchmarking/models.py core/benchmarking/cases.py scripts/enrichment_benchmark.py tests/test_benchmark_cases.py tests/fixtures/benchmarks/sample_traces.jsonl
git commit -m "feat: add benchmark case preparation"
```

### Task 3: Add Gold Record Validation And Labeling Workflow

**Files:**
- Modify: `core/benchmarking/models.py`
- Modify: `core/benchmarking/cases.py`
- Modify: `scripts/enrichment_benchmark.py`
- Create: `tests/test_benchmark_labeling.py`
- Create: `tests/fixtures/benchmarks/cases/case_0001.json`
- Create: `tests/fixtures/benchmarks/gold/case_0001.json`

- [ ] **Step 1: Write failing gold-schema and status tests**

Add tests covering:

```python
def test_write_gold_stub_creates_all_required_fields(tmp_path):
    path = write_gold_stub(case, bench_dir=tmp_path)
    data = json.loads(path.read_text())
    assert data["canonical"]["summary"] == ""
    assert "suggested_folder" in data["alternates"]

def test_labeling_status_counts_done_and_remaining(tmp_path):
    status = build_labeling_status(bench_dir=fixture_bench_dir)
    assert status["labeled"] == 1
    assert status["remaining"] >= 0
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `python3 -m pytest tests/test_benchmark_labeling.py -q`
Expected: FAIL because gold helpers and fixtures do not exist yet.

- [ ] **Step 3: Implement minimal labeling workflow**

Add gold helpers in `core/benchmarking/cases.py` to:

- create gold stub files with full field set
- validate canonical field presence
- store alternates for `suggested_tags` and `suggested_folder`
- generate `status/labeling_status.json`

Extend CLI with:

```python
show_case = subparsers.add_parser("show-case")
show_case.add_argument("--bench-dir", default=".evals/benchmarks")
show_case.add_argument("--case-id", required=True)

status = subparsers.add_parser("status")
status.add_argument("--bench-dir", default=".evals/benchmarks")
```

`show-case` should print:

- case metadata
- exact prompt
- baseline `gpt-4.1-mini` output
- path to gold file

`status` should print:

- total cases
- labeled count
- remaining count
- next unlabeled case id

- [ ] **Step 4: Run targeted tests to verify pass**

Run: `python3 -m pytest tests/test_benchmark_labeling.py -q`
Expected: PASS

- [ ] **Step 5: Smoke-test labeling commands**

Run: `python3 scripts/enrichment_benchmark.py show-case --bench-dir tests/fixtures/benchmarks --case-id case_0001`
Expected: prints prompt, baseline response, and gold path

Run: `python3 scripts/enrichment_benchmark.py status --bench-dir tests/fixtures/benchmarks`
Expected: prints labeled and remaining counts

- [ ] **Step 6: Commit**

```bash
git add core/benchmarking/models.py core/benchmarking/cases.py scripts/enrichment_benchmark.py tests/test_benchmark_labeling.py tests/fixtures/benchmarks/cases/case_0001.json tests/fixtures/benchmarks/gold/case_0001.json
git commit -m "feat: add benchmark gold-label workflow"
```

### Task 4: Add Metadata-Rich OpenRouter Replay Path

**Files:**
- Modify: `providers/llm/openrouter_llm.py`
- Create: `tests/test_benchmark_runner.py`

- [ ] **Step 1: Write failing replay-client tests**

Add tests for a new provider method that returns raw response metadata while preserving current `generate()` behavior:

```python
def test_generate_with_metadata_returns_content_usage_and_latency(tmp_path):
    result = generator.generate_with_metadata("hello", max_tokens=77)
    assert result["content"] == '{"summary":"ok"}'
    assert result["response"]["usage"]["total_tokens"] == 42
    assert result["latency_ms"] >= 0
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `python3 -m pytest tests/test_benchmark_runner.py -k generate_with_metadata -q`
Expected: FAIL because method does not exist yet.

- [ ] **Step 3: Implement provider replay helper**

Refactor `providers/llm/openrouter_llm.py` so one internal request path returns a metadata dict:

```python
{
    "content": content.strip(),
    "request": trace_request,
    "response": data,
    "latency_ms": latency_ms,
}
```

Implementation rules:

- keep retry behavior unchanged
- keep trace capture behavior unchanged
- keep `generate()` public contract unchanged by delegating to the new helper
- preserve HTTP and timeout exceptions

- [ ] **Step 4: Run targeted tests to verify pass**

Run: `python3 -m pytest tests/test_benchmark_runner.py -k generate_with_metadata -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add providers/llm/openrouter_llm.py tests/test_benchmark_runner.py
git commit -m "refactor: expose openrouter replay metadata"
```

### Task 5: Implement Benchmark Runner And Scoring Engine

**Files:**
- Create: `core/benchmarking/scoring.py`
- Create: `core/benchmarking/runner.py`
- Modify: `scripts/enrichment_benchmark.py`
- Modify: `tests/test_benchmark_runner.py`
- Create: `tests/test_benchmark_scoring.py`

- [ ] **Step 1: Write failing scoring tests**

Add field-level tests for the agreed weights and rules:

```python
def test_score_folder_accepts_configured_alternate():
    gold = {"canonical": {"suggested_folder": "2-Housing/Applications"}, "alternates": {"suggested_folder": ["2-Housing/Rentals"]}}
    pred = {"enr_suggested_folder": "2-Housing/Rentals"}
    assert score_case(pred, gold).field_scores["suggested_folder"] == 1.0

def test_parse_failure_scores_zero():
    result = score_failed_case(error="json_parse_error")
    assert result.total_score == 0.0
    assert result.reliability["parse_failed"] is True
```

- [ ] **Step 2: Run scoring tests to verify failure**

Run: `python3 -m pytest tests/test_benchmark_scoring.py -q`
Expected: FAIL because scoring module does not exist yet.

- [ ] **Step 3: Implement scoring engine**

Implement in `core/benchmarking/scoring.py`:

- default field weights matching spec
- set-overlap scoring for `doc_type`, `topics`, `keywords`, `suggested_tags`
- alternate acceptance and parent-branch partial credit for `suggested_folder`
- numeric distance scoring for `importance`
- exact-normalized entity/date matching
- lightweight summary scoring
- aggregate weighted total

Use the public parser from `doc_enrichment.py` so model outputs normalize the same way production does:

```python
normalized = parse_enrichment_response(raw_output)
```

- [ ] **Step 4: Write failing runner tests**

Add tests in `tests/test_benchmark_runner.py` for:

```python
def test_run_benchmark_persists_per_case_results_and_summary(tmp_path):
    run = run_benchmark(bench_dir=fixture_bench_dir, model="openai/gpt-4.1-mini", run_id="baseline", replay_client=fake_client)
    assert (tmp_path / "runs" / "baseline" / "per_case.jsonl").exists()
    assert run.summary["case_count"] == 1
```

- [ ] **Step 5: Run runner tests to verify failure**

Run: `python3 -m pytest tests/test_benchmark_runner.py -q`
Expected: FAIL because runner does not exist yet.

- [ ] **Step 6: Implement runner**

Implement in `core/benchmarking/runner.py`:

- load cases and gold labels
- call `OpenRouterGenerator(...).generate_with_metadata(...)`
- persist raw output and normalized output
- catch timeouts/HTTP/parse failures per case
- write `runs/<run_id>/per_case.jsonl`
- write `runs/<run_id>/summary.json`

Add CLI command:

```python
run_cmd = subparsers.add_parser("run")
run_cmd.add_argument("--bench-dir", default=".evals/benchmarks")
run_cmd.add_argument("--model", required=True)
run_cmd.add_argument("--run-id", required=True)
run_cmd.add_argument("--max-cases", type=int)
```

- [ ] **Step 7: Run scoring and runner tests to verify pass**

Run: `python3 -m pytest tests/test_benchmark_scoring.py tests/test_benchmark_runner.py -q`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add core/benchmarking/scoring.py core/benchmarking/runner.py scripts/enrichment_benchmark.py tests/test_benchmark_scoring.py tests/test_benchmark_runner.py
git commit -m "feat: add benchmark runner and scoring"
```

### Task 6: Implement Reporting

**Files:**
- Create: `core/benchmarking/reporting.py`
- Modify: `scripts/enrichment_benchmark.py`
- Create: `tests/test_benchmark_reporting.py`

- [ ] **Step 1: Write failing report tests**

Add tests for JSON/CSV/Markdown outputs:

```python
def test_build_report_writes_leaderboard_and_field_breakdown(tmp_path):
    paths = write_reports(run_dir=fixture_run_dir)
    assert paths["json"].exists()
    assert paths["csv"].exists()
    assert "overall_score" in paths["json"].read_text()
    assert "| model | overall_score |" in paths["markdown"].read_text()
```

- [ ] **Step 2: Run reporting tests to verify failure**

Run: `python3 -m pytest tests/test_benchmark_reporting.py -q`
Expected: FAIL because reporting module does not exist yet.

- [ ] **Step 3: Implement report builders**

Implement in `core/benchmarking/reporting.py`:

- summary JSON writer
- field-score CSV writer
- Markdown leaderboard with:
  - overall weighted score
  - success rate
  - parse failure rate
  - latency p50/p95
  - optional token/cost columns
  - worst-case examples

Add CLI command:

```python
report_cmd = subparsers.add_parser("report")
report_cmd.add_argument("--bench-dir", default=".evals/benchmarks")
report_cmd.add_argument("--run-id", required=True)
```

- [ ] **Step 4: Run reporting tests to verify pass**

Run: `python3 -m pytest tests/test_benchmark_reporting.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/benchmarking/reporting.py scripts/enrichment_benchmark.py tests/test_benchmark_reporting.py
git commit -m "feat: add benchmark reporting"
```

### Task 7: Full Fixture Verification

**Files:**
- Verify only

- [ ] **Step 1: Run full benchmark test slice**

Run: `python3 -m pytest tests/test_enrichment.py tests/test_benchmark_cases.py tests/test_benchmark_labeling.py tests/test_benchmark_scoring.py tests/test_benchmark_runner.py tests/test_benchmark_reporting.py -q`
Expected: PASS

- [ ] **Step 2: Verify CLI help and fixture workflow**

Run: `python3 scripts/enrichment_benchmark.py --help`
Expected: shows `prepare-cases`, `show-case`, `status`, `run`, `report`

Run: `python3 scripts/enrichment_benchmark.py prepare-cases --trace-dir tests/fixtures/benchmarks --bench-dir /tmp/llm-bench --limit 2 --seed 7`
Expected: prints `Prepared 2 cases`

Run: `python3 scripts/enrichment_benchmark.py run --bench-dir /tmp/llm-bench --model openai/gpt-4.1-mini --run-id fixture-baseline --max-cases 1`
Expected: prints completion summary and writes `/tmp/llm-bench/runs/fixture-baseline/summary.json`

- [ ] **Step 3: Review git diff**

Run: `git status --short`
Expected: only benchmark code, tests, and fixtures changed.

### Task 8: Bootstrap Real Benchmark Dataset

**Files:**
- Runtime only under `.evals/benchmarks/`

- [ ] **Step 1: Generate real 100-case benchmark set**

Run: `python3 scripts/enrichment_benchmark.py prepare-cases --trace-dir .evals/llm-traces --bench-dir .evals/benchmarks --limit 100 --seed 42`
Expected: prints `Prepared 100 cases` and writes `.evals/benchmarks/cases/manifest.json`

- [ ] **Step 2: Verify case mix**

Run: `python3 scripts/enrichment_benchmark.py status --bench-dir .evals/benchmarks`
Expected: prints `total=100`, `labeled=0`, `remaining=100`, and next case id

- [ ] **Step 3: Start labeling pass**

Run: `python3 scripts/enrichment_benchmark.py show-case --bench-dir .evals/benchmarks --case-id case_0001`
Expected: prints first case prompt, baseline output, and gold file path for manual labeling

- [ ] **Step 4: Baseline replay smoke test after a few labels exist**

Run: `python3 scripts/enrichment_benchmark.py run --bench-dir .evals/benchmarks --model openai/gpt-4.1-mini --run-id baseline-4.1-mini --max-cases 5`
Expected: writes run artifacts under `.evals/benchmarks/runs/baseline-4.1-mini/`

- [ ] **Step 5: Summarize residual risks**

Record:

- benchmark quality depends on label quality
- folder partial-credit heuristics may need tuning after first run
- token/cost availability depends on provider response metadata
