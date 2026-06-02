# LLM Benchmark Hard Cases Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular LLM benchmark framework that preserves the existing enrichment benchmark and adds hard-case mining from real `.evals/llm-traces` logs.

**Architecture:** Keep existing enrichment benchmark behavior as the first task module. Add small, focused modules for task registration, suite path handling, trace mining, hard-case materialization, provider-failure classification, and suite-aware reporting. Real prompts, responses, labels, and run outputs stay under gitignored `.evals/`.

**Tech Stack:** Python 3.13, pytest, stdlib `argparse`, JSON/JSONL, existing OpenRouter replay client, existing enrichment scorer.

---

## File Structure

Create:

- `core/benchmarking/tasks/__init__.py` — task registry exports.
- `core/benchmarking/tasks/base.py` — small task protocol and task lookup helpers.
- `core/benchmarking/tasks/enrichment.py` — enrichment task adapter wrapping current scoring/loading behavior.
- `core/benchmarking/mining.py` — trace metadata extraction, hard-flag scoring, prompt dedupe, hard-case selection.
- `tests/test_benchmark_tasks.py` — task registry and backward-compat tests.
- `tests/test_benchmark_mining.py` — trace mining and hard-case selection tests.
- `tests/fixtures/benchmarks/hard_traces.jsonl` — synthetic hard traces only.

Modify:

- `core/benchmarking/models.py` — add metadata dataclasses for suites, mined traces, hard-case selections, failure classifications.
- `core/benchmarking/cases.py` — add suite path helpers and hard-case materialization while preserving old layout.
- `core/benchmarking/runner.py` — accept task/suite options, use task adapter, keep current positional behavior.
- `core/benchmarking/reporting.py` — include hard-case and provider-failure breakdowns when present.
- `scripts/enrichment_benchmark.py` — add `--task`, `--suite`, `mine-hard`, and suite-aware command behavior.
- `tests/test_benchmark_cases.py`
- `tests/test_benchmark_runner.py`
- `tests/test_benchmark_reporting.py`

Do not modify real `.evals` data in tracked changes.

Before editing code symbols, follow `AGENTS.md`: run `gitnexus_impact` for every function/class/method being edited and report risk. Before each commit, run `gitnexus_detect_changes`.

## Task 1: Task Registry And Suite Paths

**Files:**
- Create: `core/benchmarking/tasks/__init__.py`
- Create: `core/benchmarking/tasks/base.py`
- Create: `core/benchmarking/tasks/enrichment.py`
- Modify: `core/benchmarking/models.py`
- Modify: `core/benchmarking/cases.py`
- Test: `tests/test_benchmark_tasks.py`
- Test: `tests/test_benchmark_cases.py`

- [ ] **Step 1: Run impact checks**

Run GitNexus impact checks before edits:

```text
impact(target="BenchmarkCase", direction="upstream")
impact(target="load_case", direction="upstream")
impact(target="prepare_cases", direction="upstream")
```

Expected: no HIGH/CRITICAL risk. If HIGH/CRITICAL appears, stop and report before editing.

- [ ] **Step 2: Write failing task registry tests**

Add `tests/test_benchmark_tasks.py`:

```python
from core.benchmarking.tasks import get_task


def test_get_task_returns_enrichment_task():
    task = get_task("enrichment")

    assert task.name == "enrichment"
    assert task.default_score_mode == "standard"


def test_get_task_rejects_unknown_task():
    try:
        get_task("missing")
    except ValueError as exc:
        assert "unknown benchmark task" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

Extend `tests/test_benchmark_cases.py`:

```python
from core.benchmarking.cases import resolve_bench_path


def test_resolve_bench_path_preserves_legacy_layout(tmp_path):
    assert resolve_bench_path(bench_dir=tmp_path, task="enrichment", suite="standard") == tmp_path


def test_resolve_bench_path_uses_nested_layout_for_nonstandard_suite(tmp_path):
    assert resolve_bench_path(bench_dir=tmp_path, task="enrichment", suite="hard") == (
        tmp_path / "tasks" / "enrichment" / "hard"
    )
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
python3 -m pytest tests/test_benchmark_tasks.py tests/test_benchmark_cases.py -q
```

Expected: FAIL because task modules and `resolve_bench_path` do not exist.

- [ ] **Step 4: Implement minimal task registry**

Create `core/benchmarking/tasks/base.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


ScoreFunction = Callable[[str, Mapping[str, Any]], tuple[dict[str, str], Any]]


@dataclass(frozen=True)
class BenchmarkTask:
    name: str
    default_score_mode: str
    score_modes: frozenset[str]
    score_raw_output: Callable[[str, Mapping[str, Any], str], tuple[dict[str, str], Any]]
```

Create `core/benchmarking/tasks/enrichment.py`:

```python
from __future__ import annotations

from typing import Any, Mapping

from core.benchmarking.scoring import score_audit_raw_case, score_raw_case
from core.benchmarking.tasks.base import BenchmarkTask


def score_enrichment_output(
    raw_output: str,
    gold: Mapping[str, Any],
    score_mode: str,
) -> tuple[dict[str, str], Any]:
    if score_mode == "audit":
        return score_audit_raw_case(raw_output, gold)
    if score_mode == "standard":
        return score_raw_case(raw_output, gold)
    raise ValueError(f"unsupported enrichment score_mode: {score_mode}")


ENRICHMENT_TASK = BenchmarkTask(
    name="enrichment",
    default_score_mode="standard",
    score_modes=frozenset({"standard", "audit"}),
    score_raw_output=score_enrichment_output,
)
```

Create `core/benchmarking/tasks/__init__.py`:

```python
from __future__ import annotations

from core.benchmarking.tasks.base import BenchmarkTask
from core.benchmarking.tasks.enrichment import ENRICHMENT_TASK

_TASKS = {ENRICHMENT_TASK.name: ENRICHMENT_TASK}


def get_task(name: str = "enrichment") -> BenchmarkTask:
    try:
        return _TASKS[name]
    except KeyError as exc:
        raise ValueError(f"unknown benchmark task: {name}") from exc


__all__ = ["BenchmarkTask", "get_task"]
```

Add `resolve_bench_path` in `core/benchmarking/cases.py`:

```python
def resolve_bench_path(
    *,
    bench_dir: str | Path,
    task: str = "enrichment",
    suite: str = "standard",
) -> Path:
    root = Path(bench_dir)
    if task == "enrichment" and suite == "standard":
        return root
    return root / "tasks" / task / suite
```

- [ ] **Step 5: Run tests to verify pass**

Run:

```bash
python3 -m pytest tests/test_benchmark_tasks.py tests/test_benchmark_cases.py -q
```

Expected: PASS.

- [ ] **Step 6: Run change detection and commit**

Run:

```text
gitnexus_detect_changes(scope="staged")
```

Then:

```bash
git add core/benchmarking/tasks core/benchmarking/models.py core/benchmarking/cases.py tests/test_benchmark_tasks.py tests/test_benchmark_cases.py
git commit -m "feat: add benchmark task registry"
```

## Task 2: Trace Mining Metadata

**Files:**
- Create: `core/benchmarking/mining.py`
- Modify: `core/benchmarking/models.py`
- Create: `tests/test_benchmark_mining.py`
- Create: `tests/fixtures/benchmarks/hard_traces.jsonl`

- [ ] **Step 1: Run impact checks**

Run:

```text
impact(target="TraceRow", direction="upstream")
impact(target="core/benchmarking/models.py", direction="upstream")
```

Expected: no HIGH/CRITICAL risk.

- [ ] **Step 2: Write failing mining tests**

Create `tests/fixtures/benchmarks/hard_traces.jsonl` with synthetic rows only:

```json
{"ts":"2026-05-01T00:00:00+00:00","provider":"openrouter","model":"openai/gpt-4.1-mini","success":true,"latency_ms":15000,"request":{"payload":{"messages":[{"role":"system","content":"system"},{"role":"user","content":"Extract metadata from this document. Respond with ONLY valid JSON, no other text.\nThe context_* fields are required output keys; never omit them.\n\n## Available Tags\n- TenantCloud\n## Available Folders\n- 1-Projects/Rent-Collection/\n\nPRIMARY ITEM\nDocument title: Payment cleared notice\nDocument type: pg_message\n\nDocument text:\nHELLO PINEFIELD GROUP,\nA payment sent from Brianna Reaver is successfully cleared.\nhttps://tracking.example.com/noisy-link\nNEARBY SAME-CHANNEL CONTEXT CANDIDATES\nBEFORE MESSAGES\n[BEFORE 2026-05-01 source_message_id=abc] prior payment notice"}]}}},"response":{"choices":[{"message":{"content":"{\"summary\":\"Payment cleared\"}"}}]}}
{"ts":"2026-05-02T00:00:00+00:00","provider":"openrouter","model":"openai/gpt-4.1-mini","success":false,"latency_ms":200,"request":{"payload":{"messages":[{"role":"user","content":"Document title: Failed billing\nDocument type: pg_message\n\nDocument text:\nOpenRouter billing failed"}]}}},"error":{"type":"HTTPStatusError","status_code":402,"message":"Payment Required"}}
```

Create `tests/test_benchmark_mining.py`:

```python
from pathlib import Path

from core.benchmarking.mining import load_trace_metadata, score_hard_flags, select_hard_cases


def test_load_trace_metadata_extracts_safe_fields_only():
    rows = load_trace_metadata(Path("tests/fixtures/benchmarks/hard_traces.jsonl"))

    assert rows[0].trace_file == "hard_traces.jsonl"
    assert rows[0].trace_line == 1
    assert rows[0].title == "Payment cleared notice"
    assert rows[0].source_type == "pg_message"
    assert rows[0].prompt_hash
    assert rows[0].prompt_excerpt == ""


def test_score_hard_flags_detects_real_hard_patterns():
    row = load_trace_metadata(Path("tests/fixtures/benchmarks/hard_traces.jsonl"))[0]

    scored = score_hard_flags(row)

    assert "nearby_context" in scored.flags
    assert "taxonomy_bloat" in scored.flags
    assert "link_noise" in scored.flags
    assert "business_critical" in scored.flags
    assert "slow_success" in scored.flags
    assert scored.hard_score > 0


def test_select_hard_cases_keeps_provider_failures_separate():
    rows = load_trace_metadata(Path("tests/fixtures/benchmarks/hard_traces.jsonl"))

    selected = select_hard_cases(rows, limit=10)

    assert selected.hard_cases
    assert selected.provider_failure_cases
    assert selected.provider_failure_cases[0].failure_status_code == 402
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
python3 -m pytest tests/test_benchmark_mining.py -q
```

Expected: FAIL because `core.benchmarking.mining` does not exist.

- [ ] **Step 4: Add metadata dataclasses**

Modify `core/benchmarking/models.py`:

```python
@dataclass(frozen=True)
class TraceMetadata:
    trace_file: str
    trace_line: int
    timestamp: str
    provider: str
    model: str
    success: bool
    latency_ms: float | None
    title: str
    source_type: str
    prompt_length: int
    text_length: int
    prompt_hash: str
    response_looks_parseable: bool
    failure_type: str | None = None
    failure_status_code: int | None = None
    prompt_excerpt: str = ""


@dataclass(frozen=True)
class HardCaseCandidate:
    trace: TraceMetadata
    flags: tuple[str, ...]
    hard_score: int


@dataclass(frozen=True)
class HardCaseSelection:
    hard_cases: list[HardCaseCandidate]
    provider_failure_cases: list[TraceMetadata]
```

- [ ] **Step 5: Implement mining module**

Create `core/benchmarking/mining.py`.

Key functions:

```python
def load_trace_metadata(trace_path: str | Path) -> list[TraceMetadata]: ...
def score_hard_flags(row: TraceMetadata) -> HardCaseCandidate: ...
def select_hard_cases(rows: list[TraceMetadata], *, limit: int = 50) -> HardCaseSelection: ...
```

Implementation rules:

- Parse JSONL line by line.
- Extract user prompt from `request.payload.messages`.
- Extract title/type via regex.
- Compute `prompt_hash = sha256(prompt.encode()).hexdigest()`.
- Do not store prompt text or response text in `TraceMetadata`.
- Use empty `prompt_excerpt` by default to avoid leaking sensitive text.
- Detect provider failure from `success == false`, `error.type`, and `error.status_code`.
- Deduplicate by `prompt_hash`.

Flag thresholds:

```python
LONG_PROMPT_CHARS = 25_000
HUGE_PROMPT_CHARS = 250_000
SLOW_SUCCESS_MS = 10_000
VERY_SLOW_SUCCESS_MS = 25_000
```

- [ ] **Step 6: Run mining tests**

Run:

```bash
python3 -m pytest tests/test_benchmark_mining.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run `gitnexus_detect_changes(scope="staged")`, then:

```bash
git add core/benchmarking/models.py core/benchmarking/mining.py tests/test_benchmark_mining.py tests/fixtures/benchmarks/hard_traces.jsonl
git commit -m "feat: mine hard llm benchmark traces"
```

## Task 3: Hard Suite Materialization

**Files:**
- Modify: `core/benchmarking/cases.py`
- Modify: `core/benchmarking/mining.py`
- Test: `tests/test_benchmark_mining.py`
- Test: `tests/test_benchmark_cases.py`

- [ ] **Step 1: Run impact checks**

Run:

```text
impact(target="prepare_cases", direction="upstream")
impact(target="write_gold_stub", direction="upstream")
impact(target="build_labeling_status", direction="upstream")
impact(target="load_trace_rows", direction="upstream")
```

Expected: no HIGH/CRITICAL risk.

- [ ] **Step 2: Write failing materialization test**

Add to `tests/test_benchmark_mining.py`:

```python
import json

from core.benchmarking.mining import materialize_hard_suite


def test_materialize_hard_suite_writes_cases_manifest_and_gold_stubs(tmp_path):
    result = materialize_hard_suite(
        trace_dir=Path("tests/fixtures/benchmarks"),
        out_dir=tmp_path,
        task="enrichment",
        suite="hard",
        limit=5,
    )

    suite_dir = tmp_path / "tasks" / "enrichment" / "hard"
    manifest = json.loads((suite_dir / "cases" / "manifest.json").read_text())

    assert result.selected_count == 1
    assert manifest["suite"] == "hard"
    assert manifest["task"] == "enrichment"
    assert manifest["selection_flags"]
    assert (suite_dir / "cases" / "case_0001.json").is_file()
    assert (suite_dir / "gold" / "case_0001.json").is_file()
```

- [ ] **Step 3: Run test to verify failure**

Run:

```bash
python3 -m pytest tests/test_benchmark_mining.py::test_materialize_hard_suite_writes_cases_manifest_and_gold_stubs -q
```

Expected: FAIL because `materialize_hard_suite` does not exist.

- [ ] **Step 4: Implement materialization**

Add `materialize_hard_suite` to `core/benchmarking/mining.py`.

Behavior:

- Scan all `*.jsonl` in `trace_dir`.
- Select hard cases with `select_hard_cases`.
- Resolve output path using `resolve_bench_path`.
- Write `cases/case_XXXX.json`.
- Write `cases/manifest.json`.
- Write gold stubs using `write_gold_stub`.
- Write provider failures to `provider_failures.jsonl` instead of normal quality cases.
- Include trace refs and flags in manifest, but not full prompt text.

Important: benchmark case files under `.evals` may contain real prompts because `.evals` is gitignored. Synthetic tests must not contain private data.

- [ ] **Step 5: Run tests**

Run:

```bash
python3 -m pytest tests/test_benchmark_mining.py tests/test_benchmark_cases.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run `gitnexus_detect_changes(scope="staged")`, then:

```bash
git add core/benchmarking/cases.py core/benchmarking/mining.py tests/test_benchmark_mining.py tests/test_benchmark_cases.py
git commit -m "feat: materialize hard benchmark suites"
```

## Task 4: Suite-Aware Runner

**Files:**
- Modify: `core/benchmarking/runner.py`
- Modify: `core/benchmarking/tasks/enrichment.py`
- Test: `tests/test_benchmark_runner.py`

- [ ] **Step 1: Run impact checks**

Run:

```text
impact(target="run_benchmark", direction="upstream")
impact(target="_run_case", direction="upstream")
impact(target="_build_summary", direction="upstream")
impact(target="_validate_score_mode", direction="upstream")
```

Expected: no HIGH/CRITICAL risk.

- [ ] **Step 2: Write failing suite-aware runner test**

Add to `tests/test_benchmark_runner.py`:

```python
def test_run_benchmark_accepts_task_and_suite_nested_layout(tmp_path):
    fixture_bench_dir = tmp_path / "benchmarks"
    suite_dir = fixture_bench_dir / "tasks" / "enrichment" / "hard"
    suite_dir.mkdir(parents=True)
    _write_case_and_gold(suite_dir, case_id="case_0001")

    fake_client = FakeReplayClient(
        content='{"summary":"Lease renewal request.","doc_type":["lease"],"entities_people":[],"entities_places":[],"entities_orgs":[],"entities_dates":["2026-03-01"],"topics":["lease renewal"],"keywords":["renewal terms"],"key_facts":["Tenant requested renewal."],"suggested_tags":["lease"],"suggested_folder":"Housing/Leases","importance":0.8}'
    )

    run = run_benchmark(
        bench_dir=fixture_bench_dir,
        task="enrichment",
        suite="hard",
        model="openai/gpt-4.1-mini",
        run_id="hard-baseline",
        replay_client=fake_client,
    )

    assert run.run_dir == suite_dir / "runs" / "hard-baseline"
    assert run.summary["task"] == "enrichment"
    assert run.summary["suite"] == "hard"
```

- [ ] **Step 3: Run test to verify failure**

Run:

```bash
python3 -m pytest tests/test_benchmark_runner.py::test_run_benchmark_accepts_task_and_suite_nested_layout -q
```

Expected: FAIL because `run_benchmark` does not accept `task`/`suite`.

- [ ] **Step 4: Implement suite-aware runner**

Modify `run_benchmark` signature:

```python
def run_benchmark(
    *,
    bench_dir: str | Path,
    model: str,
    run_id: str,
    max_cases: int | None = None,
    replay_client: Any | None = None,
    score_mode: str | None = None,
    task: str = "enrichment",
    suite: str = "standard",
) -> BenchmarkRunResult:
```

Implementation:

- Resolve `bench_path = resolve_bench_path(bench_dir=bench_dir, task=task, suite=suite)`.
- Load task with `get_task(task)`.
- Default score mode to `task.default_score_mode`.
- Validate score mode against `task.score_modes`.
- Call `task.score_raw_output(...)` in `_run_case`.
- Include `task` and `suite` in `summary.json`.
- Preserve old behavior when caller omits task/suite.

- [ ] **Step 5: Run runner tests**

Run:

```bash
python3 -m pytest tests/test_benchmark_runner.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run `gitnexus_detect_changes(scope="staged")`, then:

```bash
git add core/benchmarking/runner.py core/benchmarking/tasks/enrichment.py tests/test_benchmark_runner.py
git commit -m "feat: run benchmark suites by task"
```

## Task 5: Hard And Failure Reporting

**Files:**
- Modify: `core/benchmarking/reporting.py`
- Test: `tests/test_benchmark_reporting.py`

- [ ] **Step 1: Run impact checks**

Run:

```text
impact(target="write_reports", direction="upstream")
impact(target="_build_report", direction="upstream")
impact(target="_build_markdown", direction="upstream")
impact(target="_validate_summary", direction="upstream")
```

Expected: no HIGH/CRITICAL risk.

- [ ] **Step 2: Write failing reporting tests**

Add to `tests/test_benchmark_reporting.py`:

```python
def test_build_report_includes_task_suite_and_hard_breakdown(tmp_path):
    run_dir = _write_run_artifacts(
        tmp_path,
        summary_overrides={
            "task": "enrichment",
            "suite": "hard",
            "hard_case_breakdown": {
                "huge_prompt": {"case_count": 2, "average_total_score": 0.4},
                "nearby_context": {"case_count": 1, "average_total_score": 0.8},
            },
            "provider_failure_breakdown": {
                "402": 3,
                "ConnectError": 1,
            },
        },
    )

    paths = write_reports(run_dir=run_dir)
    report = json.loads(paths["json"].read_text())
    markdown = paths["markdown"].read_text()

    assert report["summary"]["task"] == "enrichment"
    assert report["summary"]["suite"] == "hard"
    assert report["hard_case_breakdown"]["huge_prompt"]["case_count"] == 2
    assert report["provider_failure_breakdown"]["402"] == 3
    assert "## Hard Case Breakdown" in markdown
    assert "## Provider Failure Breakdown" in markdown
```

- [ ] **Step 3: Run test to verify failure**

Run:

```bash
python3 -m pytest tests/test_benchmark_reporting.py::test_build_report_includes_task_suite_and_hard_breakdown -q
```

Expected: FAIL because report omits those sections.

- [ ] **Step 4: Implement reporting additions**

Modify `reporting.py`:

- Include `task`, `suite`, `hard_case_breakdown`, and `provider_failure_breakdown` from summary if present.
- Add markdown sections only when breakdowns are present.
- Add CSV writers:
  - `hard_case_breakdown.csv`
  - `provider_failures.csv`

Do not require these fields for old summaries.

- [ ] **Step 5: Run reporting tests**

Run:

```bash
python3 -m pytest tests/test_benchmark_reporting.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run `gitnexus_detect_changes(scope="staged")`, then:

```bash
git add core/benchmarking/reporting.py tests/test_benchmark_reporting.py
git commit -m "feat: report hard llm benchmark breakdowns"
```

## Task 6: CLI Commands

**Files:**
- Modify: `scripts/enrichment_benchmark.py`
- Test: `tests/test_benchmark_reporting.py`
- Test: `tests/test_benchmark_mining.py`

- [ ] **Step 1: Run impact checks**

Run:

```text
impact(target="build_parser", direction="upstream")
impact(target="main", direction="upstream")
impact(target="_print_run", direction="upstream")
impact(target="_print_report", direction="upstream")
```

Expected: no HIGH/CRITICAL risk.

- [ ] **Step 2: Write failing CLI parser tests**

Add to `tests/test_benchmark_reporting.py`:

```python
def test_build_parser_registers_suite_aware_run_and_report():
    parser = build_parser()

    run_args = parser.parse_args([
        "run",
        "--task", "enrichment",
        "--suite", "hard",
        "--model", "openai/gpt-4.1-mini",
        "--run-id", "hard-baseline",
    ])
    report_args = parser.parse_args([
        "report",
        "--task", "enrichment",
        "--suite", "hard",
        "--run-id", "hard-baseline",
    ])
    mine_args = parser.parse_args([
        "mine-hard",
        "--task", "enrichment",
        "--suite", "hard",
        "--limit", "50",
    ])

    assert run_args.task == "enrichment"
    assert run_args.suite == "hard"
    assert report_args.suite == "hard"
    assert mine_args.command == "mine-hard"
```

- [ ] **Step 3: Run test to verify failure**

Run:

```bash
python3 -m pytest tests/test_benchmark_reporting.py::test_build_parser_registers_suite_aware_run_and_report -q
```

Expected: FAIL because parser lacks options.

- [ ] **Step 4: Add CLI options**

Modify `build_parser`:

- Add helper:

```python
def _add_task_suite_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", default="enrichment")
    parser.add_argument("--suite", default="standard")
```

- Apply to `run`, `report`, `status`, `show-case`, `prepare-cases` where useful.
- Add `mine-hard`:

```python
mine = subparsers.add_parser("mine-hard")
mine.add_argument("--trace-dir", default=".evals/llm-traces")
mine.add_argument("--bench-dir", default=".evals/benchmarks")
mine.add_argument("--task", default="enrichment")
mine.add_argument("--suite", default="hard")
mine.add_argument("--limit", type=int, default=50)
```

Modify `main`:

- `run` passes `task` and `suite` to `run_benchmark`.
- `report` resolves nested run dir using `resolve_bench_path`.
- `mine-hard` calls `materialize_hard_suite`.
- Existing commands without task/suite still use legacy standard layout.

- [ ] **Step 5: Run CLI tests**

Run:

```bash
python3 -m pytest tests/test_benchmark_reporting.py tests/test_benchmark_mining.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run `gitnexus_detect_changes(scope="staged")`, then:

```bash
git add scripts/enrichment_benchmark.py tests/test_benchmark_reporting.py tests/test_benchmark_mining.py
git commit -m "feat: add suite-aware benchmark cli"
```

## Task 7: End-To-End Synthetic Hard Suite Test

**Files:**
- Modify: `tests/test_benchmark_mining.py`
- Modify: `tests/test_benchmark_runner.py`
- Modify: `tests/test_benchmark_reporting.py`

- [ ] **Step 1: Write failing E2E test**

Add to `tests/test_benchmark_mining.py`:

```python
from core.benchmarking.runner import run_benchmark
from core.benchmarking.reporting import write_reports
from tests.test_benchmark_runner import FakeReplayClient


def test_synthetic_hard_suite_mine_run_report_e2e(tmp_path):
    materialize_hard_suite(
        trace_dir=Path("tests/fixtures/benchmarks"),
        out_dir=tmp_path,
        task="enrichment",
        suite="hard",
        limit=5,
    )
    suite_dir = tmp_path / "tasks" / "enrichment" / "hard"

    fake_client = FakeReplayClient(
        content='{"summary":"Payment cleared notice.","doc_type":["pg_message"],"entities_people":["Brianna Reaver"],"entities_places":[],"entities_orgs":["Pinefield Group"],"entities_dates":["2026-05-01"],"topics":["rent payment"],"keywords":["payment cleared"],"key_facts":["A payment cleared."],"suggested_tags":["TenantCloud"],"suggested_folder":"1-Projects/Rent-Collection/","importance":0.8}'
    )

    run = run_benchmark(
        bench_dir=tmp_path,
        task="enrichment",
        suite="hard",
        model="openai/gpt-4.1-mini",
        run_id="synthetic-hard",
        replay_client=fake_client,
    )
    paths = write_reports(run_dir=run.run_dir)

    assert run.summary["suite"] == "hard"
    assert paths["json"].is_file()
    assert (suite_dir / "provider_failures.jsonl").is_file()
```

- [ ] **Step 2: Run test to verify failure or pass**

Run:

```bash
python3 -m pytest tests/test_benchmark_mining.py::test_synthetic_hard_suite_mine_run_report_e2e -q
```

Expected: PASS if prior tasks are complete; otherwise fix integration gaps.

- [ ] **Step 3: Run benchmark test suite**

Run:

```bash
python3 -m pytest \
  tests/test_benchmark_cases.py \
  tests/test_benchmark_labeling.py \
  tests/test_benchmark_scoring.py \
  tests/test_benchmark_runner.py \
  tests/test_benchmark_reporting.py \
  tests/test_benchmark_mining.py \
  tests/test_benchmark_tasks.py \
  -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

Run `gitnexus_detect_changes(scope="staged")`, then:

```bash
git add tests/test_benchmark_mining.py tests/test_benchmark_runner.py tests/test_benchmark_reporting.py
git commit -m "test: cover hard benchmark end to end"
```

## Task 8: Real Trace Dry Run And Docs

**Files:**
- Modify: `docs/enrichment-benchmark-results-2026-05-01.md` if appropriate, or create a new non-sensitive results note.
- Modify: `README.md` only if benchmark commands need public docs.

- [ ] **Step 1: Run non-live full benchmark tests**

Run:

```bash
python3 -m pytest tests/test_benchmark_*.py -q
```

Expected: PASS.

- [ ] **Step 2: Dry-run hard mining on real traces**

Run:

```bash
python3 scripts/enrichment_benchmark.py mine-hard \
  --trace-dir .evals/llm-traces \
  --bench-dir .evals/benchmarks \
  --task enrichment \
  --suite hard \
  --limit 50
```

Expected:

- command exits `0`
- writes `.evals/benchmarks/tasks/enrichment/hard/cases/manifest.json`
- writes no tracked sensitive files
- prints selected case count and provider failure count

- [ ] **Step 3: Check generated files are ignored**

Run:

```bash
git status --short
git check-ignore -v .evals/benchmarks/tasks/enrichment/hard/cases/manifest.json
```

Expected:

- generated `.evals/...` files are ignored
- no real prompt/response payloads appear as tracked files

- [ ] **Step 4: Optional baseline hard run**

Only run if OpenRouter credits and time are available:

```bash
python3 scripts/enrichment_benchmark.py run \
  --bench-dir .evals/benchmarks \
  --task enrichment \
  --suite hard \
  --model openai/gpt-4.1-mini \
  --run-id baseline-hard-$(date +%Y%m%d)
```

Expected:

- command exits `0`
- writes run artifacts under ignored `.evals`
- report includes hard-case breakdown

- [ ] **Step 5: Update docs if command names changed**

If README or docs need command examples, add only non-sensitive examples.

- [ ] **Step 6: Run GitNexus detection and commit docs**

Run `gitnexus_detect_changes(scope="staged")`, then commit any tracked docs:

```bash
git add README.md docs/enrichment-benchmark-results-2026-05-01.md
git commit -m "docs: document hard llm benchmark workflow"
```

Skip this commit if no tracked docs changed.

## Final Verification

- [ ] Run benchmark unit suite:

```bash
python3 -m pytest tests/test_benchmark_*.py -q
```

- [ ] Run affected non-live suite:

```bash
python3 -m pytest \
  tests/test_benchmark_cases.py \
  tests/test_benchmark_labeling.py \
  tests/test_benchmark_scoring.py \
  tests/test_benchmark_runner.py \
  tests/test_benchmark_reporting.py \
  tests/test_openrouter_trace_capture.py \
  tests/test_enrichment.py \
  -q
```

- [ ] Verify real mined hard suite remains ignored:

```bash
git check-ignore -v .evals/benchmarks/tasks/enrichment/hard/cases/manifest.json
```

- [ ] Run `gitnexus_detect_changes(scope="all")` and review affected scope.

- [ ] If all tests pass, prepare PR summary with:
  - task registry added
  - hard trace mining added
  - suite-aware runner added
  - hard/failure reporting added
  - real trace dry-run counts
  - test output
