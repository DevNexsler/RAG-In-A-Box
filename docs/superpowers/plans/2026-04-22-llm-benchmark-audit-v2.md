## Goal

Add a separate benchmark audit v2 path under `.evals/benchmarks/audit/` without changing current benchmark behavior. Seed 30 audit cases from the existing benchmark corpus, keep audit labels blank/manual-only, and add richer scoring/reporting structure for future reruns.

## File Map

- `scripts/enrichment_benchmark.py`
  - Add audit-aware CLI commands or flags for seeding, status, show-case, run, and report.
- `core/benchmarking/models.py`
  - Add audit-specific dataclasses for rubric metadata and richer run summaries if needed.
- `core/benchmarking/cases.py`
  - Add audit directory helpers, 30-case seeding from existing corpus, blank manual-only gold stubs, and audit status handling.
- `core/benchmarking/scoring.py`
  - Add v2 audit score computation: `extraction_core`, `filing_taxonomy`, `summary_quality`, `overall_composite`.
  - Keep v1 scoring intact for backward compatibility.
- `core/benchmarking/runner.py`
  - Add optional audit-mode scoring path and expanded run summary fields.
- `core/benchmarking/reporting.py`
  - Add audit report sections for composite/subscores and preserve current report format for v1.
- `tests/test_benchmark_cases.py`
  - Add audit seeding/status/show-case coverage.
- `tests/test_benchmark_scoring.py`
  - Add audit v2 scoring tests, especially summary rubric handling and subscore rollups.
- `tests/test_benchmark_runner.py`
  - Add audit-mode run summary tests without live model calls.
- `tests/test_benchmark_reporting.py`
  - Add audit report content/serialization coverage.

## Task 1. Add audit data model and case seeding

Write tests first:
- Audit seed command creates `.evals/benchmarks/audit/cases`, `gold`, `status`, `runs`.
- Audit seed selects 30 cases from existing corpus and writes manifest.
- Audit gold stubs are blank/manual-only and do not copy old gold labels.
- Audit status counts labeled/unlabeled cases correctly.

Implement:
- Add audit path helpers and manifest format.
- Seed from existing benchmark `cases/` corpus, not raw trace archive.
- Stratify selection across categories/difficulty where possible.
- Gold stub includes `label_source: "manual_audit"` and empty `summary_rubric`.

Verify:
- Focused cases/labeling tests pass.

Commit:
- `feat: add benchmark audit corpus scaffolding`

## Task 2. Add audit v2 scoring

Write tests first:
- Audit score returns separate `extraction_core`, `filing_taxonomy`, `summary_quality`, `overall_composite`.
- Summary score uses rubric fields, not lexical overlap.
- Folder/tag scores live only inside filing taxonomy subscore.
- Missing rubric fields fail clearly.
- Current v1 scoring behavior stays unchanged.

Implement:
- Keep existing `score_case()` path for v1.
- Add separate audit scoring entrypoint for v2.
- Use fixed composite weights:
  - extraction_core `0.7`
  - filing_taxonomy `0.2`
  - summary_quality `0.1`

Verify:
- Focused scoring tests pass.

Commit:
- `feat: add audit benchmark scoring`

## Task 3. Add audit runner/reporting/CLI

Write tests first:
- Audit run writes summary/per-case structure with all three subscores plus composite.
- Audit report JSON/CSV/Markdown contain audit-specific metrics.
- CLI exposes audit seed/status/show-case/run/report paths without breaking current commands.

Implement:
- Extend runner to choose v1 vs audit-v2 scoring based on benchmark path or explicit mode.
- Extend reporting to serialize audit summary fields and leaderboard rows.
- Add audit CLI commands or flags with clear output paths.

Verify:
- Runner/reporting/CLI tests pass.
- Existing benchmark tests still pass.

Commit:
- `feat: add audit benchmark cli and reports`

## Task 4. Final verification

Run:
- `PYTHONPATH=. pytest -q tests/test_benchmark_cases.py tests/test_benchmark_labeling.py tests/test_benchmark_scoring.py tests/test_benchmark_runner.py tests/test_benchmark_reporting.py`

Then smoke local audit scaffold only:
- Seed audit corpus into `.evals/benchmarks/audit/`
- Check `status`
- Show one audit case

Do not run live model benchmarks in this task.
