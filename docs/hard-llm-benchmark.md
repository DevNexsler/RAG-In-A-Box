# Hard LLM Benchmark

Hard benchmark suites replay difficult enrichment prompts mined from real trace
logs. Runtime data stays under gitignored `.evals/`; only the framework, tests,
and synthetic fixtures are committed.

## Mine Cases

```bash
python3 scripts/enrichment_benchmark.py mine-hard \
  --trace-dir .evals/llm-traces \
  --bench-dir .evals/benchmarks \
  --task enrichment \
  --suite hard \
  --limit 50
```

This is deterministic local parsing. It does not call an LLM. It scores traces
for hard-case signals such as large prompts, taxonomy bloat, nearby context,
link noise, slow successes, parse-suspect outputs, and provider failures.

Generated files:

- `.evals/benchmarks/tasks/enrichment/hard/cases/`
- `.evals/benchmarks/tasks/enrichment/hard/gold/`
- `.evals/benchmarks/tasks/enrichment/hard/cases/manifest.json`
- `.evals/benchmarks/tasks/enrichment/hard/provider_failures.jsonl`

## Label Gold

Open each generated gold stub and fill expected enrichment fields before using
the suite for model comparison. Provider-failure traces are preserved separately
for reliability analysis and are not normal scored cases.

## Run Suite

```bash
python3 scripts/enrichment_benchmark.py run \
  --bench-dir .evals/benchmarks \
  --task enrichment \
  --suite hard \
  --model openai/gpt-4.1-mini \
  --run-id hard-4.1-mini
```

This replays prompts through the configured model provider and writes run
artifacts under:

```text
.evals/benchmarks/tasks/enrichment/hard/runs/hard-4.1-mini/
```

## Report

```bash
python3 scripts/enrichment_benchmark.py report \
  --bench-dir .evals/benchmarks \
  --task enrichment \
  --suite hard \
  --run-id hard-4.1-mini
```

Reports include aggregate scoring plus hard-case and provider-failure
breakdowns:

- `summary.json`
- `summary.md`
- `results.csv`
- `hard_case_breakdown.csv`
- `provider_failures.csv`

## Safety

Do not commit `.evals/` artifacts. Trace prompts and gold labels may contain
real document content or operational metadata.
