# Full LLM Benchmark With Hard Log Cases

Date: 2026-06-02
Status: Draft for review
Scope: Build a modular LLM benchmark framework for every task that uses `openai/gpt-4.1-mini`, starting with document enrichment.

## Goal

Create a repeatable benchmark that compares candidate LLMs against the current `openai/gpt-4.1-mini` enrichment baseline using real production trace data, with special coverage for hard examples found in logs.

The benchmark must answer:

- Which model gives best quality for each LLM task?
- Which model is fastest and cheapest at acceptable quality?
- Which cases fail due to model quality versus provider/reliability problems?
- Do candidate models handle the hard cases that actually appear in production logs?

## Current Findings

Current runtime `4.1-mini` usage is document enrichment through `OpenRouterGenerator`.

Existing benchmark assets:

- `100` broad enrichment cases under `.evals/benchmarks/cases/`
- `100` gold labels under `.evals/benchmarks/gold/`
- `39` saved benchmark runs under `.evals/benchmarks/runs/`
- Existing runner, scoring, reporting, and CLI in `core/benchmarking/` and `scripts/enrichment_benchmark.py`

Trace scan findings from `.evals/llm-traces/`:

- `74,333` trace rows across `35` trace files
- `20,769` failed LLM calls
- Largest observed prompt around `325,696` chars
- Real hard examples include huge prompt bodies, nearby-message context, taxonomy-heavy prompts, link-heavy email boilerplate, business-critical property operations, 402 billing failures, DNS/connect failures, and slow successful calls.

Representative hard references:

- `.evals/llm-traces/2026-05-19-openrouter-openai-gpt-4.1-mini.jsonl:3721` — `325k` char prompt, nearby context, link noise, property ops, slow success.
- `.evals/llm-traces/2026-05-21-openrouter-openai-gpt-4.1-mini.jsonl:1453` — `325k` char prompt, nearby context, link noise, `79s` success.
- `.evals/llm-traces/2026-05-10-openrouter-openai-gpt-4.1-mini.jsonl:101` — 402 Payment Required failure.
- `.evals/llm-traces/2026-05-05-openrouter-openai-gpt-4.1-mini.jsonl:88` — DNS/connect failure.

These references are used as selection inputs only. Sensitive trace payloads remain under gitignored `.evals/`.

## Non-Goals

- Do not commit trace payloads, benchmark cases, gold labels, or model outputs that may contain sensitive data.
- Do not benchmark retrieval, embeddings, BM25, or reranking in this project. Those need a separate retrieval benchmark.
- Do not use another LLM as judge for the primary score.
- Do not treat provider failures as model intelligence failures.

## Architecture

Use a modular benchmark framework with task modules.

```text
core/benchmarking/
  tasks/
    enrichment.py
    base.py
  mining.py
  cases.py
  scoring.py
  runner.py
  reporting.py
scripts/enrichment_benchmark.py
```

`tasks/base.py` defines the task contract:

- task name
- how to load a case
- how to replay a case against a model
- how to normalize output
- how to score output
- what summary/subscore fields to report

`tasks/enrichment.py` adapts the current enrichment benchmark to the task contract.

`mining.py` scans trace JSONL files and emits candidate hard-case manifests without copying sensitive prompts into git.

## Benchmark Suites

### Standard Suite

The current 100 labeled enrichment cases remain the broad baseline.

Purpose:

- measure general enrichment quality
- preserve continuity with prior benchmark results
- compare candidate models against `openai/gpt-4.1-mini`

### Hard Suite

The hard suite is mined from `.evals/llm-traces/` using deterministic signals.

Signals:

- `huge_prompt`: prompt over configured length threshold, default `250,000` chars
- `long_prompt`: prompt over configured length threshold, default `25,000` chars
- `nearby_context`: prompt contains same-channel context candidates
- `taxonomy_bloat`: prompt contains large available tags/folders taxonomy block
- `link_noise`: prompt contains many HTTP/tracking links
- `business_critical`: prompt mentions rent, tenant, payment, lease, eviction, invoice, maintenance, damage, urgent, or scam
- `slow_success`: successful call over configured latency threshold
- `provider_failure`: failed call with HTTP or transport error
- `parse_suspect`: response exists but does not look like valid structured enrichment JSON

The hard suite should target `30-50` cases. It must preserve a balanced mix rather than taking only largest prompts.

### Regression Suite

Small fast subset, around `10` cases.

Purpose:

- quick local comparison
- CI smoke test
- guard against scorer/runner regressions

### Provider-Failure Suite

Failure traces are benchmarked separately.

Purpose:

- classify 402, DNS/connect, timeout, and HTTP provider failures
- verify reports separate operational reliability from model quality
- preserve visibility into degraded production periods

Provider-failure cases do not receive model-quality scores.

## Case Selection Flow

1. Scan trace files under `.evals/llm-traces/`.
2. Extract safe metadata: file, line, timestamp, model, success, latency, prompt length, document type, title prefix, flags, token usage, error type/status.
3. Score hard candidates using deterministic flag weights.
4. Deduplicate repeated prompts by hash.
5. Stratify by hard-case category and source document type.
6. Materialize selected cases under gitignored `.evals/benchmarks/<task>/<suite>/cases/`.
7. Create gold stubs for cases that need manual quality labels.
8. Never write raw prompt text to tracked files.

## Scoring

Enrichment quality scoring keeps the current field-weighted scorer:

- summary
- doc_type
- topics
- keywords
- key_facts
- suggested_tags
- suggested_folder
- importance
- entities_people
- entities_places
- entities_orgs
- entities_dates

Context-aware cases add optional subscores:

- atomic extraction quality
- context extraction quality
- context rejection/acceptance correctness
- context source ID correctness
- context warning correctness

Reliability metrics are reported separately:

- parse failure rate
- transport failure rate
- provider failure rate
- success rate
- latency p50/p95
- token total/average
- cost total/average when present

## Reporting

Each run writes:

- `per_case.jsonl`
- `summary.json`
- `leaderboard.md`
- `field_scores.csv`
- optional `hard_case_breakdown.csv`
- optional `failure_breakdown.csv`

Reports must separate:

- model quality
- latency/cost
- parse reliability
- provider/transport reliability
- hard-case category performance

## CLI

Keep the existing `scripts/enrichment_benchmark.py` entry point, but extend it with suite-aware commands:

```bash
python3 scripts/enrichment_benchmark.py mine-hard \
  --trace-dir .evals/llm-traces \
  --bench-dir .evals/benchmarks \
  --task enrichment \
  --suite hard \
  --limit 50

python3 scripts/enrichment_benchmark.py run \
  --bench-dir .evals/benchmarks \
  --task enrichment \
  --suite hard \
  --model openai/gpt-4.1-mini \
  --run-id baseline-hard

python3 scripts/enrichment_benchmark.py report \
  --bench-dir .evals/benchmarks \
  --task enrichment \
  --suite hard \
  --run-id baseline-hard
```

Existing commands should remain backward compatible for the current `.evals/benchmarks/` layout.

## Privacy

All sensitive benchmark data remains gitignored.

Tracked files may include:

- benchmark framework code
- tests with synthetic fixtures
- docs
- non-sensitive schema examples

Tracked files must not include:

- real prompts
- real responses
- real gold labels
- trace payload excerpts
- customer/tenant/private source text

## Testing

Unit tests:

- task-module contract
- trace metadata extraction
- hard-case scoring and selection
- dedupe by prompt hash
- suite path resolution
- provider failure classification
- report aggregation
- backward compatibility for existing enrichment benchmark commands

Integration tests:

- prepare hard cases from synthetic trace fixtures
- run one synthetic case through fake replay client
- verify report includes hard-case breakdown and failure breakdown

Live tests:

- optional only
- run manually with OpenRouter credentials
- not part of default test suite

## Success Criteria

- Existing 100-case enrichment benchmark still runs.
- New hard suite can be mined from real traces without committing sensitive data.
- Hard suite includes real long-context/taxonomy/context/provider-failure examples.
- Candidate model reports show quality, latency, cost, parse reliability, provider reliability, and hard-category performance.
- Framework can add a future LLM task without rewriting runner/scoring/reporting.
