# LLM Enrichment Benchmark Design

Date: 2026-04-21
Status: Approved for planning
Scope: Benchmark only LLM-powered document enrichment currently using `openai/gpt-4.1-mini` via OpenRouter.

## Goal

Build a repeatable benchmark that answers one question:

Which model is the best replacement for `gpt-4.1-mini` on this repo's enrichment task?

The benchmark must:

- use real saved prompts from production trace capture
- use gold labels reviewed and authored by Codex
- replay exact prompts against candidate models
- score candidate outputs with weighted field-level quality metrics
- report quality, reliability, latency, and optional cost separately

## Non-Goals

- benchmarking retrieval, chunking, or embeddings
- benchmarking OCR or vision models
- using a second LLM as judge
- committing sensitive benchmark cases or outputs into git

## Current System Context

`gpt-4.1-mini` is used in one place that matters for this benchmark:

- document enrichment prompt generation in `doc_enrichment.py`
- OpenRouter transport in `providers/llm/openrouter_llm.py`

Trace capture now records full prompts and responses to `.evals/llm-traces/*.jsonl`. That archive is the source material for case creation.

## Recommended Approach

Use a gold-case harness.

1. Curate 100 real traces from `.evals/llm-traces/`
2. Author one gold answer per case manually in structured JSON
3. Allow alternate acceptable values for ambiguous filing fields
4. Replay exact prompts against candidate models
5. Normalize outputs through the same enrichment normalization logic
6. Score each field with weighted rules
7. Produce a leaderboard plus detailed failure analysis

This is preferred over baseline-against-baseline comparison because it measures correctness against reviewed targets, not similarity to current model behavior.

## Data Layout

All sensitive benchmark data lives under gitignored `.evals/benchmarks/`.

Proposed layout:

```text
.evals/benchmarks/
  cases/
    manifest.json
    <case_id>.json
  gold/
    <case_id>.json
  runs/
    <run_id>/
      config.json
      per_case.jsonl
      summary.json
      summary.csv
      report.md
  status/
    labeling_status.json
```

### Case Record

Each case file stores:

- `case_id`
- source trace file path
- source trace line number
- trace timestamp
- provider and source model
- exact saved prompt to replay
- original raw model output from `gpt-4.1-mini`
- extracted title
- extracted source type
- rough category
- difficulty label
- prompt size and source text size

### Gold Record

Each gold file stores:

- canonical answer for all enrichment fields
- allowed alternate values for `suggested_tags`
- allowed alternate values for `suggested_folder`
- optional notes about ambiguity or scoring caveats
- optional per-case field weight overrides

### Run Record

Each benchmark run stores:

- model identifier
- replay settings
- raw output per case
- normalized output per case
- parse or transport errors
- latency
- usage metadata when available
- per-field scores
- total weighted score

## Case Selection

Target set: 100 cases.

Selection should be stratified, not first-100.

Desired mix:

- routine easy documents
- ambiguous folder/tag decisions
- long or truncated documents
- noisy or malformed text
- entity-heavy legal/admin documents
- communication-heavy threads and short messages
- housing/property management content

Selection pipeline:

1. scan all traces
2. filter out obvious synthetic or trivial smoke cases
3. infer coarse categories from title, source type, and text signals
4. infer difficulty from length, ambiguity signals, and current output quality
5. sample across categories so benchmark is not dominated by one source class

The manifest should record why each case was selected.

## Labeling Workflow

Codex authors the gold labels by reviewing each saved prompt and existing output.

Per case workflow:

1. load prompt, source text, and current `gpt-4.1-mini` answer
2. inspect what the document actually says
3. write canonical structured answer for all fields
4. record allowed alternates for `suggested_tags` and `suggested_folder` where ambiguity is real
5. save progress immediately

Important constraint:

- gold labels are not derived automatically from the current model output
- the current model output is only reference material during review

## Scoring

### Hard Gate

Before semantic scoring:

- output must be returned successfully
- output must parse to valid JSON
- normalized record must include all enrichment fields

If parse fails, case score is `0` and reliability metrics are incremented.

### Field Weights

Default field weights:

- `doc_type`: `0.18`
- `topics`: `0.18`
- `keywords`: `0.14`
- `key_facts`: `0.16`
- `suggested_tags`: `0.10`
- `suggested_folder`: `0.10`
- `importance`: `0.06`
- `entities_people`: `0.03`
- `entities_places`: `0.02`
- `entities_orgs`: `0.01`
- `entities_dates`: `0.01`
- `summary`: `0.01`

### Field Rules

`doc_type`, `topics`, `keywords`, `suggested_tags`

- compare as normalized sets
- score with balanced precision and recall
- normalize case, whitespace, punctuation, obvious duplicates
- allow configured alternates for tags

`key_facts`

- compare as normalized fact lists
- award partial credit for same fact with different wording
- penalize hallucinated facts more than missing low-value detail

`suggested_folder`

- exact match gets full credit
- configured alternate gets full credit
- same branch or close parent-child relationship gets partial credit
- unrelated branch gets low or zero credit

`importance`

- score by numeric distance from gold value
- small deviation tolerated
- large deviation penalized smoothly, not all-or-nothing

Entity and date fields

- mostly exact normalized matching
- dates normalized before compare
- duplicates removed before scoring

`summary`

- lightweight score only
- used as a sanity signal, not a primary ranking field

## Reliability, Latency, Cost

Quality score alone is not enough.

Each run also reports:

- success rate
- parse failure rate
- transport failure rate
- latency p50
- latency p95
- optional input and output token counts
- optional cost per 100 cases

Recommendation:

- rank by quality first
- show latency and cost as adjacent comparison columns, not folded into one blended score

## Components

### 1. Case Preparation Tool

Reads trace archive and generates curated benchmark cases.

Responsibilities:

- parse trace JSONL
- extract replay prompt and baseline output
- infer metadata for selection
- create case files and manifest

### 2. Labeling Tool

Terminal workflow for opening one case at a time and saving gold labels.

Responsibilities:

- show source prompt and source output
- create or update gold record
- track progress
- avoid data loss during long labeling sessions

### 3. Benchmark Runner

Replays exact prompts against a target model.

Responsibilities:

- send candidate model requests through OpenRouter
- persist raw outputs
- normalize outputs using repo enrichment logic
- compute scores and metrics
- save per-case and aggregate run results

### 4. Reporting Tool

Produces analyst-friendly summaries.

Responsibilities:

- leaderboard across runs
- field-by-field comparison
- best and worst cases
- disagreement examples
- failure buckets

## Data Flow

```text
llm trace archive
  -> prepare_cases
  -> curated case set
  -> label_cases
  -> gold dataset
  -> run_benchmark(model X)
  -> normalized outputs + scores
  -> report_benchmark
```

## Error Handling

Benchmarking must be resilient.

- malformed trace row: skip and log
- missing prompt or baseline response: mark unusable during preparation
- candidate model timeout/error: record failure, continue next case
- parse failure: score zero for that case, preserve raw output
- interrupted labeling session: recover from saved progress
- scoring bug on one case: isolate case failure, do not discard whole run

## Testing Strategy

Regression coverage should protect benchmark correctness.

Core tests:

- trace loader parses real trace schema
- case preparation filters and samples deterministically with seeded selection
- gold record schema validation
- output normalization matches repo enrichment normalization behavior
- scoring unit tests for each field type
- alternate folder and tag acceptance rules
- parse-failure case scores zero
- run summary aggregates correctly
- report generation emits stable machine-readable artifacts

Integration tests:

- small fixture benchmark with 2 to 5 cases
- replay path mocked at network layer
- end-to-end score report generated from fixture cases

## Security and Privacy

Benchmark data may contain sensitive document text.

Rules:

- cases, gold labels, and run outputs stay under `.evals/benchmarks/`
- `.evals/` remains gitignored
- no benchmark fixture should be committed unless explicitly sanitized later
- auth headers and API keys must never be persisted in benchmark artifacts

## Out of Scope for First Version

- UI for labeling or reports
- automatic semantic judge model
- multi-rater adjudication workflow
- benchmarking non-OpenRouter providers
- active-learning case selection

## Success Criteria

Version 1 is successful when:

- 100 curated real cases exist
- every case has a gold label authored by Codex
- runner can benchmark at least one candidate model end-to-end
- report shows weighted quality score, reliability, latency, and optional cost
- result is strong enough to compare candidate models against `gpt-4.1-mini`

## Recommended Next Step

Write implementation plan for:

1. benchmark dataset schema
2. case preparation command
3. labeling workflow
4. scoring engine
5. benchmark runner
6. reporting and tests
