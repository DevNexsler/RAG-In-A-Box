# LLM Trace Capture Design

**Date:** 2026-04-21

**Goal:** Add opt-in prompt/response capture for the OpenRouter enrichment path so future benchmark work can mine real GPT-4.1-mini cases without changing enrichment behavior.

## Scope

- Capture only the current `gpt-4.1-mini` enrichment path.
- Hook at provider layer, not `doc_enrichment.py`, so captured data matches the real transport payload.
- Store traces locally in a gitignored archive.
- Keep trace writing best-effort: failures must never break enrichment.

## Decisions

### Capture Boundary

Add a shared recorder module under `providers/llm/`. `OpenRouterGenerator.generate()` will call it for success and failure paths.

### Config

Use existing enrichment config namespace:

```yaml
enrichment:
  trace_capture:
    enabled: false
    directory: ".evals/llm-traces"
```

Default is off. When enabled, traces are written locally for every OpenRouter enrichment call.

### Storage Format

Write JSON Lines files under `.evals/llm-traces/`.

Suggested file name:

`YYYY-MM-DD-openrouter-openai-gpt-4.1-mini.jsonl`

Each line stores one request/response event with:

- timestamp
- provider
- model
- request payload
- response body or failure body
- success flag
- latency in milliseconds
- structured error metadata when the call fails

### Data Fidelity

When trace capture is enabled, store full prompt and full response body. Never store auth headers or API keys.

### Failure Model

Trace capture is best-effort only. File I/O failures log a warning and return. API failures continue to raise exactly as before after the failure trace is attempted.

## Test Strategy

Add regression tests for:

- recorder disabled path
- recorder enabled append path
- parent directory creation
- recorder write failure isolation
- OpenRouter success trace contents
- OpenRouter failure trace contents
- absence of auth secrets in stored traces

## Files

- `providers/llm/trace_recorder.py`
- `providers/llm/openrouter_llm.py`
- `providers/llm/__init__.py`
- `config*.yaml.example`
- `.gitignore`
- `tests/test_openrouter_trace_capture.py`

