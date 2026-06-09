# Enrichment Model Benchmark Results - 2026-05-01

This record preserves the enrichment benchmark work performed through 2026-05-01.
Raw run artifacts live under `.evals/benchmarks/runs/`, which is gitignored.

## Scoring Notes

- Dataset: 100 labeled enrichment benchmark cases in `.evals/benchmarks/gold/`.
- Current scorer: semantic/fuzzy scorer from `core/benchmarking/scoring.py`.
- Older saved summaries for some baseline runs used the old scorer. Current scores below were re-scored from raw `per_case.jsonl` outputs where available.
- Primary quality fields: `summary`, `key_facts`, `doc_type`, `topics`, `keywords`, `suggested_tags`, `suggested_folder`.
- Manual audits matter because folder taxonomy can over- or under-weight otherwise good extraction.

## Main Results

| Run | Model | Cases | Current score | Success | Parse fail | Request fail | p50 ms | p95 ms | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `baseline-4.1-mini-100` | `openai/gpt-4.1-mini` | 100 | 0.840293 | 1.0 | 0 | 0 | 3845.736 | 6932.085 | Current quality baseline. Best overall manual quality. |
| `deepseek-v4-flash-100-20260430-semantic` | `deepseek/deepseek-v4-flash` | 100 | 0.703408 | 1.0 | 0 | 0 | 15156.817 | 30858.709 | Usable but slower and weaker facts/folders than 4.1 mini. |
| `qwen3.5-27b-100-20260430-semantic` | `qwen/qwen3.5-27b` | 100 | 0.611564 | 1.0 | 0 | 0 | 24580.018 | 41454.165 | Worse and slower than DeepSeek/GPT. |
| `gemma-4-31b-it-100-timeoutfix` | `google/gemma-4-31b-it` | 100 | 0.656407 | 1.0 | 0 | 0 | 8400.892 | 19018.205 | Re-scored from raw output. Mid-tier. |
| `nemotron-3-super-120b-a12b-100-20260430-semantic` | `nvidia/nemotron-3-super-120b-a12b` | 100 | 0.036111 | 0.1 | 76 | 14 | 12852.376 | 23475.537 | Not viable. Parse/provider failures. |
| `gemma-4-31b-it-free-100` | `google/gemma-4-31b-it:free` | 100 | 0.0 | 0.0 | 0 | 100 | | | Not viable. |

## Ten-Case Parameter Sweeps

| Run | Model | Cases | Current score | Success | p50 ms | p95 ms | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| `deepseek-v3.2-10-json-t0-none-512-20260430` | `deepseek/deepseek-v3.2` | 10 | 0.740386 | 1.0 | 9885.173 | 14302.069 | Best DeepSeek V3.2 config tested. Non-thinking JSON wins. |
| `deepseek-v3.2-10-schema-t0-none-2048-20260430` | `deepseek/deepseek-v3.2` | 10 | 0.732081 | 1.0 | 9486.494 | 12766.846 | Close to JSON route. |
| `deepseek-v3.2-10-schema-reasoning-high-65536-20260430` | `deepseek/deepseek-v3.2` | 10 | 0.677362 | 1.0 | 63000.357 | 77880.477 | Reasoning high slower and worse. |
| `mimo-v2-flash-10-json-t02-top09-none-2048-20260430` | `xiaomi/mimo-v2-flash` | 10 | 0.700395 | 1.0 | 3149.814 | 4125.365 | Best MiMo config. Fastest good candidate. |
| `mimo-v2-flash-10-json-t0-none-512-20260430` | `xiaomi/mimo-v2-flash` | 10 | 0.678493 | 1.0 | 2975.742 | 3777.750 | Good, but worse than temp 0.2/top_p 0.9. |
| `mimo-v2-flash-10-json-reasoning-high-65536-20260430` | `xiaomi/mimo-v2-flash` | 10 | 0.653838 | 1.0 | 14051.119 | 18740.570 | Reasoning high slower and worse. |
| `granite-4.1-8b-10-20260430` | `ibm-granite/granite-4.1-8b` | 10 | 0.644136 | 1.0 | 1993.485 | 3057.911 | Very fast. Backup/simple-doc candidate only. |
| `granite-4.1-8b-10-json-t0-512-20260430` | `ibm-granite/granite-4.1-8b` | 10 | 0.640248 | 1.0 | 1912.228 | 2702.818 | Similar to default. |
| `granite-4.1-8b-10-schema-t02-top09-2048-20260430` | `ibm-granite/granite-4.1-8b` | 10 | 0.632704 | 1.0 | 1742.200 | 2902.759 | Structured route did not improve quality. |
| `granite-4.1-8b-10-20260430-structured` | `ibm-granite/granite-4.1-8b` | 10 | 0.622619 | 1.0 | 1700.045 | 3249.989 | Worse than JSON/default. |
| `granite-4.1-8b-10-schema-t0-2048-20260430` | `ibm-granite/granite-4.1-8b` | 10 | 0.614436 | 1.0 | 1701.941 | 2892.619 | Worse than JSON/default. |
| `glm-4.7-flash-10-20260430-effort-none-structured` | `z-ai/glm-4.7-flash` | 10 | 0.608805 | 1.0 | 9247.049 | 20535.413 | Only viable with reasoning disabled. Not competitive. |
| `step-3.5-flash-10-schema-reasoning-medium-65536-20260430` | `stepfun/step-3.5-flash` | 10 | 0.678458 | 1.0 | 37949.920 | 44209.070 | Best StepFun config, but too slow. |
| `step-3.5-flash-10-schema-reasoning-low-65536-20260430` | `stepfun/step-3.5-flash` | 10 | 0.674685 | 1.0 | 29773.670 | 56299.856 | Too slow. |
| `step-3.5-flash-10-schema-reasoning-high-65536-valid-20260430` | `stepfun/step-3.5-flash` | 10 | 0.651470 | 1.0 | 34728.728 | 56392.830 | Too slow. |
| `gemma-4-31b-it-10-timeoutfix` | `google/gemma-4-31b-it` | 10 | 0.672222 | 1.0 | 8607.995 | 13261.531 | Mid-tier. |

## Failed Or Invalid Routes

| Run | Model | Issue |
|---|---|---|
| `glm-4.7-flash-10-20260430` | `z-ai/glm-4.7-flash` | Default route returned no usable content. Needs `reasoning: {"effort": "none"}`. |
| `mimo-v2-flash-10-schema-t0-none-2048-20260430` | `xiaomi/mimo-v2-flash` | Upstream 429/provider failure for schema route. |
| `mimo-v2-flash-10-schema-reasoning-high-65536-20260430` | `xiaomi/mimo-v2-flash` | Upstream 429/provider failure for schema route. |
| `step-3.5-flash-10-json-t0-none-512-20260430` | `stepfun/step-3.5-flash` | `json_object` unsupported and/or reasoning none rejected. |
| `step-3.5-flash-10-schema-t0-none-2048-20260430` | `stepfun/step-3.5-flash` | Reasoning none rejected. |
| `step-3.5-flash-10-schema-reasoning-high-65536-20260430` | `stepfun/step-3.5-flash` | Initial route invalid. Valid high-reasoning run is `...-valid-20260430`. |
| `gemma-4-31b-it-free-100` | `google/gemma-4-31b-it:free` | Provider route failed. |
| `nemotron-3-super-120b-a12b-100-20260430-semantic` | `nvidia/nemotron-3-super-120b-a12b` | High parse/provider failure rate. |

## DeepSeek Reasoning Findings

- `deepseek/deepseek-v4-flash` supports OpenRouter `reasoning`, `include_reasoning`, `top_k`, `top_p`, `response_format`, and `structured_outputs`.
- Non-thinking/default 100-case score: 0.703408.
- High reasoning 3-case smoke: 0.558405.
- High reasoning plus strict structured output 3-case smoke: 0.584627.
- Structured reasoning worked, but did not beat non-thinking JSON in the quick tests.
- For `deepseek/deepseek-v3.2`, high reasoning was materially slower and scored lower than non-thinking JSON.

## Manual Audit: 4.1 Mini vs DeepSeek V3.2 vs MiMo V2 Flash

Compared runs:

- `baseline-4.1-mini-100`
- `deepseek-v3.2-10-json-t0-none-512-20260430`
- `mimo-v2-flash-10-json-t02-top09-none-2048-20260430`

Exact first-10 current scores:

| Model | First-10 score | p50 ms | p95 ms |
|---|---:|---:|---:|
| `openai/gpt-4.1-mini` | 0.801236 | 3573.991 | 5292.198 |
| `deepseek/deepseek-v3.2` | 0.740386 | 9885.173 | 14302.069 |
| `xiaomi/mimo-v2-flash` | 0.700395 | 3149.814 | 4125.365 |

Manual verdict:

- `openai/gpt-4.1-mini` is still best overall. It has the strongest factual precision and cleanest summaries, especially on nuanced lease/tenant messages.
- `deepseek/deepseek-v3.2` is close second. It often scores well on tags/folders/doc type, but facts are sometimes less crisp than 4.1 mini.
- `xiaomi/mimo-v2-flash` is best speed/value. It is good enough for simple messages, but key facts can be compressed or generic.

Case-level audit:

| Case | Manual winner | Notes |
|---|---|---|
| `case_0001` | 4.1 mini | DeepSeek scored slightly higher, but 4.1 mini captured scam warning and details cleaner. |
| `case_0002` | DeepSeek V3.2 | Slight win. 4.1 mini close. MiMo too compressed. |
| `case_0003` | 4.1 mini | Best exact lease/reminder log framing. |
| `case_0004` | DeepSeek V3.2 / tie | All three good. DeepSeek has best balance. |
| `case_0005` | 4.1 mini | Simple follow-up, concise and correct. |
| `case_0006` | 4.1 mini | Best exact payment/water/property facts. |
| `case_0007` | 4.1 mini | Strongest rental inquiry/scam safety extraction. MiMo close. |
| `case_0008` | DeepSeek V3.2 | Slight win on case status follow-up. |
| `case_0009` | 4.1 mini | Clear win. Best lease/communication nuance. |
| `case_0010` | 4.1 mini | Slight win, low-stakes short thank-you. |

## Current Recommendation

- Keep `openai/gpt-4.1-mini` as the default enrichment model for quality.
- Run a full 100-case benchmark for `deepseek/deepseek-v3.2` before considering it as a default replacement.
- Consider `xiaomi/mimo-v2-flash` as a fast/cheap route for simple message-like documents, not for nuanced enrichment.
- Do not use high-reasoning routes for this workload unless a future benchmark shows a clear lift. Current tests show higher latency and lower score.
- Avoid `nemotron-3-super-120b-a12b`, `stepfun/step-3.5-flash`, and default `z-ai/glm-4.7-flash` for production enrichment based on current evidence.

## Raw Artifact Index

Primary artifacts:

- `.evals/benchmarks/runs/baseline-4.1-mini-100/`
- `.evals/benchmarks/runs/deepseek-v4-flash-100-20260430-semantic/`
- `.evals/benchmarks/runs/qwen3.5-27b-100-20260430-semantic/`
- `.evals/benchmarks/runs/nemotron-3-super-120b-a12b-100-20260430-semantic/`
- `.evals/benchmarks/runs/gemma-4-31b-it-100-timeoutfix/`
- `.evals/benchmarks/runs/deepseek-v3.2-10-json-t0-none-512-20260430/`
- `.evals/benchmarks/runs/deepseek-v3.2-10-schema-t0-none-2048-20260430/`
- `.evals/benchmarks/runs/deepseek-v3.2-10-schema-reasoning-high-65536-20260430/`
- `.evals/benchmarks/runs/mimo-v2-flash-10-json-t02-top09-none-2048-20260430/`
- `.evals/benchmarks/runs/mimo-v2-flash-10-json-t0-none-512-20260430/`
- `.evals/benchmarks/runs/mimo-v2-flash-10-json-reasoning-high-65536-20260430/`
- `.evals/benchmarks/runs/granite-4.1-8b-10-20260430/`
- `.evals/benchmarks/runs/granite-4.1-8b-10-json-t0-512-20260430/`
- `.evals/benchmarks/runs/granite-4.1-8b-10-schema-t0-2048-20260430/`
- `.evals/benchmarks/runs/granite-4.1-8b-10-schema-t02-top09-2048-20260430/`
- `.evals/benchmarks/runs/glm-4.7-flash-10-20260430-effort-none-structured/`
- `.evals/benchmarks/runs/step-3.5-flash-10-schema-reasoning-low-65536-20260430/`
- `.evals/benchmarks/runs/step-3.5-flash-10-schema-reasoning-medium-65536-20260430/`
- `.evals/benchmarks/runs/step-3.5-flash-10-schema-reasoning-high-65536-valid-20260430/`
