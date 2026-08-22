# Qwen Bulk Enrichment Design

## Goal

Prevent Qwen3.8/vLLM constrained-decoding loops during document enrichment while preserving strict JSON-schema requests for models that support them.

## Request policy

One LiteLLM model-capability policy owns enrichment request shape. `qwen-bulk` uses `json_object`, temperature `0.7`, top-p `0.8`, presence penalty `1.5`, and `extra_body` with `top_k: 20` plus `chat_template_kwargs.enable_thinking: false`. All other LiteLLM models retain `json_schema` and configured temperature.

## Validation and retry

LiteLLM validates a returned response as one JSON object containing non-empty `summary` and `doc_type` before accepting it. Invalid JSON or missing core fields retry through the existing transport retry loop, then raise a transient error. `doc_enrichment` remains owner of normalization: omitted optional and context fields become existing empty defaults.

## Verification

Unit tests assert Qwen policy, unchanged default policy, validation retry, and optional-field normalization. A live replay verifies request payload, stop finish reason, valid JSON, and usage below the configured limit.
