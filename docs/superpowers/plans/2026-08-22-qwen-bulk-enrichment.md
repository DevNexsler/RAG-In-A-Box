# Qwen Bulk Enrichment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route `qwen-bulk` enrichment requests around the vLLM JSON-schema loop and reject invalid structured responses before acceptance.

**Architecture:** A centralized LiteLLM request-policy helper selects Qwen's JSON-object/non-thinking sampling parameters. Response validation remains in the generator retry boundary; existing `doc_enrichment` normalization supplies empty optional/context values after valid JSON is accepted.

**Tech Stack:** Python 3.13, httpx, pytest.

**Spec:** `docs/superpowers/specs/2026-08-22-qwen-bulk-enrichment.md`

## Global Constraints

- `qwen-bulk`: `json_object`, temperature `0.7`, top-p `0.8`, presence penalty `1.5`, `top_k: 20`, and `chat_template_kwargs.enable_thinking: false`.
- All other LiteLLM models retain existing `json_schema` behavior.
- Never accept malformed JSON or responses without non-empty `summary` and `doc_type`.

---

### Task 1: Centralized request policy and response validation

**Files:**
- Modify: `providers/llm/litellm_llm.py`
- Test: `tests/test_litellm_llm.py`

**Interfaces:**
- Produces: one policy helper used by `LiteLLMGenerator._request_with_metadata`.
- Produces: retryable validation of response content before `generate()` returns it.

- [x] **Step 1: Write failing tests**

```python
def test_qwen_bulk_uses_json_object_non_thinking_sampling():
    ...
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.8
    assert payload["presence_penalty"] == 1.5
    assert payload["extra_body"] == {
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }

def test_qwen_bulk_retries_malformed_json_then_returns_valid_json():
    ...
    assert post.call_count == 2
```

- [x] **Step 2: Run tests and verify expected failure**

Run: `python3 -m pytest tests/test_litellm_llm.py -q`

Expected: FAIL because Qwen-specific policy and JSON validation do not exist.

- [x] **Step 3: Implement minimal policy and validation**

```python
def _enrichment_request_policy(model: str) -> dict[str, Any]:
    if model.casefold() == "qwen-bulk":
        return {
            "response_format": {"type": "json_object"},
            "temperature": 0.7,
            "top_p": 0.8,
            "presence_penalty": 1.5,
            "extra_body": {
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }
    return {"response_format": json_schema, "temperature": configured_temperature}
```

Validate each successful content string with `json.loads`, require non-empty `summary` and `doc_type`, and retry malformed/missing-core responses before raising `TransientError`.

- [x] **Step 4: Run focused tests**

Run: `python3 -m pytest tests/test_litellm_llm.py -q`

Expected: PASS.

### Task 2: Optional-field normalization regression coverage

**Files:**
- Modify: `tests/test_enrichment.py`

**Interfaces:**
- Consumes: existing `parse_enrichment_response`.
- Produces: regression proof that valid partial JSON has empty optional/context enrichment defaults.

- [x] **Step 1: Write a failing normalization test**

```python
def test_parse_enrichment_response_defaults_missing_optional_context_fields():
    result = parse_enrichment_response('{"summary":"ok","doc_type":["email"]}')
    assert result["enr_context_warning"] == ""
    assert result["enr_entities_people"] == ""
```

- [x] **Step 2: Run the test and verify expected failure if normalization is incomplete**

Run: `python3 -m pytest tests/test_enrichment.py -k defaults_missing_optional -q`

- [x] **Step 3: Add only code needed for a failure**

Use existing normalization when the test is already green; do not change parser behavior without a demonstrated gap.

- [x] **Step 4: Run focused enrichment tests**

Run: `python3 -m pytest tests/test_enrichment.py -q`

Expected: PASS.

### Task 3: Live replay and integration verification

**Files:**
- Test only: live LiteLLM proxy, no source changes.

**Interfaces:**
- Consumes: `LiteLLMGenerator(model="qwen-bulk")` and a captured enrichment prompt.
- Produces: redacted request-shape, latency, usage, finish reason, and JSON-validation result.

- [x] **Step 1: Replay one captured enrichment prompt**

Run a `qwen-bulk` request through `LiteLLMGenerator` with a 16,384 maximum and report only request policy fields, latency, usage, finish reason, and validation status.

- [x] **Step 2: Run regression tests**

Run: `make gate-fast`

Expected: PASS.

- [x] **Step 3: Detect changed symbols before commit**

Run: GitNexus `detect_changes({scope: "all"})`.

- [ ] **Step 4: Commit and push**

```bash
git add providers/llm/litellm_llm.py tests/test_litellm_llm.py tests/test_enrichment.py docs/superpowers/specs/2026-08-22-qwen-bulk-enrichment.md docs/superpowers/plans/2026-08-22-qwen-bulk-enrichment.md
git commit -m "fix: route qwen bulk enrichment around schema loop"
git push -u origin fix/qwen-bulk-enrichment
```
