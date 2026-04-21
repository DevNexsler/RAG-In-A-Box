# LLM Trace Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in OpenRouter enrichment trace capture to a gitignored JSONL archive without changing enrichment behavior, and lock it down with regression tests.

**Architecture:** A small shared recorder module will own trace persistence. `OpenRouterGenerator` will construct a trace-safe request snapshot, record success and failure events, and keep trace write failures non-fatal. Config lives under `enrichment.trace_capture`.

**Tech Stack:** Python, pytest, httpx, JSONL file append, existing YAML config loader.

---

### Task 1: Add Recorder Module

**Files:**
- Create: `providers/llm/trace_recorder.py`
- Test: `tests/test_openrouter_trace_capture.py`

- [ ] **Step 1: Write failing recorder tests**

Add tests for disabled no-op, directory creation, JSONL append, and write-failure isolation.

- [ ] **Step 2: Run recorder tests to verify failure**

Run: `python3 -m pytest tests/test_openrouter_trace_capture.py -q`

- [ ] **Step 3: Write minimal recorder implementation**

Implement a small class or functions that:
- accept config
- build date/model-based file path
- append one JSON record per line
- swallow file errors after logging

- [ ] **Step 4: Run recorder tests to verify pass**

Run: `python3 -m pytest tests/test_openrouter_trace_capture.py -q`

### Task 2: Wire Recorder Into OpenRouter Provider

**Files:**
- Modify: `providers/llm/openrouter_llm.py`
- Modify: `providers/llm/__init__.py`
- Test: `tests/test_openrouter_trace_capture.py`

- [ ] **Step 1: Write failing provider trace tests**

Cover:
- success trace contents
- HTTP failure trace contents
- timeout final failure trace contents
- auth header/api key omission

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `python3 -m pytest tests/test_openrouter_trace_capture.py -q`

- [ ] **Step 3: Implement provider wiring**

Pass trace config from factory to provider. Record request/response snapshots around the existing `httpx.post(...)` call while preserving retries and current exceptions.

- [ ] **Step 4: Run targeted tests to verify pass**

Run: `python3 -m pytest tests/test_openrouter_trace_capture.py -q`

### Task 3: Expose Config And Ignore Rules

**Files:**
- Modify: `config.yaml.example`
- Modify: `config_test.yaml.example`
- Modify: `config.local.yaml.example`
- Modify: `config.vps.yaml.example`
- Modify: `.gitignore`
- Test: `tests/test_openrouter_trace_capture.py`

- [ ] **Step 1: Add failing config/ignore assertions**

Assert examples expose `enrichment.trace_capture` defaults and `.evals/` remains ignored.

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `python3 -m pytest tests/test_openrouter_trace_capture.py -q`

- [ ] **Step 3: Implement example config updates**

Add documented defaults without changing runtime behavior.

- [ ] **Step 4: Run targeted tests to verify pass**

Run: `python3 -m pytest tests/test_openrouter_trace_capture.py -q`

### Task 4: Final Verification

**Files:**
- Verify only

- [ ] **Step 1: Run focused regression suite**

Run: `python3 -m pytest tests/test_openrouter_trace_capture.py tests/test_enrichment.py tests/test_provider_errors.int.test.py -q -m "not live"`

- [ ] **Step 2: Review git diff**

Run: `git status --short`

- [ ] **Step 3: Summarize residual risks**

Confirm current scope is OpenRouter-only and capture stays opt-in.
