# LiteLLM-Primary OCR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route OCR extraction and image description exclusively through LiteLLM aliases `ocr` and `vision`, with updated live gates and real endpoint coverage.

**Architecture:** Add a thin `LiteLLMOCR` implementation of `OCRProvider` that composes two existing `LiteLLMFallback` request clients, one per model alias. `build_ocr_provider` returns it directly for top-level `provider: litellm`, leaving legacy direct providers and application-level fallback behavior unchanged. Live preflight and live tests move from direct Mac Mini protocols to LiteLLM model metadata plus two real multimodal calls.

**Tech Stack:** Python 3.13, `httpx`, PyYAML, Pillow, pytest, Ruff, Docker Compose, GitNexus.

---

## File Map

- Create `providers/ocr/litellm_ocr.py`: first-class LiteLLM OCR/vision adapter.
- Create `tests/test_litellm_ocr.py`: unit and factory tests for payload routing, errors, auth, and validation.
- Create `tests/test_litellm_ocr_live.py`: required live end-to-end calls through aliases `ocr` and `vision`.
- Modify `providers/ocr/__init__.py`: recognize top-level `provider: litellm` and return the new provider directly.
- Modify `tests/test_ocr.py`: remove orphaned imports causing the pre-existing Ruff failure; retire direct-Mac live tests while preserving legacy provider unit coverage.
- Modify `scripts/live_preflight.py`: replace the direct Mac probe with authenticated LiteLLM model/alias validation.
- Modify `tests/test_live_preflight.py`: test every new preflight success/failure branch.
- Modify `tests/test_service_health.py`: replace direct DeepSeek health tests with LiteLLM alias health validation.
- Modify `config.yaml.example`: document first-class LiteLLM OCR configuration.
- Modify `docs/TESTING.md`: document new live dependency, preflight, and endpoint test.
- Operational-only `/home/danpark/projects/RAG-in-a-Box/config.yaml`: switch active ignored config after tracked code passes.

Use project interpreter `/home/danpark/projects/RAG-in-a-Box/.venv/bin/python`. For Make targets, prepend `/home/danpark/projects/RAG-in-a-Box/.venv/bin` to `PATH`.

## Execution Workspace and GitNexus Setup

Run Tasks 1–8 only from:

```text
/home/danpark/projects/RAG-in-a-Box/.worktrees/litellm-primary-ocr
```

The plan and spec both exist inside that worktree under `docs/superpowers/`. Do not execute
implementation commands from the dirty main checkout.

Before Task 1, inspect the worktree's `.gitnexus/meta.json` if present, then run
`npx gitnexus analyze` from the worktree (`--embeddings` only if its prior embeddings count
was nonzero). Call `gitnexus_list_repos`, select the entry whose path exactly matches the
worktree, and use its returned repository identity (normally `litellm-primary-ocr`) for all
Tasks 1–8 impact, rename, context, and change-detection calls. Never select dirty main merely
by its `RAG-in-a-Box` name. GitNexus may refresh generated instruction blocks in `AGENTS.md`
or `CLAUDE.md`; never stage those generated diffs with feature commits. Reindex the main
checkout after final integration so the shared MCP index points back to main.

### Task 1: Repair Pre-existing Static Baseline

**Files:**
- Modify: `tests/test_ocr.py:24`

- [ ] **Step 1: Preserve existing red evidence**

Existing clean-worktree command:

```bash
make gate-fast
```

Expected current failure: Ruff `F401` for unused `begin_degradation_capture` and `collect_degradations` imports.

- [ ] **Step 2: Remove only the orphaned import**

Delete:

```python
from extractors import begin_degradation_capture, collect_degradations
```

No symbol changes; GitNexus impact is not required for an import-only edit.

- [ ] **Step 3: Verify focused static check**

Run:

```bash
ruff check tests/test_ocr.py
```

Expected: `All checks passed!`

- [ ] **Step 4: Verify clean baseline**

Run:

```bash
make gate-fast
```

Expected: static, unit, and integration tiers pass. If another pre-existing failure appears, stop and report it before feature work.

- [ ] **Step 5: Stage, detect scope, and commit**

Stage the intended file, run GitNexus staged change detection, verify only that file appears,
then commit:

```bash
git add tests/test_ocr.py
git commit -m "chore(test): remove stale OCR imports"
```

### Task 2: Add First-class LiteLLM OCR Adapter

**Files:**
- Create: `tests/test_litellm_ocr.py`
- Create: `providers/ocr/litellm_ocr.py`

- [ ] **Step 1: Write failing request-routing tests**

Create tests using a real temporary PNG and monkeypatched
`providers.fallback.litellm_fallback.httpx.post`:

```python
def test_extract_routes_to_ocr_alias(monkeypatch, tmp_path):
    image = tmp_path / "page.png"
    image.write_bytes(b"png-bytes")
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "invoice 123"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(
        "providers.fallback.litellm_fallback.httpx.post", fake_post
    )
    provider = LiteLLMOCR(
        "http://lite/v1", "ocr", "vision", timeout=41, api_key="unit-key"
    )

    assert provider.extract(image, page=7) == "invoice 123"
    assert captured["url"] == "http://lite/v1/chat/completions"
    assert captured["json"]["model"] == "ocr"
    assert "Transcribe all text" in captured["json"]["messages"][0]["content"][0]["text"]
    assert captured["timeout"] == 41
    assert captured["headers"]["Authorization"] == "Bearer unit-key"
```

Add parallel `test_describe_routes_to_vision_alias`, asserting model `vision` and the detailed-description prompt. Add tests proving reachable empty returns `""`, environment auth is used when no explicit key is supplied, and a 401/malformed response remains transient through the reused client.

- [ ] **Step 2: Run tests to verify red**

Run:

```bash
python -m pytest tests/test_litellm_ocr.py -q
```

Expected: collection fails with `ModuleNotFoundError: providers.ocr.litellm_ocr`.

- [ ] **Step 3: Implement minimal provider**

Create `providers/ocr/litellm_ocr.py`:

```python
"""First-class OCR and image description through LiteLLM model aliases."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from providers.fallback.litellm_fallback import LiteLLMFallback, image_encoder
from providers.ocr.base import OCRProvider

logger = logging.getLogger(__name__)

_EXTRACT_PROMPT = "Transcribe all text in this image verbatim."
_DESCRIBE_PROMPT = "Describe this image in detail for document search."


class LiteLLMOCR(OCRProvider):
    def __init__(
        self,
        endpoint: str,
        extract_model: str,
        describe_model: str,
        timeout: float = 300.0,
        *,
        api_key: str | None = None,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.extract_model = extract_model
        self.describe_model = describe_model
        self.timeout = timeout
        self._extract_client = LiteLLMFallback(
            self.endpoint, extract_model, _EXTRACT_PROMPT, image_encoder,
            api_key=api_key, timeout=timeout,
        )
        self._describe_client = LiteLLMFallback(
            self.endpoint, describe_model, _DESCRIBE_PROMPT, image_encoder,
            api_key=api_key, timeout=timeout,
        )
        logger.info(
            "LiteLLMOCR: %s extract_model=%s describe_model=%s",
            self.endpoint, extract_model, describe_model,
        )

    def extract(self, file_path: str | Path, page: Optional[int] = None) -> str:
        return self._extract_client.run(file_path)

    def describe(self, file_path: str | Path) -> str:
        return self._describe_client.run(file_path)
```

- [ ] **Step 4: Verify green**

Run:

```bash
python -m pytest tests/test_litellm_ocr.py tests/test_fallback_litellm.py -q
ruff check providers/ocr/litellm_ocr.py tests/test_litellm_ocr.py
```

Expected: all tests and Ruff pass.

- [ ] **Step 5: Stage, detect scope, and commit**

Stage the intended files, run GitNexus staged change detection, verify expected symbols and
flows, then commit:

```bash
git add providers/ocr/litellm_ocr.py tests/test_litellm_ocr.py
git commit -m "feat(ocr): add LiteLLM primary provider"
```

### Task 3: Wire LiteLLM into OCR Factory

**Files:**
- Modify: `tests/test_litellm_ocr.py`
- Modify: `providers/ocr/__init__.py:58-92`

- [ ] **Step 1: Write failing factory tests**

Add:

```python
def test_factory_builds_first_class_litellm_provider(monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "env-key")
    provider = build_ocr_provider({
        "ocr": {
            "enabled": True,
            "provider": "litellm",
            "endpoint": "http://lite/v1",
            "extract_model": "ocr",
            "describe_model": "vision",
            "timeout": 99,
            "api_key": "must-not-be-read-from-yaml",
        }
    })
    assert isinstance(provider, LiteLLMOCR)
    assert not isinstance(provider, FallbackOCRProvider)
    assert provider._extract_client.api_key == "env-key"
    assert provider._describe_client.api_key == "env-key"
```

Parameterize missing `endpoint`, `extract_model`, and `describe_model`; each must raise a clear `ValueError` naming the missing key.

- [ ] **Step 2: Run tests to verify red**

Run:

```bash
python -m pytest tests/test_litellm_ocr.py -q
```

Expected: factory test fails with `Unknown OCR provider: litellm`.

- [ ] **Step 3: Run mandatory impact analysis**

Run GitNexus upstream impact for `build_ocr_provider`. Report direct callers, affected processes, and risk before editing. If risk is HIGH or CRITICAL, stop and warn user.

- [ ] **Step 4: Implement minimal factory branch**

At the start of `build_ocr_provider`, after enabled validation and before legacy provider construction:

```python
if ocr_cfg.get("provider") == "litellm":
    required = ("endpoint", "extract_model", "describe_model")
    missing = [key for key in required if not ocr_cfg.get(key)]
    if missing:
        raise ValueError(
            "ocr litellm provider missing required config: " + ", ".join(missing)
        )
    from providers.ocr.litellm_ocr import LiteLLMOCR
    return LiteLLMOCR(
        endpoint=ocr_cfg["endpoint"],
        extract_model=ocr_cfg["extract_model"],
        describe_model=ocr_cfg["describe_model"],
        timeout=ocr_cfg.get("timeout", 300.0),
    )
```

Do not read `api_key` from YAML. Do not alter legacy split-provider or fallback behavior.

- [ ] **Step 5: Verify green and regression suite**

Run:

```bash
python -m pytest tests/test_litellm_ocr.py tests/test_ocr.py tests/test_fallback_ocr.py -q
ruff check providers/ocr tests/test_litellm_ocr.py tests/test_ocr.py
```

Expected: all pass.

- [ ] **Step 6: Stage, detect scope, and commit**

Stage the intended files, run GitNexus staged change detection, verify expected symbols and
flows, then commit:

```bash
git add providers/ocr/__init__.py tests/test_litellm_ocr.py
git commit -m "feat(ocr): select LiteLLM as routing authority"
```

### Task 4: Migrate Live Preflight to LiteLLM

**Files:**
- Modify: `tests/test_live_preflight.py:75-132,263-266`
- Modify: `scripts/live_preflight.py:1-118,147-153`

- [ ] **Step 1: Replace direct-Mac tests with failing LiteLLM tests**

Tests must cover:

```python
def test_litellm_ocr_models_available(tmp_path, monkeypatch):
    cfg = _write_config(
        tmp_path / "config.yaml",
        """ocr:
  provider: litellm
  endpoint: http://lite:4000/v1
  extract_model: ocr
  describe_model: vision
""",
    )
    monkeypatch.setattr(lp, "config_candidates", lambda: [cfg])
    monkeypatch.setenv("LITELLM_API_KEY", "key")
    captured = {}

    def fake_get(url, **kwargs):
        captured.update(url=url, **kwargs)
        return httpx.Response(
            200,
            json={"data": [{"id": "ocr"}, {"id": "vision"}]},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(lp.httpx, "get", fake_get)
    ok, reason = lp.check_litellm_ocr()
    assert ok, reason
    assert captured["url"] == "http://lite:4000/v1/models"
    assert captured["headers"]["Authorization"] == "Bearer key"
```

Also cover missing config file, missing endpoint, each missing model key, missing both LiteLLM key env vars, connection error, 401, malformed response, and one missing alias. Update expected `CHECKS` names to include `litellm_ocr` instead of `mac_ocr`.

Add a focused test proving `config_candidates()` prefers worktree/CWD `config.yaml` before
the main checkout copy, so live validation exercises the branch configuration.

- [ ] **Step 2: Run tests to verify red**

Run:

```bash
python -m pytest tests/test_live_preflight.py -q
```

Expected: failures because `check_litellm_ocr` does not exist and registry still says `mac_ocr`.

- [ ] **Step 3: Run mandatory impact and rename analysis**

Run GitNexus upstream impact for `check_mac_ocr` and `config_candidates`. Report risk. Then
run `gitnexus_rename` from `check_mac_ocr` to `check_litellm_ocr` with `dry_run: true`;
review graph and text-search edits before applying `dry_run: false`.

- [ ] **Step 4: Implement non-generating model validation**

Replace function body with logic that:

1. Loads first available real `config.yaml` candidate, preferring CWD/worktree before main.
2. Requires `ocr.endpoint`, `ocr.extract_model`, and `ocr.describe_model`.
3. Requires `LITELLM_API_KEY` or `LITELLM_MASTER_KEY` from environment.
4. GETs `{endpoint.rstrip('/')}/models` with bearer auth and five-second timeout.
5. Calls `raise_for_status()`.
6. Parses `data[*].id` and requires both configured aliases.
7. Returns concise actionable failure reasons without exposing credentials.

Update module comments and registered check name. Keep production-indexer-idle protection, but describe contention as shared LiteLLM local hardware rather than direct Mac access.

- [ ] **Step 5: Verify green**

Run:

```bash
python -m pytest tests/test_live_preflight.py -q
ruff check scripts/live_preflight.py tests/test_live_preflight.py
```

Expected: all pass.

- [ ] **Step 6: Stage, detect scope, and commit**

Stage the intended files, run GitNexus staged change detection, verify expected symbols and
flows, then commit:

```bash
git add scripts/live_preflight.py tests/test_live_preflight.py
git commit -m "test(live): preflight LiteLLM OCR aliases"
```

### Task 5: Update Live Health Surface and Retire Direct-Mac Live Tests

**Files:**
- Modify: `tests/test_service_health.py:1-43,126-193`
- Modify: `tests/test_ocr.py:446-556`

- [ ] **Step 1: Write new health assertion**

Replace `TestDeepSeekOCR2Health` with:

```python
@pytest.mark.live
class TestLiteLLMOCRHealth:
    def test_ocr_and_vision_aliases_available(self):
        from scripts.live_preflight import check_litellm_ocr
        ok, reason = check_litellm_ocr()
        assert ok, reason
```

Remove now-unused direct DeepSeek resolver/probe helpers and PNG builder. Remove the explicit direct-Mac live block from `tests/test_ocr.py`; retain all DeepSeek and Ollama unit tests.

- [ ] **Step 2: Run mandatory impacts before editing existing symbols**

Run GitNexus upstream impacts for these exact file-qualified symbols before removal:

- `tests/test_service_health.py::_probe_url`
- `tests/test_service_health.py::_minimal_png`
- `tests/test_service_health.py::_resolve_ocr_base_url`
- `tests/test_service_health.py::_deepseek_ocr2_extract_available`
- `tests/test_service_health.py::TestDeepSeekOCR2Health`
- `tests/test_ocr.py::_resolve_ocr_base_url`
- `tests/test_ocr.py::_deepseek_ocr2_running`
- `tests/test_ocr.py::_minimal_png`
- `tests/test_ocr.py::_deepseek_ocr2_model_ready`
- `tests/test_ocr.py::TestDeepSeekOCR2Live`

Use `file_path` to disambiguate repeated helper names. Report any HIGH/CRITICAL result before
editing.

- [ ] **Step 3: Apply minimal test migration**

Delete only direct live-service helpers/classes. Preserve legacy provider unit tests and the user's independent cooldown regression when later integrating the branch.

- [ ] **Step 4: Verify collection and focused tests**

Run:

```bash
python -m pytest tests/test_service_health.py tests/test_ocr.py --collect-only -q
python -m pytest tests/test_ocr.py -m unit -q
ruff check tests/test_service_health.py tests/test_ocr.py
```

Expected: collection succeeds, legacy unit tests pass, Ruff passes.

- [ ] **Step 5: Stage, detect scope, and commit**

Stage the intended files, run GitNexus staged change detection, verify expected symbols and
flows, then commit:

```bash
git add tests/test_service_health.py tests/test_ocr.py
git commit -m "test(live): retire direct Mac OCR checks"
```

### Task 6: Add Required Real LiteLLM End-to-End Test

**Files:**
- Create: `tests/test_litellm_ocr_live.py`

- [ ] **Step 1: Add deterministic generated fixture**

Create a large white PNG with high-contrast text `RAGBOX LIVE OCR 7319`, a red circle, and a blue square. Use Pillow and a bundled/default font; do not add a binary fixture.

- [ ] **Step 2: Add two non-skipping live assertions**

```python
@pytest.fixture(scope="module")
def provider():
    from core.config import load_config
    built = build_ocr_provider(load_config())
    assert isinstance(built, LiteLLMOCR)
    return built


@pytest.mark.live
def test_litellm_ocr_extract_live(provider, live_image):
    text = provider.extract(live_image)
    normalized = "".join(ch for ch in text.upper() if ch.isalnum())
    assert "RAGBOXLIVEOCR7319" in normalized


@pytest.mark.live
def test_litellm_vision_describe_live(provider, live_image):
    description = provider.describe(live_image).lower()
    assert "red" in description
    assert "blue" in description
    assert "circle" in description or "round" in description
    assert "square" in description or "box" in description
```

No `skipif`: preflight failure blocks spend; inference failure fails the live tier.

- [ ] **Step 3: Verify collection without spend**

Run:

```bash
python -m pytest tests/test_litellm_ocr_live.py --collect-only -q
```

Expected: exactly two live tests collected.

- [ ] **Step 4: Stage, detect scope, and commit**

Stage the intended file, run GitNexus staged change detection, verify expected scope, then
commit:

```bash
git add tests/test_litellm_ocr_live.py
git commit -m "test(live): cover LiteLLM OCR and vision"
```

### Task 7: Update Example Configuration and Testing Manual

**Files:**
- Modify: `config.yaml.example:64-102`
- Modify: `docs/TESTING.md:1-75,134-185`

- [ ] **Step 1: Document first-class provider**

Make the example OCR block show:

```yaml
ocr:
  enabled: true
  provider: "litellm"
  endpoint: "http://YOUR_LITELLM_PROXY:4000/v1"
  extract_model: "ocr"
  describe_model: "vision"
  timeout: 300
  concurrency: 1
```

State that credentials come only from `LITELLM_API_KEY` or `LITELLM_MASTER_KEY`, and LiteLLM owns local/cloud fallback policy. Keep legacy direct-provider examples compact if still useful, but do not present them as production routing.

- [ ] **Step 2: Update operator manual**

Replace Mac Mini live-tier/preflight text with LiteLLM endpoint, model aliases, key requirements, non-generating `/models` preflight, and the two-call generated-image live test. Keep full `make gate` as release requirement and retain the production-idle guard explanation.

- [ ] **Step 3: Verify docs and config syntax**

Run:

```bash
python -c 'import yaml; yaml.safe_load(open("config.yaml.example"))'
ruff check .
```

Expected: YAML parses and Ruff passes.

- [ ] **Step 4: Stage, detect scope, and commit**

Stage the intended files, run GitNexus staged change detection, verify expected scope, then
commit:

```bash
git add config.yaml.example docs/TESTING.md
git commit -m "docs(config): document LiteLLM-primary OCR"
```

### Task 8: Configure Worktree and Run Required Gates

**Files:**
- Local ignored: `.env`, `config.yaml`, `config_test.yaml`

- [ ] **Step 1: Copy ignored live-test inputs without printing secrets**

Copy `.env`, `config.yaml`, and `config_test.yaml` from the main checkout into the worktree.
Do not display either secret file. `config_candidates()` now prefers this worktree copy, so
preflight and live tests exercise the branch configuration rather than legacy main config.

- [ ] **Step 2: Switch worktree active OCR config**

Patch only the ignored `ocr:` block to the approved LiteLLM configuration. Preserve every unrelated setting.

- [ ] **Step 3: Run non-generating preflight**

Run:

```bash
python scripts/live_preflight.py
```

Expected: `ok litellm_ocr` lists endpoint and both aliases without revealing the key. If production indexer is active, wait until its heartbeat is stale and rerun; do not bypass the guard.

- [ ] **Step 4: Run development gate**

Run:

```bash
make gate-fast
```

Expected: static, unit, integration pass.

- [ ] **Step 5: Run full release gate including approved spend**

Run:

```bash
make gate
```

Expected: all five tiers pass. Read `result.json` and `report.md`; confirm both `tests/test_litellm_ocr_live.py` tests passed, not skipped.

- [ ] **Step 6: Run GitNexus final scope audit**

Run `gitnexus_detect_changes(scope="compare", base_ref="main")`. Confirm only planned symbols/files/flows changed and all d=1 dependents remain covered.

### Task 9: Integrate and Roll Out

**Files:**
- Operational-only: `/home/danpark/projects/RAG-in-a-Box/config.yaml`

- [ ] **Step 1: Use branch-finishing workflow**

Invoke `superpowers:finishing-a-development-branch`. Preserve main checkout edits in `AGENTS.md`, `CLAUDE.md`, `tests/test_ocr.py`, and `docs/TICKET-26-stage3-recovery-plan.md`. Never reset or overwrite them.

Before attempting integration:

1. Save a read-only backup of main's uncommitted patch under `/tmp` and record `git status`.
2. Compare main's `tests/test_ocr.py` dirty hunk with branch hunks; verify the user's cooldown
   test is disjoint and remains present in the proposed result.
3. Preview committed integration with `git merge-tree`.
4. Attempt normal non-destructive merge only when preview is clean. If Git refuses because
   the working tree is dirty, stop and ask user; do not stash, reset, checkout, or clean.
5. After a successful merge, compare main's remaining diff with the saved patch and verify
   all pre-existing user edits still exist before any rollout.

- [ ] **Step 2: Update active ignored config after integration**

Patch the main checkout's `ocr:` block to the approved LiteLLM configuration. Re-read the exact block and verify unrelated settings are unchanged.

- [ ] **Step 3: Rebuild and restart production service**

From the integrated main checkout, run:

```bash
docker compose -f docker-compose.yml up -d --build --wait doc-organizer
```

This rebuilds `doc-organizer:latest`, recreates only the named service, and waits for its
configured health check.

- [ ] **Step 4: Verify deployed state**

Run:

```bash
curl --max-time 10 -sS -i http://127.0.0.1:7788/health
docker logs --tail 200 doc-organizer
```

Expected: HTTP 200, healthy container, and startup log showing LiteLLM endpoint plus aliases `ocr` and `vision`, with no credential output.

- [ ] **Step 5: Refresh GitNexus after final commit/merge**

Check `.gitnexus/meta.json` embeddings count, then run `npx gitnexus analyze` with `--embeddings` only if embeddings were already present.
