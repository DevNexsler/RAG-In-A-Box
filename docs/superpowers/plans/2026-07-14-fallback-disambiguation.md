# Fallback Disambiguation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the transient-vs-blank ambiguity in enrichment (image/ocr/video/audio) by consulting a per-modality LiteLLM fallback, so provider outages retry (never cap/strand) and genuinely-blank inputs settle to clean — uniformly across all four modalities.

**Architecture:** One shared decision-rule helper (`core/fallback.py::resolve_with_fallback`) + one shared LiteLLM client (`providers/fallback/litellm_fallback.py`), reused by two thin protocol adapters (`FallbackOCRProvider`, `MediaFallbackProvider`). Every enrichment provider is always wrapped (fallback `None` = dark). Every caller classifies degradations via `core.resilience.is_transient`.

**Tech Stack:** Python 3.12, pytest (+ pytest-asyncio), httpx, existing `core.resilience` (`TransientError`, `is_transient`, `call_with_retry`), existing provider protocols (`OCRProvider`, `MediaProvider`).

**Spec:** `docs/superpowers/specs/2026-07-14-describe-fallback-disambiguation-design.md`

**Branch:** `feat/describe-fallback-disambiguation` (already contains PRs #57–#61 merged off the integration branch; this plan reconciles them on top).

**Test interpreter:** `.venv/bin/python -m pytest` (repo has no `python` on PATH; use `.venv/bin/python`).

---

## File Structure

**Create:**
- `core/fallback.py` — `resolve_with_fallback(primary_call, fallback_call)` (pure decision rule, zero I/O).
- `providers/fallback/__init__.py` — package marker + `build_litellm_fallback(cfg)` helper.
- `providers/fallback/litellm_fallback.py` — `LiteLLMFallback(endpoint, model, prompt, encoder)` client + image/audio/video encoders.
- `tests/test_fallback_core.py` — shared-core branch matrix.
- `tests/test_fallback_litellm.py` — client (mocked httpx): empty→"", unreachable→raises transient.
- `tests/test_fallback_ocr.py` — `FallbackOCRProvider` describe/extract matrix.
- `tests/test_fallback_media.py` — `MediaFallbackProvider` video/audio matrix.

**Modify:**
- `providers/ocr/ollama_vision.py` — `describe()` raises transient on cooldown/connect (was: return `""`); delete `_note_describe_degradation`.
- `providers/ocr/fallback.py` — NEW `FallbackOCRProvider` (create; listed under modify-area since it lives beside existing OCR providers).
- `providers/ocr/__init__.py` — `build_ocr_provider` always-wraps describe+extract; parses `.fallback` subsections; registers `litellm` fallback build.
- `providers/media/fallback.py` — NEW `MediaFallbackProvider`.
- `providers/media/openrouter_media.py` — `transcribe_audio` all-fail path raises `TransientError` (was: bare `RuntimeError`).
- `providers/media/__init__.py` — `build_media_provider` wraps in `MediaFallbackProvider`; parses `media.video/audio.fallback`.
- `extractors.py` — `extract_image`/`extract_audio`/`extract_video` except → `is_transient(e)` classification; drop all three empty notes.
- `tests/test_ocr.py`, `tests/test_extractors.py` — reconcile the notes/raise behavior (in the tasks that change it).

---

## Task 1: Establish the RED baseline

**Files:** none (verification only)

- [ ] **Step 1: Confirm the integration regression is currently red**

Run: `.venv/bin/python -m pytest tests/test_degraded_ledger.py::test_provider_outage_never_caps_doc -q`
Expected: **FAIL** — `assert 5 == 0` (the stacked #59+#60+#61 bug this plan fixes).

- [ ] **Step 2: Snapshot the other soon-to-change tests**

Run: `.venv/bin/python -m pytest tests/test_ocr.py tests/test_extractors.py -q 2>&1 | tail -15`
Expected: some FAIL (the 6 contract-drift tests). Note the names; they are reconciled in later tasks. No commit.

---

## Task 2: Shared decision-rule core

**Files:**
- Create: `core/fallback.py`
- Test: `tests/test_fallback_core.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fallback_core.py
import pytest
from core.fallback import resolve_with_fallback
from core.resilience import TransientError


def _raises_transient():
    raise TransientError("unreachable")


def test_primary_text_passthrough_no_fallback_called():
    calls = {"fb": 0}
    def fb():
        calls["fb"] += 1
        return "fallback"
    assert resolve_with_fallback(lambda: "primary text", fb) == "primary text"
    assert calls["fb"] == 0  # cost guard: fallback not called when primary succeeds


def test_primary_unreachable_propagates_and_fallback_not_called():
    calls = {"fb": 0}
    def fb():
        calls["fb"] += 1
        return "fallback"
    with pytest.raises(TransientError):
        resolve_with_fallback(_raises_transient, fb)
    assert calls["fb"] == 0  # cost guard: no fallback on unreachable primary


def test_dark_mode_empty_raises_transient():
    # fallback is None -> unconfirmed empty must retry, never be treated as clean
    with pytest.raises(TransientError):
        resolve_with_fallback(lambda: "", None)


def test_reachable_empty_fallback_text_recovers():
    assert resolve_with_fallback(lambda: "   ", lambda: "recovered") == "recovered"


def test_reachable_empty_fallback_empty_confirms_blank():
    assert resolve_with_fallback(lambda: "", lambda: "") == ""


def test_reachable_empty_fallback_unreachable_raises_transient():
    with pytest.raises(TransientError):
        resolve_with_fallback(lambda: "", _raises_transient)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_fallback_core.py -q`
Expected: FAIL (`ModuleNotFoundError: core.fallback`).

- [ ] **Step 3: Implement**

```python
# core/fallback.py
"""Modality-agnostic transient-vs-blank disambiguation for enrichment providers.

One rule reused by every fallback wrapper (OCR describe/extract, media video/audio):
consult an independent fallback ONLY when the primary is reachable-but-empty, so a
provider outage never fans out into paid calls and never caps a doc.
"""
from __future__ import annotations

from typing import Callable

from core.resilience import TransientError


def resolve_with_fallback(
    primary_call: Callable[[], str],
    fallback_call: Callable[[], str] | None,
) -> str:
    """Return enrichment text, or "" for a fallback-CONFIRMED blank.

    Contract for both callables: RAISE a transient error (satisfying
    core.resilience.is_transient) when the provider is unreachable; RETURN "" when it
    is reachable but produced nothing.

    - primary raises           -> propagates (unreachable; fallback NOT called)
    - primary returns text      -> returned as-is
    - primary reachable-empty:
        - fallback is None       -> raise TransientError (unconfirmed empty -> retry)
        - fallback returns text  -> returned (RECOVERED)
        - fallback returns ""    -> return "" (CONFIRMED BLANK -> caller treats as clean)
        - fallback raises        -> propagates (both down -> retry)
    """
    text = primary_call()
    if text.strip():
        return text
    if fallback_call is None:
        raise TransientError(
            "enrichment empty and no fallback configured (unconfirmed blank)"
        )
    recovered = fallback_call()
    return recovered if recovered.strip() else ""
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_fallback_core.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add core/fallback.py tests/test_fallback_core.py
git commit -m "feat(fallback): shared transient-vs-blank decision rule"
```

---

## Task 3: #59 reconcile — ollama_vision raises transient (not swallow-to-empty)

**Files:**
- Modify: `providers/ocr/ollama_vision.py` (`describe()` cooldown path ~270-278, connect path ~281-284; delete `_note_describe_degradation` ~155-161)
- Modify: `tests/test_ocr.py` (`test_describe_connect_error_enters_cooldown_and_notes_degradation`)

- [ ] **Step 1: Rewrite the test to expect a transient RAISE + cooldown short-circuit**

Replace `test_describe_connect_error_enters_cooldown_and_notes_degradation` in `tests/test_ocr.py` with:

```python
def test_describe_connect_error_raises_transient_and_enters_cooldown(self, tmp_path):
    from core.resilience import TransientError, is_transient
    from providers.ocr.ollama_vision import OllamaVisionOCR

    state = {"calls": 0}

    class _Client:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def stream(self, method, url, json=None):
            state["calls"] += 1
            raise httpx.ConnectError("Connection refused")

    with patch("providers.ocr.ollama_vision.httpx.Client", _Client):
        provider = OllamaVisionOCR(base_url="http://ollama:11434")
        with pytest.raises(Exception) as first:
            provider.describe(self._img(tmp_path))
        assert is_transient(first.value)          # outage is transient
        # second call is short-circuited by cooldown -> raises WITHOUT hitting client
        with pytest.raises(TransientError):
            provider.describe(self._img(tmp_path))
    assert state["calls"] == 1                     # cooldown suppressed the 2nd real call
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest "tests/test_ocr.py::TestOllamaVisionBudget::test_describe_connect_error_raises_transient_and_enters_cooldown" -q`
Expected: FAIL (describe currently returns `""`, does not raise).

- [ ] **Step 3: Implement — raise instead of swallow; delete the degradation hook**

In `providers/ocr/ollama_vision.py`:

Delete the `_note_describe_degradation` method (lines ~155-161) entirely — the wrapper/caller now owns classification, so the `extractors` import cycle hack is gone.

Change the cooldown branch in `describe()`:

```python
        remaining = self._describe_cooldown_remaining()
        if remaining > 0:
            logger.info(
                "Vision describe skipped for %s; backend cooldown %.0fs remaining",
                file_path, remaining,
            )
            raise TransientError(
                f"vision describe backend in cooldown, {remaining:.0f}s remaining"
            )
```

Change the connect-error branch:

```python
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            self._mark_describe_unavailable(file_path, exc)
            raise  # transient (httpx.ConnectError); wrapper/caller classifies + degrades
```

Add the import near the top: `from core.resilience import TransientError`.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_ocr.py -q`
Expected: PASS (the rewritten test green; the empty-retry return-`""` path unchanged).

- [ ] **Step 5: Commit**

```bash
git add providers/ocr/ollama_vision.py tests/test_ocr.py
git commit -m "fix(ocr): #59 describe raises transient on cooldown/connect instead of swallowing to empty"
```

---

## Task 4: FallbackOCRProvider adapter

**Files:**
- Create: `providers/ocr/fallback.py`
- Test: `tests/test_fallback_ocr.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_fallback_ocr.py
import pytest
from core.resilience import TransientError
from providers.ocr.base import OCRProvider
from providers.ocr.fallback import FallbackOCRProvider


class _Stub(OCRProvider):
    def __init__(self, describe_fn, extract_fn=None):
        self._d, self._e = describe_fn, extract_fn or (lambda p, pg=None: "")
    def describe(self, file_path): return self._d(file_path)
    def extract(self, file_path, page=None): return self._e(file_path, page)


def _fb(text):
    def run(path): return text
    return run


def test_describe_recovers_from_reachable_empty():
    w = FallbackOCRProvider(_Stub(lambda p: ""), describe_fallback=_fb("desc"))
    assert w.describe("/x.png") == "desc"


def test_describe_confirmed_blank_returns_empty():
    w = FallbackOCRProvider(_Stub(lambda p: ""), describe_fallback=_fb(""))
    assert w.describe("/x.png") == ""


def test_describe_primary_unreachable_raises_and_fallback_not_called():
    calls = {"n": 0}
    def fb(path): calls["n"] += 1; return "x"
    def boom(p): raise TransientError("down")
    w = FallbackOCRProvider(_Stub(boom), describe_fallback=fb)
    with pytest.raises(TransientError):
        w.describe("/x.png")
    assert calls["n"] == 0


def test_describe_dark_mode_empty_raises_transient():
    w = FallbackOCRProvider(_Stub(lambda p: ""), describe_fallback=None)
    with pytest.raises(TransientError):
        w.describe("/x.png")


def test_extract_passthrough_when_primary_has_text():
    w = FallbackOCRProvider(_Stub(lambda p: "", lambda p, pg=None: "text"),
                            extract_fallback=_fb("fb"))
    assert w.extract("/x.pdf", page=1) == "text"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_fallback_ocr.py -q`
Expected: FAIL (`ModuleNotFoundError: providers.ocr.fallback`).

- [ ] **Step 3: Implement**

```python
# providers/ocr/fallback.py
"""OCRProvider that adds a per-method LiteLLM fallback via the shared decision rule."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from core.fallback import resolve_with_fallback
from providers.ocr.base import OCRProvider

FallbackRun = Callable[[str], str]  # path -> text; raises transient on unreachable


class FallbackOCRProvider(OCRProvider):
    def __init__(
        self,
        primary: OCRProvider,
        describe_fallback: Optional[FallbackRun] = None,
        extract_fallback: Optional[FallbackRun] = None,
    ):
        self._primary = primary
        self._describe_fb = describe_fallback
        self._extract_fb = extract_fallback

    def describe(self, file_path: str | Path) -> str:
        p = str(file_path)
        return resolve_with_fallback(
            lambda: self._primary.describe(p),
            (lambda: self._describe_fb(p)) if self._describe_fb else None,
        )

    def extract(self, file_path: str | Path, page: Optional[int] = None) -> str:
        p = str(file_path)
        return resolve_with_fallback(
            lambda: self._primary.extract(p, page),
            (lambda: self._extract_fb(p)) if self._extract_fb else None,
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_fallback_ocr.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add providers/ocr/fallback.py tests/test_fallback_ocr.py
git commit -m "feat(ocr): FallbackOCRProvider adapter over shared decision rule"
```

---

## Task 5: LiteLLM fallback client + image encoder

**Files:**
- Create: `providers/fallback/__init__.py`, `providers/fallback/litellm_fallback.py`
- Test: `tests/test_fallback_litellm.py`

- [ ] **Step 1: Write failing tests (mocked httpx)**

```python
# tests/test_fallback_litellm.py
import httpx, pytest
from core.resilience import is_transient
from providers.fallback.litellm_fallback import LiteLLMFallback, image_encoder


def _resp(content):
    return {"choices": [{"message": {"content": content}}]}


def test_returns_text(monkeypatch, tmp_path):
    img = tmp_path / "a.png"; img.write_bytes(b"\x89PNG\r\n")
    def fake_post(url, **kw):
        return httpx.Response(200, json=_resp("a description"), request=httpx.Request("POST", url))
    monkeypatch.setattr("providers.fallback.litellm_fallback.httpx.post", fake_post)
    fb = LiteLLMFallback("http://lite/v1", "m", "describe", image_encoder, api_key="k")
    assert fb.run(str(img)) == "a description"


def test_reachable_empty_returns_empty(monkeypatch, tmp_path):
    img = tmp_path / "a.png"; img.write_bytes(b"x")
    def fake_post(url, **kw):
        return httpx.Response(200, json=_resp("   "), request=httpx.Request("POST", url))
    monkeypatch.setattr("providers.fallback.litellm_fallback.httpx.post", fake_post)
    fb = LiteLLMFallback("http://lite/v1", "m", "p", image_encoder, api_key="k")
    assert fb.run(str(img)) == ""


def test_unreachable_raises_transient(monkeypatch, tmp_path):
    img = tmp_path / "a.png"; img.write_bytes(b"x")
    def fake_post(url, **kw):
        raise httpx.ConnectError("refused", request=httpx.Request("POST", url))
    monkeypatch.setattr("providers.fallback.litellm_fallback.httpx.post", fake_post)
    fb = LiteLLMFallback("http://lite/v1", "m", "p", image_encoder, api_key="k",
                         attempts=1)
    with pytest.raises(Exception) as e:
        fb.run(str(img))
    assert is_transient(e.value)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_fallback_litellm.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# providers/fallback/__init__.py
from providers.fallback.litellm_fallback import LiteLLMFallback  # noqa: F401
```

```python
# providers/fallback/litellm_fallback.py
"""One OpenAI-compatible LiteLLM client used as the fallback for every modality.

Contract (matches core.fallback.resolve_with_fallback's fallback_call):
  - unreachable -> raises a transient error (httpx.ConnectError / 5xx via is_transient)
  - reachable but empty -> returns ""
Per-modality difference is ONLY (endpoint, model, prompt, encoder).
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Callable

import httpx

from core.resilience import DEFAULT_ATTEMPTS, DEFAULT_BACKOFF, call_with_retry

Encoder = Callable[[Path, str], list]  # (path, prompt) -> OpenAI content parts


def image_encoder(path: Path, prompt: str) -> list:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    suffix = path.suffix.lstrip(".") or "png"
    return [
        {"type": "text", "text": prompt},
        {"type": "image_url",
         "image_url": {"url": f"data:image/{suffix};base64,{b64}"}},
    ]

# audio_encoder / video_encoder are added in Task 8's follow-up step (input_audio /
# frame content); the client shape below is identical for all three.


class LiteLLMFallback:
    def __init__(self, endpoint: str, model: str, prompt: str, encoder: Encoder,
                 *, api_key: str | None = None, timeout: float = 300.0,
                 attempts: int = DEFAULT_ATTEMPTS,
                 backoff: tuple[float, ...] = DEFAULT_BACKOFF):
        self.base_url = endpoint.rstrip("/")
        self.model = model
        self.prompt = prompt
        self.encoder = encoder
        self.timeout = timeout
        self.attempts = attempts
        self.backoff = backoff
        self.api_key = (api_key or os.environ.get("LITELLM_API_KEY", "")
                        or os.environ.get("LITELLM_MASTER_KEY", ""))

    def run(self, file_path: str | Path) -> str:
        path = Path(file_path)
        content = self.encoder(path, self.prompt)
        payload = {"model": self.model,
                   "messages": [{"role": "user", "content": content}],
                   "temperature": 0.0}
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}

        def _once() -> str:
            resp = httpx.post(f"{self.base_url}/chat/completions",
                              json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()  # 5xx -> HTTPStatusError -> is_transient True
            data = resp.json()
            return (data["choices"][0]["message"]["content"] or "").strip()

        # call_with_retry re-raises the transient exc UNCHANGED on exhaustion, so an
        # unreachable endpoint propagates as a transient error to resolve_with_fallback.
        return call_with_retry(_once, attempts=self.attempts, backoff=self.backoff,
                               label=f"litellm-fallback {self.model}")
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_fallback_litellm.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add providers/fallback/ tests/test_fallback_litellm.py
git commit -m "feat(fallback): shared LiteLLM client + image encoder"
```

---

## Task 6: Wire build_ocr_provider (always-wrap + fallback subsections)

**Files:**
- Modify: `providers/ocr/__init__.py`
- Test: `tests/test_fallback_ocr.py` (add factory tests)

- [ ] **Step 1: Write failing factory tests**

```python
# append to tests/test_fallback_ocr.py
from providers.ocr import build_ocr_provider
from providers.ocr.fallback import FallbackOCRProvider


def test_factory_always_wraps_even_without_fallback():
    prov = build_ocr_provider({"ocr": {"enabled": True, "provider": "ollama_vision"}})
    assert isinstance(prov, FallbackOCRProvider)  # dark: wrapped, fallback None


def test_factory_builds_describe_fallback_from_config():
    prov = build_ocr_provider({"ocr": {"enabled": True, "provider": "ollama_vision",
        "describe": {"fallback": {"provider": "litellm",
                                  "endpoint": "http://lite/v1", "model": "m"}}}})
    assert isinstance(prov, FallbackOCRProvider)
    assert prov._describe_fb is not None


def test_factory_missing_fallback_model_raises():
    with pytest.raises(Exception):
        build_ocr_provider({"ocr": {"enabled": True, "provider": "ollama_vision",
            "describe": {"fallback": {"provider": "litellm", "endpoint": "http://lite/v1"}}}})
```

**ALSO reconcile the two existing factory tests in `tests/test_ocr.py`** that assert a bare
provider type (they will break once the factory always-wraps). Update them to inspect the
wrapper's `_primary`:

- `test_build_ocr_provider_deepseek` (~line 148): change
  `assert isinstance(provider, DeepSeekOCR2Local)` → `assert isinstance(provider, FallbackOCRProvider)`
  and read the base/timeout off `provider._primary` (e.g. `assert isinstance(provider._primary, DeepSeekOCR2Local)`, `assert provider._primary.base_url == ...`).
- `test_build_ocr_provider_routes_images_to_ollama_describe` (~line 165): change
  `assert isinstance(provider, CompositeOCRProvider)` → `assert isinstance(provider, FallbackOCRProvider)`
  and `assert isinstance(provider._primary, CompositeOCRProvider)`, then inspect `_primary._describe` / `_primary._extract` as before.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_fallback_ocr.py -q -k factory` and `.venv/bin/python -m pytest tests/test_ocr.py -q -k build_ocr_provider`
Expected: FAIL (factory returns bare provider / no wrap; the two updated existing tests also red until Step 3).

- [ ] **Step 3: Implement**

In `providers/ocr/__init__.py`, add a fallback-builder and always-wrap the describe/extract providers. Add:

```python
_DESCRIBE_PROMPT = "Describe this image in detail for document search."
_EXTRACT_PROMPT = "Transcribe all text in this image verbatim."


def _build_fallback_run(cfg: dict | None, prompt: str):
    if not cfg:
        return None
    if cfg.get("provider") != "litellm":
        raise ValueError(f"Unknown fallback provider: {cfg.get('provider')}")
    if not cfg.get("model"):
        raise ValueError("fallback.model is required (no implicit default)")
    from providers.fallback.litellm_fallback import LiteLLMFallback, image_encoder
    client = LiteLLMFallback(cfg["endpoint"], cfg["model"], prompt, image_encoder,
                             api_key=cfg.get("api_key"))
    return client.run
```

Then in `build_ocr_provider`, after computing `extract_prov`/`describe_prov` (and the single-default case), wrap the enrichment provider(s). Replace the `return`s so the describe path is ALWAYS a `FallbackOCRProvider`:

```python
    from providers.ocr.fallback import FallbackOCRProvider

    describe_fb = _build_fallback_run((describe_cfg or {}).get("fallback")
                                      or ocr_cfg.get("describe", {}).get("fallback"),
                                      _DESCRIBE_PROMPT)
    extract_fb = _build_fallback_run((extract_cfg or {}).get("fallback")
                                     or ocr_cfg.get("extract", {}).get("fallback"),
                                     _EXTRACT_PROMPT)

    primary = _compose_primary(extract_prov, describe_prov, default)
    if primary is None:
        return None
    return FallbackOCRProvider(primary,
                               describe_fallback=describe_fb,
                               extract_fallback=extract_fb)
```

Refactor the existing compose logic (providers/ocr/__init__.py:49-81) into a small
`_compose_primary(extract_prov, describe_prov, default)` helper that returns the bare primary
`OCRProvider`, preserving today's exact cases:
- `extract_prov is None and describe_prov is None` → return `None` (caller returns unwrapped `None`; nothing to wrap).
- `extract_prov is describe_prov` → return that single provider.
- otherwise → return `CompositeOCRProvider(extract_provider=extract_prov or describe_prov, describe_provider=describe_prov or extract_prov)`.

The factory then: if `not ocr_cfg.get("enabled")` → `None` (unchanged); elif `_compose_primary(...) is None` → `None`; else → wrap the composed primary once in `FallbackOCRProvider(primary, describe_fallback=..., extract_fallback=...)`. Do NOT wrap the `enabled: false` / all-`none` cases.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_fallback_ocr.py tests/test_ocr.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add providers/ocr/__init__.py tests/test_fallback_ocr.py
git commit -m "feat(ocr): always-wrap describe/extract in FallbackOCRProvider; litellm fallback config"
```

---

## Task 7: extract_image caller reconciliation

**Files:**
- Modify: `extractors.py` (`extract_image` ~519-536)
- Modify: `tests/test_extractors.py` (image empty/failed tests)

- [ ] **Step 1: Update the tests to the wrapper contract**

In `tests/test_extractors.py`: delete `test_extract_image_empty_describe_notes_degradation` (the empty note is gone — empty from a wrapped provider means confirmed-blank → clean). Change `test_extract_image_failed_describe_notes_single_reason` to assert the `Degradation` shape:

```python
def test_extract_image_failed_describe_notes_transient_reason():
    from extractors import begin_degradation_capture, collect_degradations, extract_image
    from providers.ocr.base import OCRProvider

    class BoomOCR(OCRProvider):
        def extract(self, file_path, page=None): return ""
        def describe(self, file_path): raise ConnectionError("vision host down")

    begin_degradation_capture()
    extract_image("/fake/path.png", ocr_provider=BoomOCR())
    degs = collect_degradations()
    assert [d.reason for d in degs] == ["ocr_describe_failed"]
    assert degs[0].transient is True  # ConnectionError is transient
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_extractors.py -q -k image`
Expected: FAIL (empty note still present).

- [ ] **Step 3: Implement — drop the empty note**

In `extractors.py::extract_image`, remove the `else:` empty-note block (lines ~525-532). The `try/except` keeps only:

```python
    try:
        vision_text = ocr_provider.describe(str(file_path))
    except Exception as e:
        logger.warning("OCR describe failed for %s: %s", file_path, e)
        note_degradation("ocr_describe_failed", transient=is_transient(e))
        vision_text = ""
    header = _format_image_metadata_header(meta)
    ...
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_extractors.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add extractors.py tests/test_extractors.py
git commit -m "fix(index): #61 drop extract_image empty-note (wrapper owns blank classification)"
```

---

## Task 8: MediaFallbackProvider + audio/video encoders

**Files:**
- Create: `providers/media/fallback.py`
- Add audio/video encoders to `providers/fallback/litellm_fallback.py`
- Test: `tests/test_fallback_media.py`

- [ ] **Step 1: Write failing tests** — mirror `tests/test_fallback_ocr.py`'s matrix on a `MediaProvider` stub (`analyze_video`, `transcribe_audio`): recover, confirmed-blank, primary-unreachable-no-fallback-call, dark-mode-empty-raises. (Same shape; see Task 4.)

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError: providers.media.fallback`.

- [ ] **Step 3: Implement**

```python
# providers/media/fallback.py
from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional
from core.fallback import resolve_with_fallback
from providers.media.base import MediaProvider

FallbackRun = Callable[[str], str]


class MediaFallbackProvider(MediaProvider):
    def __init__(self, primary: MediaProvider,
                 video_fallback: Optional[FallbackRun] = None,
                 audio_fallback: Optional[FallbackRun] = None):
        self._primary = primary
        self._video_fb = video_fallback
        self._audio_fb = audio_fallback

    def analyze_video(self, file_path: str | Path) -> str:
        p = str(file_path)
        return resolve_with_fallback(
            lambda: self._primary.analyze_video(p),
            (lambda: self._video_fb(p)) if self._video_fb else None)

    def transcribe_audio(self, file_path: str | Path) -> str:
        p = str(file_path)
        return resolve_with_fallback(
            lambda: self._primary.transcribe_audio(p),
            (lambda: self._audio_fb(p)) if self._audio_fb else None)
```

Add `audio_encoder(path, prompt)` (OpenAI `input_audio` base64 part) and `video_encoder(path, prompt)` to `litellm_fallback.py`, mirroring `image_encoder`. Unit-test each encoder returns the expected content-part shape (no network).

- [ ] **Step 4: Run to verify pass** — `.venv/bin/python -m pytest tests/test_fallback_media.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add providers/media/fallback.py providers/fallback/litellm_fallback.py tests/test_fallback_media.py
git commit -m "feat(media): MediaFallbackProvider + audio/video encoders"
```

---

## Task 9: openrouter_media transcribe_audio raises TransientError

**Files:**
- Modify: `providers/media/openrouter_media.py` (all-audio-models-fail path)
- Test: `tests/test_fallback_media.py` (add)

- [ ] **Step 1: Write failing test** — patch `OpenRouterMediaProvider` internals so every audio model fails with a network error; assert `transcribe_audio` raises an error for which `is_transient(...)` is True (today it raises a bare `RuntimeError`, `is_transient` → False).

- [ ] **Step 2: Run to verify failure** — the raised `RuntimeError` is not transient.

- [ ] **Step 3: Implement** — at the `raise RuntimeError("All OpenRouter audio models failed")` site, raise `TransientError("All OpenRouter audio models failed") from last_exc` (import `from core.resilience import TransientError`). Keep the reachable-but-empty return `""` path unchanged.

- [ ] **Step 4: Run to verify pass** — PASS.

- [ ] **Step 5: Commit**

```bash
git add providers/media/openrouter_media.py tests/test_fallback_media.py
git commit -m "fix(media): transcribe_audio all-fail raises transient (was bare RuntimeError)"
```

---

## Task 10: Wire build_media_provider (wrap + fallback subsections)

**Files:**
- Modify: `providers/media/__init__.py`
- Test: `tests/test_fallback_media.py` (factory tests)

- [ ] **Step 1: Write failing factory tests** — `media: {enabled, provider: openrouter}` → returns `MediaFallbackProvider` (dark); `media.video.fallback` present → `_video_fb` set; missing `model` → raises.

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement** — after building the `OpenRouterMediaProvider`, build `video_fallback`/`audio_fallback` runs (reuse a shared `_build_fallback_run` — factor the OCR one into `providers/fallback/__init__.py::build_litellm_fallback(cfg, prompt, encoder)` and import from both factories, DRY), then `return MediaFallbackProvider(primary, video_fallback=..., audio_fallback=...)`. Use `video_encoder`/`audio_encoder`. Keep `enabled: false`/`provider: none` → `None`.

- [ ] **Step 4: Run to verify pass.**

- [ ] **Step 5: Commit**

```bash
git add providers/media/__init__.py providers/fallback/__init__.py tests/test_fallback_media.py
git commit -m "feat(media): always-wrap in MediaFallbackProvider; shared litellm fallback builder"
```

---

## Task 11: extract_video + extract_audio caller reconciliation

**Files:**
- Modify: `extractors.py` (`extract_audio` ~564-573, `extract_video` ~584-599)
- Modify: `tests/test_extractors.py` (audio/video tests)

- [ ] **Step 1: Update tests** — the existing audio/video tests use bare-string membership
(`"audio_extract_failed" in collect_degradations()` at test_extractors.py ~436/444/460), which no
longer matches the `Degradation` NamedTuple. Rewrite them to the same shape Task 7 uses
(`[d.reason for d in collect_degradations()]` and `d.transient`). Assertions:
  - audio outage (`ConnectionError`) → `reasons == ["audio_extract_failed"]`, `transient is True`; **no** `audio_transcript_empty` note.
  - video outage (`ConnectionError`) → `reasons == ["video_extract_failed"]`, `transient is True` (NOT a skip).
  - video non-transient error (e.g. `ValueError("oversized")`) → `reasons == ["video_extract_failed"]`, `transient is False`.
  - **no** `video_analysis_empty` note on the empty path.

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

`extract_audio`:
```python
    try:
        transcript = media_provider.transcribe_audio(file_path)
    except Exception as e:
        logger.warning("Audio extraction failed for %s: %s", file_path, e)
        note_degradation("audio_extract_failed", transient=is_transient(e))
        transcript = ""
    return ExtractionResult.from_text(transcript, frontmatter=fm)
```

`extract_video` (drop `note_skip`; classify transient; drop empty note):
```python
    try:
        notes = media_provider.analyze_video(file_path)
    except Exception as e:
        logger.warning("Video extraction failed for %s: %s", file_path, e)
        # transient (outage) -> retries; permanent (oversized/codec, non-transient)
        # -> caps at _DEGRADED_MAX_ATTEMPTS then stops.
        note_degradation("video_extract_failed", transient=is_transient(e))
        notes = ""
    return ExtractionResult.from_text(notes, frontmatter=fm)
```

Note in the commit body: video permanent-failures move from the skip ledger to the degraded ledger (bounded 5-attempt cap instead of immediate skip) — accepted for uniformity.

- [ ] **Step 4: Run to verify pass** — `.venv/bin/python -m pytest tests/test_extractors.py -q` → PASS. If any `test_skip_ledger.py`/enrichment test asserted the old `video_extract_failed` skip, reconcile it here.

- [ ] **Step 5: Commit**

```bash
git add extractors.py tests/test_extractors.py
git commit -m "fix(index): uniform transient classification in extract_video/extract_audio; drop empty notes"
```

---

## Task 12: Integration — outage never caps; recover; confirmed-blank

**Files:**
- Modify/Add: `tests/test_degraded_ledger.py` (or `tests/test_targeted_index.py`)

- [ ] **Step 1: Confirm the resurrected repro now passes**

Run: `.venv/bin/python -m pytest tests/test_degraded_ledger.py::test_provider_outage_never_caps_doc -q`
Expected: **PASS** (was the Task 1 RED baseline). This test passes a **raw** `OllamaVisionOCR`
directly to `extract_image` (it does not go through `build_ocr_provider`), so what turns it
green is **Task 3** — `describe()` now *raises* `httpx.ConnectError` instead of swallowing to
`""` — combined with `extract_image`'s existing `note_degradation(..., transient=is_transient(e))`.
No factory wrapping is involved here. (The production doc flow *does* wrap via
`build_ocr_provider`/`build_media_provider` at flow_index_vault.py:1715/1727/2313/2314, so the
always-wrap takes effect in prod; it's just not what this particular test exercises.)

- [ ] **Step 2: Add two integration tests**

- reachable-but-empty + configured fallback returning text → the recovered text is indexed (`upsert_nodes` gets non-empty content, doc NOT in degraded ledger).
- reachable-but-empty + fallback returning "" (confirmed blank) → doc indexed as stub AND dropped from the degraded ledger (uses `clean_now` path).

- [ ] **Step 3: Run to verify** — PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_degraded_ledger.py
git commit -m "test(index): outage-no-cap green; fallback recover + confirmed-blank integration"
```

---

## Task 13: Full gate + integration rebuild verification

**Files:** none (verification)

- [ ] **Step 1: Targeted suite**

Run: `.venv/bin/python -m pytest tests/test_fallback_core.py tests/test_fallback_ocr.py tests/test_fallback_media.py tests/test_fallback_litellm.py tests/test_ocr.py tests/test_extractors.py tests/test_degraded_ledger.py tests/test_mcp_contract.py -q`
Expected: all PASS.

- [ ] **Step 2: Full gate**

Run: `make gate-fast` then `make gate` (see @docs/TESTING.md).
Expected: static ✓, unit ✓, integration ✓, staging-e2e ✓. Live tier remains preflight-blocked on this host (known `comm_postgres` DNS + `config_test.yaml` in worktree — NOT a real failure; do not "fix").

- [ ] **Step 3: Confirm scope with GitNexus (per CLAUDE.md)**

Run: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})` and verify only the intended files/flows changed. Run `gitnexus_impact` on any modified public symbol flagged before finalizing.

- [ ] **Step 4: Push branch; DO NOT auto-merge**

```bash
git push -u origin feat/describe-fallback-disambiguation
```
Stop and hand back: the branch (5 reconciled PRs + fallback) is the merge unit — get human approval before merging to `main` and before superseding/closing PRs #59 and #61.

---

## Notes for the executor

- **DRY:** factor the litellm fallback builder once (`providers/fallback/__init__.py::build_litellm_fallback`) and import from both factories (Task 10 does this).
- **YAGNI:** no budget counter, no "similar-answer" comparison, no protocol merge (spec non-goals).
- **Dark ship:** with no `fallback` config, everything is wrapped with `None` fallbacks — unconfirmed empties raise transient and retry. This is the #0251/#0264 fix and needs no live endpoint.
- **Do not touch** #60's `_merge_degraded_ledger`, #57 (health async), or #58 (Lance/disk) — orthogonal foundations.
