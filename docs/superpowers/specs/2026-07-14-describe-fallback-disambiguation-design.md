# Fallback Disambiguation (all modalities) — Design

- **Date:** 2026-07-14
- **Status:** Approved (design); pending spec review
- **Owner:** dpark@nexsler.com
- **Related tickets:** #0251 (transient outage caps docs), #0264 (stub-indexed images unledgered), #0263 (ollama vision host outage), infra #0240 (host restart)
- **Related PRs:** #57, #58, #59, #60, #61 (the doc-organizer outage cluster)

## Problem

Enrichment describes/transcribes media with a model. When the model yields **no text**, the
pipeline cannot tell *why*, and the two causes need opposite handling:

1. **Provider transient / down** (connection refused, cooldown, starved/reasoning-eaten
   budget) — the content is describable; **retry later**, never permanently strand the doc.
2. **Genuinely blank / undescribable** — no retry will ever produce text; **accept the
   metadata-only stub as final** and stop spending compute.

A single provider's silence is ambiguous, so every fix (see "Integration bug") has to
*guess*. This design resolves the ambiguity uniformly across **all four enrichment
modalities** — image-describe, OCR-extract, video-analyze, audio-transcribe — by consulting
a second, independent model (a per-modality LiteLLM endpoint): if it **also** returns empty,
the input is genuinely blank; if it returns text, the primary failure was transient (and we
recover the content).

### Integration bug this supersedes

Merging PRs #59 + #60 + #61 together (each green in isolation) reintroduces the exact #0251
stranding #60 fixes. Confirmed on an integration branch:

- `_merge_degraded_ledger` (#60) charges a degraded-ledger attempt unless **every**
  degradation in a run is `transient=True`; at 5 attempts a doc is capped forever.
- `OllamaVisionOCR.describe()` (#59) catches `httpx.ConnectError`, notes `ocr_describe_failed`,
  and **returns `""`** — branched from `main` it notes with default `transient=False`, and by
  swallowing the exception never reaches #60's `is_transient(e)` classifier in `extract_image`.
- `extract_image` (#61) then sees `""` and notes `ocr_describe_empty`, also `transient=False`.
- Run degradations = `[ocr_describe_failed(F), ocr_describe_empty(F)]` → not all-transient →
  attempts charged every ~15-min run → doc caps at 5 → stranded. #60's own repro
  `test_provider_outage_never_caps_doc` fails with `assert 5 == 0`.

Diagnostic (reverted): flipping **both** notes to `transient=True` makes the repro pass;
flipping only #59's does not — two contributing causes. Rather than ship an interim
transient-flag heuristic the fallback would discard, the fallback design *is* the fix.

## Core principle: one pattern for every endpoint

**Every modality is the same three things** — a *primary* (whatever exists today), a
*LiteLLM fallback* at its own URL, and the *same decision rule*. Nothing is modality-specific
except the endpoint URL, the prompt, and the input encoding. No bespoke architecture per
endpoint.

```
        ┌──────────────── SHARED CORE (written once) ────────────────┐
        │ resolve_with_fallback(primary_call, fallback_call) -> str:  │
        │   primary raises transient   → re-raise      (unreachable)  │
        │   primary returns non-empty  → return it     (primary ok)   │
        │   primary returns "" (persistent, reachable):               │
        │       fallback is None       → raise transient (unconfirmed)│
        │       fallback returns text  → return it       (RECOVER)    │
        │       fallback returns ""    → return ""        (BLANK)     │
        │       fallback raises        → re-raise transient (both down)│
        │                                                             │
        │ LiteLLMFallback(endpoint, model, prompt).run(path) -> str   │
        │   one client (reuses providers/llm/litellm_llm.py patterns),│
        │   identical for all four; differs only by endpoint/model/   │
        │   prompt + input encoding (image/pdf/video/audio, which     │
        │   LiteLLM normalizes).                                       │
        └─────────────────────────────────────────────────────────────┘
             ▲              ▲               ▲               ▲
   image describe     ocr extract     video analyze    audio transcribe
   ${LITELLM_IMAGE_URL} ${LITELLM_OCR_URL} ${LITELLM_VIDEO_URL} ${LITELLM_AUDIO_URL}
```

## Locked decisions

1. **Purpose = recover + classify.** A successful fallback is indexed as the doc's real
   content (not a throwaway probe) *and* marks the primary failure transient.
2. **One uniform pattern for all four modalities** (image, OCR-extract, video, audio), each
   with its own LiteLLM fallback endpoint placeholder. Shared decision-rule helper + shared
   LiteLLM client; only thin per-protocol adapters differ.
3. **Trigger = reachable-but-empty ONLY.** If the primary is unreachable (connect-refused /
   cooldown), skip the fallback → mark transient → retry later. A whole-host outage (#0263)
   costs **$0** in fallback calls.
4. **Confirmed-blank (both models empty) = accept stub as final / clean.** Index the stub,
   drop from the degraded ledger, never re-run. **"Confirmed" requires a fallback to have
   actually run and agreed.** An *unconfirmed* empty (no fallback configured, or fallback
   unreachable) is **transient → retry**, never clean — else a single model's silence would
   silently strand describable content (#0264 failure mode).
5. **Cost guard is structural** — fallback fires only on reachable-but-empty; no
   counter/budget needed.
6. **The enrichment path is ALWAYS wrapped** (degenerate wrapper when no fallback configured)
   so the wrapper is the single outcome classifier in every deployment state, dark included.
7. **Reuse existing primaries as-is.** Audio's primary stays the existing OpenRouter
   whisper→voxtral→gemini provider; the wrapper only adds the uniform LiteLLM second opinion.
   Same for image (ollama), OCR (deepseek), video (openrouter single model).

## Architecture

### Shared core (modality-agnostic)
- **`resolve_with_fallback(primary_call, fallback_call)`** (`core/fallback.py`) — the pure
  decision rule above. Takes two zero-arg callables returning `str` and raising a transient
  error (satisfying `core.resilience.is_transient`) when unreachable. Returns `str` or
  re-raises transient. Fully unit-testable, zero I/O, zero modality knowledge.
- **`LiteLLMFallback`** (`providers/fallback/litellm_fallback.py`) — one client wrapping
  `{endpoint, model, prompt}`, `.run(path) -> str`, reusing `providers/llm/litellm_llm.py`
  patterns. Per-modality difference is only the endpoint/model/prompt and input encoding.

### Two thin protocol adapters (only because two protocols exist)
- **`FallbackOCRProvider`** (`providers/ocr/fallback.py`) implements `OCRProvider`; its
  `describe()` and `extract()` each call `resolve_with_fallback(primary.method, litellm.run)`.
- **`MediaFallbackProvider`** (`providers/media/fallback.py`) implements `MediaProvider`; its
  `analyze_video()` and `transcribe_audio()` do the same.

Each adapter holds **two** independent `LiteLLMFallback` clients — one per method with its own
endpoint (OCR: describe vs extract; media: video vs audio), exactly as the config models it.
Each adapter is a few lines — the logic lives in the shared core. Both factories
(`build_ocr_provider`, `build_media_provider`) **always** wrap their enrichment provider
(each method's fallback `None` unless configured). Wrappers **never import `extractors`**; they
communicate outcome purely via return/raise.

### Provider contract (uniform, required of every primary)
- **Unreachable** (connect refused / cooldown) → **raise** a transient error.
- **Reachable but empty** → **return `""`**.

`OllamaVisionOCR` (#59) already needs the change from "swallow to `''`" to "raise transient".
The OpenRouter media primary needs the same only for **`transcribe_audio`**: its all-fail path
raises `RuntimeError("All OpenRouter audio models failed")`, and `is_transient(RuntimeError)`
is **False** — so that site must raise a `TransientError` (or re-raise the chained underlying
`httpx` error) instead. `analyze_video` and `deepseek_ocr2` already honor the contract
(raise `httpx.ConnectError` on unreachable, return `""` on empty) and are unchanged.

## Outcome → ledger mapping (uniform across all callers)

| Outcome | Wrapper action | Caller sees | Ledger result |
|---|---|---|---|
| Recovered | returns fallback text | non-empty | clean — real content indexed |
| Confirmed blank (fallback ran, agreed) | returns `""` | empty | clean — **drops from ledger, never re-run** |
| Unconfirmed empty (dark / fallback down) | raises transient | `except` → transient note | attempts 0, **retries next run** |
| Primary unreachable | raises transient | `except` → transient note | attempts 0, retries next run |

This holds **only if every caller classifies transient uniformly** — the next section.

## Caller-side ledger reconciliation (makes all modalities uniform — and fixes a latent bug)

Today the enrichment callers classify inconsistently:

- `extract_image` — already `note_degradation(..., transient=is_transient(e))`. ✓
- `extract_video` — `note_skip("video_extract_failed")` = **permanent, never retried**. ✗
- `extract_audio` — `note_degradation(...)` default `transient=False`. ✗

Uniform change: **every** enrichment caller's `except` classifies via `is_transient(e)`, and
each drops its ad-hoc empty note (the wrapper now owns "empty" classification: unconfirmed
empty is raised transient and never arrives as `""`; a `""` that arrives is confirmed-blank →
clean). This both enables the uniform pattern and fixes video's silent permanent-skip on
outage.

## Changes to the five PRs

- **#59** — keep the 300s cooldown, but on connect-error / active cooldown **raise** a
  transient `ProviderUnavailable` (satisfies `is_transient`) instead of swallowing to `""`.
  Net: smaller.
- **#61** — **remove** the `extract_image` `ocr_describe_empty` note (safe *only* because the
  path is always wrapped). Keep `_record_single_doc_outcome` and
  `scripts/backfill_unledgered_stub_docs.py`. #61's `audio_transcript_empty` /
  `video_analysis_empty` notes are removed here too under the same uniform rule.
- **#60** — **unchanged.** Its ledger transient semantics are the foundation.
- **#57, #58** — orthogonal (health probe / Lance storage); unaffected.

## Configuration (perfectly regular)

```yaml
ocr:
  provider: ollama_vision            # image describe primary
  describe:
    fallback: { provider: litellm, endpoint: "${LITELLM_IMAGE_URL}", model: "..." }
  extract:
    provider: deepseek_ocr2          # ocr extract primary
    fallback: { provider: litellm, endpoint: "${LITELLM_OCR_URL}", model: "..." }
media:
  provider: openrouter               # video + audio primary (audio keeps its model list)
  video:
    fallback: { provider: litellm, endpoint: "${LITELLM_VIDEO_URL}", model: "..." }
  audio:
    fallback: { provider: litellm, endpoint: "${LITELLM_AUDIO_URL}", model: "..." }
```

- Any `fallback` absent → that provider is still wrapped, `fallback=None` (dark). No paid
  call; empties retry as transient. Ships dark and is the #0251/#0264 fix even before
  endpoints are wired — **not** byte-identical to `main`, but identical in contacting no
  external provider.
- `model` is **required** in each fallback block (no silent default); a missing model is a
  startup config error.
- New LiteLLM adapters are registered in **both** factories: `build_ocr_provider` and
  `build_media_provider` (the latter currently only accepts `provider: openrouter`).

## Error handling

- Primary unreachable → transient raise, **fallback not called** (asserted in tests, per
  modality).
- Fallback unreachable / misconfigured / endpoint down → treated as unreachable (transient),
  **never** as blank — a config error must not silently mark inputs blank.
- Primary short empty-retries run *inside* the primary before "reachable-but-empty," so a
  one-off starved response doesn't reach the fallback as empty.
- No import cycle: wrappers never import `extractors`.

## Testing

- **Shared core** `resolve_with_fallback` unit tests, all branches: primary-unreachable →
  raise + fallback-not-called; primary-text → passthrough; dark (`fallback=None`) + empty →
  raise transient (retry, not clean — #0264 guard); reachable-empty + fallback-text → recover;
  + fallback-empty → confirmed blank `""`; + fallback-unreachable → raise transient.
- **Each of the four wrappers** (`FallbackOCRProvider.describe`, `.extract`,
  `MediaFallbackProvider.analyze_video`, `.transcribe_audio`) exercises the same matrix via
  the shared core (thin — mostly delegation + the cost-guard "fallback not called on
  unreachable" assertion).
- **Caller ledger reconciliation**: `extract_video` outage → transient (retries, no longer a
  permanent skip); `extract_audio` all-down → transient; the resurrected
  `test_provider_outage_never_caps_doc` passes; reachable-empty → recovered content indexed;
  confirmed-blank → drops from ledger.
- Reconcile the 6 contract-drift tests (`test_ocr.py`, `test_extractors.py`) to the
  `Degradation` shape as part of #59/#61 changes.
- Full `make gate` on the rebuilt integration branch (live tier remains preflight-blocked on
  this host, unchanged).

## Rollout & sequencing

1. Build shared core + LiteLLM client + two adapters + factory wiring + #59/#61 reconciliation
   + caller classification, on `feat/describe-fallback-disambiguation`.
2. Rebuild the integration branch from the reconciled PRs; `make gate` green.
3. Ship **dark** (no fallback endpoints set): no paid provider contacted; #0251/#0264
   regression fixed (unconfirmed empties retry as transient everywhere).
   **Operator-visible cost of dark mode:** because no fallback can *confirm* blank, every
   empty result across all four modalities — including genuine blanks (blank scan pages,
   silent/short audio, textless images) — is retried every run. This deliberately trades the
   #0251 *capping* failure for *re-processing* of true-blanks until endpoints are wired. Given
   the standing "silent re-processing is why indexing never catches up" concern, keep step 4
   close behind the dark ship; only a configured fallback lets true-blanks settle to clean.
4. Operator stands up the four LiteLLM endpoints and sets the URLs/models (do this promptly
   after the dark ship to stop true-blank re-processing).
5. Only then does the backfill of confirmed-blank / recoverable stubs become meaningful;
   `backfill_unledgered_stub_docs.py` apply stays deferred until endpoints + a healthy primary
   (per #0264 sequencing).

## Open questions / non-goals

- **Non-goal:** "similar answer" comparison for non-empty primary output (only empty/raise
  triggers fallback).
- **Non-goal:** budget/rate counter (structural guard suffices).
- **Non-goal:** merging the `OCRProvider` and `MediaProvider` protocols into one (bigger
  refactor; the two thin adapters over a shared core already deliver the uniformity).
- **To confirm during planning:** exact `ProviderUnavailable` type vs re-raising the
  underlying `httpx.ConnectError` (must satisfy `is_transient`); the input-encoding surface of
  `LiteLLMFallback.run` across image/pdf/video/audio; the OpenRouter media primary's
  raise-on-unreachable vs return-empty contract edit.
