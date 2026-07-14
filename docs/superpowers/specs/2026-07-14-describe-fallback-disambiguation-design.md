# Describe-Fallback Disambiguation — Design

- **Date:** 2026-07-14
- **Status:** Approved (design); pending spec review
- **Owner:** dpark@nexsler.com
- **Related tickets:** #0251 (transient outage caps docs), #0264 (stub-indexed images unledgered), #0263 (ollama vision host outage), infra #0240 (host restart)
- **Related PRs:** #57, #58, #59, #60, #61 (the doc-organizer outage cluster)

## Problem

Image indexing describes an image with a vision model (`OllamaVisionOCR`, qwen3-vl on the
Mac Mini). When `describe()` yields no text, the pipeline cannot tell **why**, and the two
causes require opposite handling:

1. **Provider transient / down** (connection refused, cooldown, starved/reasoning-eaten
   budget) — the content is describable; we must *retry later* and never permanently
   strand the doc.
2. **Genuinely blank / undescribable image** — no retry will ever produce text; we should
   *accept the metadata-only stub as final* and stop spending compute on it.

Today a single provider's silence is ambiguous, so every fix (see "Integration bug"
below) has to *guess*. This design resolves the ambiguity by consulting a second,
independent model: if an independent model **also** returns empty, the image is genuinely
blank; if it returns text, the primary failure was transient (and we recover the content).

### Integration bug this supersedes

Merging PRs #59 + #60 + #61 together (each green in isolation) reintroduces the exact
#0251 stranding #60 fixes. Confirmed on an integration branch:

- `_merge_degraded_ledger` (#60) charges a degraded-ledger attempt unless **every**
  degradation in a run is `transient=True`; at 5 attempts a doc is capped forever.
- `OllamaVisionOCR.describe()` (#59) catches `httpx.ConnectError`, notes
  `ocr_describe_failed`, and **returns `""`** — but branched from `main` it notes with the
  default `transient=False`, and by swallowing the exception it never reaches #60's
  `is_transient(e)` classifier in `extract_image`.
- `extract_image` (#61) then sees `""` and notes `ocr_describe_empty`, also
  `transient=False`.
- Run degradations = `[ocr_describe_failed(F), ocr_describe_empty(F)]` → not all-transient
  → attempts charged every ~15-min run → doc caps at 5 → stranded. `#60`'s own repro
  `test_provider_outage_never_caps_doc` fails with `assert 5 == 0`.

Diagnostic (reverted): flipping **both** notes to `transient=True` makes the repro pass;
flipping only #59's does not — proving two contributing causes. Rather than ship an interim
transient-flag heuristic that the fallback would discard, the fallback design *is* the fix.

## Locked decisions

1. **Purpose = recover + classify.** A successful fallback describe is indexed as the
   doc's real content (not a throwaway probe) *and* marks the primary failure transient.
2. **Provider = per-modality LiteLLM endpoints.** In scope for this spec: image
   (`${LITELLM_IMAGE_URL}`) and OCR-extract (`${LITELLM_OCR_URL}`), built via the existing
   `build_ocr_provider` factory. Video (`${LITELLM_VIDEO_URL}`) is documented as intent but
   wired in the media follow-on (see "Media path"). Operator supplies the URLs later.
3. **Trigger = reachable-but-empty ONLY.** If the primary is unreachable
   (connect-refused / in cooldown), skip the fallback entirely → mark transient → retry
   later. A whole-host outage (#0263) costs **$0** in fallback calls.
4. **Confirmed-blank (both models empty) = accept stub as final / clean.** Index the
   metadata-only stub, drop the doc from the degraded ledger, never re-describe.
   **"Confirmed" requires a fallback to have actually run and agreed.** An *unconfirmed*
   empty (no fallback configured, or the fallback was unreachable) is **transient →
   retry**, never clean — otherwise a single model's silence would silently strand a
   describable image (the #0264 failure mode).
5. **Cost guard is structural** — because fallback fires only on reachable-but-empty,
   no counter/budget is required. (A budget backstop was considered and rejected as
   unnecessary given the structural guard.)
6. **The describe path is ALWAYS wrapped** (degenerate wrapper when no fallback is
   configured) so the wrapper is the single describe-outcome classifier in every
   deployment state, including dark mode.

## Architecture (Approach A: wrapper provider)

A new `FallbackOCRProvider` (`providers/ocr/fallback.py`) implements `OCRProvider`,
wrapping `{primary, fallback}` where `fallback` may be `None` (degenerate/dark mode).
It composes with the existing factory and `CompositeOCRProvider`. `build_ocr_provider`
**always** wraps the describe provider in a `FallbackOCRProvider` (fallback `None` unless
configured), so the wrapper is the sole describe-outcome classifier in every state.

The wrapper **never imports `extractors`** — it communicates outcome purely via
return/raise, and the existing `extract_image` `except` block (from #60) performs ledger
classification. Describe-outcome classification thus lives in exactly one place. This is
why `extract_image` can safely drop #61's `ocr_describe_empty` note: an unconfirmed empty
no longer *reaches* `extract_image` as `""` — the wrapper raises transient for it.

### Decision flow — `FallbackOCRProvider.describe(path)`

The primary's own short empty-retry loop (`_DESCRIBE_EMPTY_RETRIES`) runs first, inside
`primary.describe()`, so "reachable-but-empty" already means *persistently* empty — this
reduces false-blank risk from a one-off starved response.

```
text = primary.describe(path)          # RAISES a transient error on unreachable/cooldown
  ├─ raises transient  → re-raise                 → extract_image marks transient, NO fallback ($0 in outage)
  ├─ returns non-empty → return text              → primary content, done
  └─ returns "" (persistently) → reachable-but-empty:
        if fallback is None →  RAISE transient     → unconfirmed empty: retry later (dark mode safe)
        ft = fallback.describe(path):
          ├─ raises transient  → re-raise          → marks transient (both down), retry later
          ├─ returns non-empty → return ft         → RECOVERED: real content indexed, clean
          └─ returns ""        → return ""         → CONFIRMED BLANK: accept stub, clean
```

Key invariant: the wrapper returns `""` **only** on fallback-confirmed blank. Every
*unconfirmed* empty (dark mode or fallback-down) is a transient **raise**, so it retries
and is never silently dropped.

The same wrapper handles `extract()` for the OCR-extract modality (same `OCRProvider`
interface, its own endpoint). Video/audio use a **separate** protocol — see "Media path"
below. "Similar answer" comparison is intentionally NOT implemented for non-empty primary
output — the trigger is empty-or-raise only (YAGNI).

### Outcome → ledger mapping (reuses #60, zero new coupling)

| Outcome | Wrapper action | `extract_image` sees | Ledger result |
|---|---|---|---|
| Recovered | returns fallback text | non-empty | clean — real content indexed |
| Confirmed blank (fallback ran, agreed) | returns `""` | empty | clean — **drops from ledger, never retried** |
| Unconfirmed empty (dark mode / fallback down) | raises transient error | `except` → `note_degradation(..., transient=True)` | attempts stay 0, **retries next run** |
| Primary unreachable | raises transient error | `except` → `note_degradation(..., transient=True)` | attempts stay 0, retries next run |

## Changes to the five PRs

- **#59** — keep the 300s cooldown (don't hammer a dead host), but on connect-error /
  active cooldown **raise** a transient `ProviderUnavailable` error (classified transient
  by `core.resilience.is_transient`) instead of swallowing to `""`. Restores the signal
  the wrapper and #60 depend on. Net effect: #59 becomes smaller.
- **#61** — **remove** the `extract_image` `ocr_describe_empty` note. This is safe **only
  because** the describe path is always wrapped: an unconfirmed empty is raised as
  transient by the wrapper (retry) and never reaches `extract_image` as `""`; a `""` that
  does reach `extract_image` is fallback-confirmed blank → clean. Keep
  `_record_single_doc_outcome` (single-doc-path ledgering) and
  `scripts/backfill_unledgered_stub_docs.py`. #61's analogous `audio_transcript_empty` /
  `video_analysis_empty` notes are left **as-is** here and reconciled in the media follow-on
  (they are not on the image-describe path this spec fixes).
- **#60** — **unchanged.** Its ledger transient semantics are the foundation.
- **#57, #58** — orthogonal (health probe / Lance storage); unaffected.

## Configuration

```yaml
ocr:
  provider: ollama_vision
  describe:
    provider: ollama_vision
    model: qwen3-vl:8b
    fallback:                            # NEW — omit to disable (exact current behavior)
      provider: litellm
      endpoint: "${LITELLM_IMAGE_URL}"   # placeholder, operator-supplied
  extract:
    fallback: { provider: litellm, endpoint: "${LITELLM_OCR_URL}", model: "..." }
# media.video.fallback (endpoint "${LITELLM_VIDEO_URL}") — follow-on spec, see "Media path"
```

- Fallback subsection **absent** → the describe provider is still wrapped, but with
  `fallback=None` (degenerate). Feature ships **dark**: no paid fallback calls, and empties
  retry as transient. This is **not** byte-identical to today's `main` — it is the intended
  #0251/#0264 fix (empties no longer cap or silently drop); it *is* identical in that no
  external/paid provider is contacted until an endpoint is configured.
- A `litellm` OCR-provider adapter is added to the factory (reusing
  `providers/llm/litellm_llm.py` patterns). `model` is required in the fallback config (no
  implicit default) so the operator's per-modality endpoint/model are explicit; a missing
  `model` is a config error surfaced at startup, not a silent default.

### Media path (video / audio) — deferred to a follow-on spec

`extract_video` / `extract_audio` do **not** use `OCRProvider` / `build_ocr_provider` —
they go through the separate `MediaProvider` protocol (`analyze_video` / `transcribe_audio`
in `providers/media/base.py`) built by `build_media_provider`. The OCR wrapper cannot wrap
them, and — critically — the media *callers* classify degradations differently from
`extract_image`, so the "reuses #60, zero coupling" property below does **not** transfer:

- `extract_video`'s except calls `note_skip("video_extract_failed")` — a **permanent** skip
  (never retried), not a transient degradation.
- `extract_audio`'s except calls `note_degradation(...)` with the default `transient=False`.

A media fallback that raises transient would therefore be mis-ledgered (permanent skip for
video; non-transient for audio) — the opposite of the intended retry. Making media correct
requires its own caller-side edits (classify via `is_transient` in `extract_video` /
`extract_audio`, relocate the empty notes) **plus** a new LiteLLM `MediaProvider` adapter
(`build_media_provider` currently only accepts `provider: openrouter`).

Because that is a distinct subsystem with its own ledger reconciliation — and the #0251 /
#0264 regression that blocks the PR cluster is entirely on the image-describe path — the
media (video/audio) fallback is **deferred to a named follow-on spec**
(`media-fallback-disambiguation`) that reuses the same decision rule (extracted into a
shared helper) but specifies the media caller edits explicitly. The `${LITELLM_VIDEO_URL}`
placeholder is documented here as intent; its wiring lands in that spec.

## Error handling

- **Primary unreachable** → transient raise, no fallback call (asserted in tests).
- **Fallback unreachable** → transient raise; the doc retries next run.
- **Fallback misconfigured / endpoint down at startup** → treated as unreachable
  (transient), never as blank — a config error must not silently mark images blank.
- **Import cycle** — avoided entirely: the wrapper does not import `extractors`.

## Testing

- `FallbackOCRProvider` unit tests, all branches:
  - primary-unreachable → raises transient AND **fallback not called** (cost-guard assertion)
  - primary returns text → passthrough, fallback not called
  - **dark mode (`fallback=None`) + empty → raises transient** (retry, not clean — #0264 guard)
  - reachable-but-empty + fallback text → returns fallback text (recovered)
  - reachable-but-empty + fallback empty → returns `""` (confirmed blank)
  - reachable-but-empty + fallback unreachable → raises transient
- (Media wrapper tests belong to the follow-on media-fallback spec.)
- Integration:
  - resurrected **`test_provider_outage_never_caps_doc`** passes (outage → transient →
    attempts 0)
  - reachable-but-empty → recovered content lands in the index
  - confirmed-blank → doc drops from degraded ledger and is not retried
- Reconcile the 6 contract-drift tests (`test_ocr.py`, `test_extractors.py`) to the
  `Degradation` shape as part of #59/#61 changes.
- Full `make gate` on the rebuilt integration branch (static → unit → integration →
  staging-e2e; live tier remains preflight-blocked on this host, unchanged).

## Rollout & sequencing

1. Build the wrapper + factory wiring + #59/#61 reconciliation on
   `feat/describe-fallback-disambiguation`.
2. Rebuild the integration branch from the reconciled PRs; `make gate` green.
3. Ship **dark** (no fallback endpoints configured): no paid provider is contacted, and
   the #0251/#0264 regression is fixed — unconfirmed empties retry as transient (wrapper
   raise + #59 raising transient on unreachable) instead of capping or silently dropping.
4. Operator stands up the LiteLLM per-modality endpoints and sets the placeholder URLs.
5. Only then does the backfill of confirmed-blank / recoverable stubs become meaningful;
   `backfill_unledgered_stub_docs.py` apply stays deferred until after endpoints + a
   healthy primary (per #0264 sequencing).

## Open questions / non-goals

- **Non-goal:** "similar answer" comparison for non-empty primary output (only empty/raise
  triggers fallback).
- **Non-goal:** budget/rate counter (structural guard suffices).
- **Scope for this spec:** OCR image-describe + OCR-extract (the `OCRProvider` path) only.
  Video/audio (`MediaProvider` path) are **deferred to the `media-fallback-disambiguation`
  follow-on** (different protocol + ledger classification + a new media adapter — see
  "Media path"). This keeps the spec to a single focused plan and fixes the actual
  regression fastest.
- **To confirm during planning:** exact `ProviderUnavailable` type vs re-raising the
  underlying `httpx.ConnectError` (must satisfy `is_transient`); the shared decision-rule
  helper's signature (so the follow-on media wrapper can reuse it).
