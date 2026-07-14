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
2. **Provider = per-modality LiteLLM endpoints** (image, OCR-extract, video), configured
   with placeholder URLs (`${LITELLM_IMAGE_URL}`, `${LITELLM_OCR_URL}`,
   `${LITELLM_VIDEO_URL}`) the operator wires later. Built via the existing
   `build_ocr_provider` factory.
3. **Trigger = reachable-but-empty ONLY.** If the primary is unreachable
   (connect-refused / in cooldown), skip the fallback entirely → mark transient → retry
   later. A whole-host outage (#0263) costs **$0** in fallback calls.
4. **Confirmed-blank (both models empty) = accept stub as final / clean.** Index the
   metadata-only stub, drop the doc from the degraded ledger, never re-describe.
5. **Cost guard is structural** — because fallback fires only on reachable-but-empty,
   no counter/budget is required. (A budget backstop was considered and rejected as
   unnecessary given the structural guard.)

## Architecture (Approach A: wrapper provider)

A new `FallbackOCRProvider` (`providers/ocr/fallback.py`) implements `OCRProvider`,
wrapping `{primary, fallback}`. It composes with the existing factory and
`CompositeOCRProvider`; one instance per modality, each pointed at its LiteLLM endpoint.

The wrapper **never imports `extractors`** — it communicates outcome purely via
return/raise, and the existing `extract_image` `except` block (from #60) performs ledger
classification. Describe-outcome classification thus lives in exactly one place.

### Decision flow — `FallbackOCRProvider.describe(path)`

```
text = primary.describe(path)          # RAISES a transient error on unreachable/cooldown
  ├─ raises transient  → re-raise                 → extract_image marks transient, NO fallback ($0 in outage)
  ├─ returns non-empty → return text              → primary content, done
  └─ returns ""        → reachable-but-empty → fallback.describe(path):
        ├─ raises transient  → re-raise           → marks transient (both down), retry later
        ├─ returns non-empty → return fallback_text  → RECOVERED: real content indexed, clean
        └─ returns ""        → return ""           → CONFIRMED BLANK: accept stub, clean
```

The same wrapper handles `extract()` for the OCR-extract modality and is reused by the
video-analysis path (its own endpoint). "Similar answer" comparison is intentionally
NOT implemented for non-empty primary output — the trigger is empty-or-raise only (YAGNI).

### Outcome → ledger mapping (reuses #60, zero new coupling)

| Outcome | Wrapper action | `extract_image` sees | Ledger result |
|---|---|---|---|
| Recovered | returns fallback text | non-empty | clean — real content indexed |
| Confirmed blank | returns `""` | empty | clean — **drops from ledger, never retried** |
| Transient (primary or fallback down) | raises transient error | `except` → `note_degradation(..., transient=True)` | attempts stay 0, retries next run |

## Changes to the five PRs

- **#59** — keep the 300s cooldown (don't hammer a dead host), but on connect-error /
  active cooldown **raise** a transient `ProviderUnavailable` error (classified transient
  by `core.resilience.is_transient`) instead of swallowing to `""`. Restores the signal
  the wrapper and #60 depend on. Net effect: #59 becomes smaller.
- **#61** — **remove** the `extract_image` `ocr_describe_empty` note (superseded: empty now
  means confirmed-blank → clean). Keep `_record_single_doc_outcome` (single-doc-path
  ledgering) and `scripts/backfill_unledgered_stub_docs.py`. Re-evaluate the analogous
  `audio_transcript_empty` / `video_analysis_empty` notes under the same wrapper rule.
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
    fallback: { provider: litellm, endpoint: "${LITELLM_OCR_URL}" }
# video-analysis media provider: fallback endpoint "${LITELLM_VIDEO_URL}"
```

- Fallback subsection **absent** → `build_ocr_provider` returns the bare primary (no
  wrapper). Feature ships **dark**; behavior is byte-identical to today until endpoints
  are wired.
- A `litellm` OCR-provider adapter is added to the factory (reusing
  `providers/llm/litellm_llm.py` patterns) so `endpoint`/model resolve per modality.

## Error handling

- **Primary unreachable** → transient raise, no fallback call (asserted in tests).
- **Fallback unreachable** → transient raise; the doc retries next run.
- **Fallback misconfigured / endpoint down at startup** → treated as unreachable
  (transient), never as blank — a config error must not silently mark images blank.
- **Import cycle** — avoided entirely: the wrapper does not import `extractors`.

## Testing

- `FallbackOCRProvider` unit tests, all four branches:
  - primary-unreachable → raises transient AND **fallback not called** (cost-guard assertion)
  - primary returns text → passthrough, fallback not called
  - reachable-but-empty + fallback text → returns fallback text (recovered)
  - reachable-but-empty + fallback empty → returns `""` (confirmed blank)
  - reachable-but-empty + fallback unreachable → raises transient
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
3. Ship **dark** (no fallback endpoints configured) — identical to current behavior, but
   with the #0251 regression fixed via #59 raising transient.
4. Operator stands up the LiteLLM per-modality endpoints and sets the placeholder URLs.
5. Only then does the backfill of confirmed-blank / recoverable stubs become meaningful;
   `backfill_unledgered_stub_docs.py` apply stays deferred until after endpoints + a
   healthy primary (per #0264 sequencing).

## Open questions / non-goals

- **Non-goal:** "similar answer" comparison for non-empty primary output (only empty/raise
  triggers fallback).
- **Non-goal:** budget/rate counter (structural guard suffices).
- **To confirm during planning:** exact `ProviderUnavailable` type vs re-raising the
  underlying `httpx.ConnectError`; whether audio gets a fallback endpoint now or later.
