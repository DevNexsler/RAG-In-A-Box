# Attachment retrieval — handoff for independent review (2026-07-22)

Reviewer: verify the claims below. Where I was wrong earlier in the session I
have said so explicitly — those corrections are themselves worth re-checking.

## Original problem

A video posted in #Tenant-Showings ("Rafael Boundurant / 16 N Main /
Phillipsburg NJ 08865 / I turned the flash on camera because it's no electric")
could not be found by the agent. User's premise: "every attachment, we take
message and before and after so we have enrichment label of what that photo or
video is for."

## Root causes found (three distinct, all evidenced)

1. **Expired-attachment stub loop.** Zoho link expiry writes a 41-byte JSON
   error envelope in place of the media; it reached the vision provider every
   run (`video_extract_failed` x9). Fixed earlier in session (`_is_nonmedia_stub`).
2. **Candidate-pool attrition (two separate cutoffs).**
   - a. Re-rank pool is `fused[:top_k*6]` (=60). An attachment below that is
     deleted before scoring. Boosting is a no-op on it. → commit `2e34b03`.
   - b. `source_type` is pushed down into *retrieval*, so filtering changes what
     is FETCHED. Unfiltered, the fixed 50-vector + 50-keyword rows can be 100%
     messages and zero attachments are retrieved. → commit `e112c9a`.
3. **Context stored but not embedded (images).** Context is extracted into
   metadata but never folded into the searchable `text` for images.

## Shipped this session (committed, pushed, deployed)

- `2e34b03` — `_ensure_media_intent_slots` (reserve slots for a named medium) +
  snapshot the candidate pool BEFORE narrowing.
- `e112c9a` — targeted per-type recall retrieval when nothing of the named
  medium was fetched; rescale injected scores onto the result-set scale.

Deployed via local docker-compose rebuild+recreate. Container healthy; deployed
`search_hybrid.py` verified to match HEAD.

## Eval results

Harness: `/tmp/claude-1000/.../scratchpad/eval_attach.py {dev|holdout|holdout2|all}`.
Pass = expected attachment doc_id appears in top 10 of a live `/api/search`.

| Set | Before | After |
|---|---|---|
| dev (2) | 1/2 | 2/2 |
| holdout1 (3) | 1/3 | 3/3 |
| holdout2 (14, blind) | 9/14 | 12/14 |
| total | — | **17/19** |

holdout2 was sampled from prod with a deterministic stride over attachments
having >=25 chars of stored context, across img/video/audio; queries were
written from stored context BEFORE running any of them.

### Caveats a reviewer should press on

- **holdout2's "9/14 before" was measured with `2e34b03` already deployed**, not
  against a clean no-fix baseline. There is no true zero-fix number for holdout2.
- **Rank-9/10 passes are guaranteed-slot injections**, not organic ranking wins.
  Roughly: 163 Washington (r10), Laura pics (r9), tenant warning (r9), 1110
  Gaspar (r9), Joycelyn eviction (r9), online payments (r10), Dan/Jared (r9).
  If you consider injection-into-the-tail not a real "find", the organic number
  is much lower. This is a design choice, not an accident — state your view.
- **3 queries errored (connection reset) mid-eval and passed on retry**; the
  17/19 counts those as passes. Container showed `restarts=0`, `OOMKilled=false`.
- **Image passes are on visual/OCR content, not context** (see below).
- `media_intent_weight` / `media_intent_slots` live in `config.yaml`, which is
  bind-mounted and `skip-worktree` (deploy-local, NOT committed). Code defaults
  are 0.35 / 2.

## The significant open finding

Counted directly in `chunks.lance` — context present in metadata vs actually
present in embedded `text`:

| type | embedded | NOT embedded |
|---|---|---|
| img | **0** | **1,460** |
| video | 9 | 2 |
| audio | 1 | 5 |

**Image-context search is 0% functional corpus-wide.** A photo cannot be found
by what was said around it. Example: `documents::000RY` has
`ctx_places = '482 #6, 163 Washington'` in metadata, but `'482'` and `'Cesar'`
appear nowhere in its embedded text (which is only the visual describe). The
videos are embedded only because they were re-indexed by hand this session.

Gotcha for anyone re-measuring: there are TWO context formats in the corpus —
`[Conversation context]` and `Conversation context for documents::<id>:`. Grep
for `conversation context` unbracketed or you get a false 0% for video/audio too.

Backfill: no describe cache exists, so a plain re-index re-runs the vision model
on all 1,460 images (`ocr.concurrency: 1`, effectively serial). Cheaper path:
describe text is already in `chunks.lance` — append context from metadata,
re-embed, `upsert_nodes()`. Embeddings only, no vision calls. NOT yet run;
awaiting user approval.

## Things I got wrong during this session (please re-check)

1. Claimed the 2 image failures were "genuine relevance misses — query terms
   don't match stored text well." **Wrong.** They are stale-index misses; the
   terms are absent from embedded text entirely.
2. Twice diagnosed the failure as a *ranking* problem and tuned a boost weight
   (0.35 -> 1.0, then moved the boost earlier). Both were no-ops. The tell I
   missed: top-3 scores returned byte-identical across all attempts, meaning the
   item was never in the pool.
3. First corpus count used the bracketed marker only and reported a false 0%
   for video/audio.
4. Earlier in session: deployed from a stale `main` (20 commits behind), briefly
   regressing prod; rolled back and re-deployed correctly.

## Also verified this session (independent of the search work)

- **LiteLLM OCR/media is working**, tested live through our own provider code:
  `extract` 1,463 / 1,463 / 521 chars of accurate text; `describe` 3,015 chars;
  `transcribe_audio` 2,271 chars (this was the previously-broken whisper path).
  One image returned 0 chars from `extract` — it is a textless kitchen photo,
  confirmed by its describe.
- **Disk**: host had hit 90.2% vs a 90.0% threshold, so prod `/health` was
  503ing `disk_full` and 5 health-probe unit tests failed. Pruned Docker
  cache/images: 96% -> 81% root usage; `/health` returns `ok`; tests green.
- Full unit tier: 1,319 passed, 0 failed.

## Suggested checks for the reviewer

- Re-run `eval_attach.py all` and see whether 17/19 reproduces.
- Decide whether tail-injection should count as a "find"; if not, re-score.
- Verify the img 0/1,460 count independently (watch the two-format gotcha).
- Sanity-check that queries naming no medium are untouched (I verified live on
  "163 Washington rent adjustment" and "lease renewal terms for tenant" — all
  10 results were messages, no injection).
- Review `_ensure_media_intent_slots` eviction logic: it drops the weakest
  NON-matching hits so already-present matches are never evicted (a unit test
  caught a bug here where a blind slice evicted a matching hit).
