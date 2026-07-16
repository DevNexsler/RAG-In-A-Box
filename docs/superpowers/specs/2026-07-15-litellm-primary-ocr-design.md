# LiteLLM-Primary OCR and Image Description Design

Date: 2026-07-15

## Context

Production OCR currently calls two services on the Mac Mini directly:

- DeepSeek-OCR2 at `http://192.168.68.70:8790` for scanned-page text extraction.
- Ollama `qwen3-vl:8b` at `http://192.168.68.70:11434` for standalone-image description.

Each method also has a LiteLLM fallback. That fallback runs only when its direct primary
returns an empty result. A connection error intentionally propagates without invoking the
fallback, so an outage of either direct service stops that modality even while LiteLLM is
healthy.

LiteLLM already exposes the `ocr` and `vision` model aliases and owns their local-to-cloud
fallback policy. RAG-in-a-Box should therefore call LiteLLM as the sole OCR orchestration
boundary instead of duplicating routing policy in the application.

## Goals

- Route scanned-page OCR through LiteLLM model alias `ocr`.
- Route standalone-image description through LiteLLM model alias `vision`.
- Remove direct Mac Mini OCR and Ollama endpoints from active runtime configuration.
- Let LiteLLM own local/cloud selection, retries, and fallback ordering.
- Preserve existing OCR provider contracts: text on success, empty string for confirmed
  blank content, transient exception when the orchestration endpoint fails.
- Keep credentials in environment variables, never `config.yaml`.

## Non-goals

- Change LiteLLM proxy routing or model definitions.
- Change media, embedding, reranking, enrichment, or text-LLM providers.
- Tune OCR concurrency or model quality.
- Broaden paid real-provider testing beyond the two explicitly approved OCR/vision smoke
  requests for this rollout.
- Retain application-level direct-provider fallback beneath LiteLLM.

## Considered Approaches

### 1. First-class LiteLLM OCR provider — selected

Add a normal `OCRProvider` implementation backed by LiteLLM. It owns separate `ocr` and
`vision` model clients and is selected with `ocr.provider: litellm`.

Benefits: one routing authority, no dead-primary delay, explicit configuration, reusable
provider contract. Cost: small adapter and factory change.

### 2. Invoke existing fallback on direct-primary exceptions

Change `resolve_with_fallback` so direct provider outages trigger LiteLLM.

Rejected: every request still touches the dead service first; application and LiteLLM both
own failover policy; change affects other modality wrappers and reverses a deliberate
no-fanout rule.

### 3. Configure an empty/null primary to force the current fallback

Keep LiteLLM labeled as a fallback and make a synthetic primary return empty.

Rejected: misleading semantics, special blank-confirmation behavior, and unnecessary
request indirection.

## Architecture

Add `LiteLLMOCR` under `providers/ocr/`. It implements `OCRProvider` and composes two
instances of the existing OpenAI-compatible LiteLLM request client:

- `extract(file_path, page)` uses model alias `ocr` and the existing verbatim
  transcription prompt.
- `describe(file_path)` uses model alias `vision` and the existing detailed-search
  description prompt.

Both calls encode the input as an OpenAI image content part and POST to
`{endpoint}/chat/completions`. `page` remains part of the provider interface; upstream PDF
handling supplies the rendered page image, so the LiteLLM request uses `file_path`.

`build_ocr_provider` recognizes top-level `provider: litellm`, validates both model aliases,
and returns `LiteLLMOCR` directly. It intentionally does not wrap this provider in
`FallbackOCRProvider`: LiteLLM is the fallback authority, and a successful empty response
means its configured routing policy completed without content. Existing providers retain
their current wrapper and empty-result disambiguation behavior.

No changes are required to `resolve_with_fallback`, `FallbackOCRProvider`, or LiteLLM proxy
configuration.

## Configuration

Active configuration becomes:

```yaml
ocr:
  enabled: true
  provider: "litellm"
  endpoint: "http://192.168.68.87:4000/v1"
  extract_model: "ocr"
  describe_model: "vision"
  timeout: 300
  concurrency: 1
```

The direct `192.168.68.70` endpoints, split `extract`/`describe` primary blocks, and nested
application-level LiteLLM fallback blocks are removed from active OCR configuration.
`config.yaml.example` documents the new first-class form.

Authentication follows existing LiteLLM behavior: explicit provider key if ever supplied,
otherwise `LITELLM_API_KEY`, then `LITELLM_MASTER_KEY`. No secret is added to tracked YAML.

## Result and Error Semantics

- Non-empty LiteLLM response: return stripped text.
- Successful empty LiteLLM response: return `""` as confirmed blank after proxy routing.
- Connection error, timeout, or 5xx after bounded retries: propagate a transient error.
- Authentication, model-alias, malformed-response, or other non-transient client error:
  normalize to a transient orchestration error so the document retries after configuration
  repair instead of being permanently capped or indexed as blank.

The adapter reuses the existing LiteLLM client retry, auth, encoding, and error-normalization
rules instead of cloning HTTP logic.

## Observability

Startup logging records endpoint plus both alias names without credentials. Request failures
identify the affected alias (`ocr` or `vision`). LiteLLM remains the source of truth for
which local or cloud backend was selected.

The application `/health` endpoint remains a process/indexer probe; provider validation uses
LiteLLM health/model metadata plus the approved live end-to-end test.

## Testing

Follow TDD for adapter and factory behavior.

Unit tests cover:

- `extract()` sends `model: ocr`, transcription prompt, and image data URL.
- `describe()` sends `model: vision`, description prompt, and image data URL.
- endpoint, timeout, and environment-based auth reach the shared client.
- empty content returns `""`.
- transport and proxy errors retain transient retry semantics.
- factory builds `LiteLLMOCR` from top-level configuration and rejects missing model aliases.
- first-class LiteLLM configuration is not wrapped in application-level OCR fallback.

Repository verification:

1. Run focused OCR/LiteLLM unit tests during red-green development.
2. Run `make gate-fast` after implementation.
3. Confirm LiteLLM liveness and alias metadata without generation spend.
4. Run a live end-to-end test through the new provider and real LiteLLM endpoint for both
   aliases. A generated fixture contains a unique printed token plus simple visual content;
   the test asserts `ocr` extracts the token and `vision` describes the expected visual
   concepts. This rollout has explicit operator approval for any real-provider spend caused
   by LiteLLM local/cloud routing.
5. Rebuild/restart `doc-organizer`, then check container health and `/health`.

## Rollout

1. Add provider and tests.
2. Update active and example configuration.
3. Run GitNexus change detection and repository gates.
4. Rebuild/restart `doc-organizer` so its image includes the adapter and it reloads mounted
   configuration.
5. Verify application health and inspect startup logs for endpoint and aliases.

Rollback restores the previous OCR configuration and prior application image, then restarts
`doc-organizer`. LiteLLM proxy state is unchanged by either rollout or rollback.

## Acceptance Criteria

- Runtime config contains no direct Mac Mini OCR or Qwen-VL endpoint.
- OCR factory selects first-class LiteLLM provider using aliases `ocr` and `vision`.
- Application performs no secondary OCR routing outside LiteLLM for this configuration.
- Focused tests and `make gate-fast` pass.
- Live end-to-end requests through the configured LiteLLM endpoint pass for both `ocr` and
  `vision` aliases.
- Rebuilt `doc-organizer` reports healthy after restart.
