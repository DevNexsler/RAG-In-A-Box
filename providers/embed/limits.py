"""Bound embedding inputs to what the embed route will accept.

An embeddings request is all-or-nothing: a single unacceptable input makes the
provider reject the WHOLE batch, and because the input never changes, that
failure is deterministic — the doc fails the embed step on every run, forever,
and is never written to the index. Retry cannot help and quarantine only stops
the bleeding; the fix is to never send an input the route cannot accept.

An input is unacceptable at either end. Too long is #0569: past the model's
context window the route rejects the batch. Too short is #1687: an input with
no text is rejected just as hard and just as permanently — OpenRouter's gateway
refuses an empty string outright, and the upstreams behind it refuse an input
that normalizes to nothing. One vector per input is part of the EmbedProvider
contract, so neither end may drop a slot; both reshape it in place.

Chunked text is already bounded by the chunker, but several call sites
legitimately embed a single un-chunked body (a conversation context block, a
taxonomy label, a context-only alias node). The guard therefore lives at the
provider boundary — the one place that knows which route, and so which limits,
apply — rather than being re-derived at each call site.

Lengths are measured with the same tokenizer LlamaIndex's SentenceSplitter
uses for `chunk_size`, so "tokens" means one thing across the pipeline. It is
an approximation of any given hosted model's own tokenizer, hence the headroom
below.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Callable

logger = logging.getLogger(__name__)

# Input context window (tokens) per embedding model, matched by substring on
# the model id so provider prefixes and revisions ("qwen/qwen3-embedding-8b",
# "qwen3-embedding:4b-q8_0") share one entry.
MODEL_INPUT_TOKENS: dict[str, int] = {
    "qwen3-embedding": 40960,
    "text-embedding-3": 8191,
    "text-embedding-ada-002": 8191,
    "nomic-embed-text": 8192,
    "bge-m3": 8192,
    "embeddinggemma": 2048,
    "gemini-embedding": 2048,
    "text-embedding-004": 2048,
}

# Unknown model: the window shared by most current embedding models. Comfortably
# above the chunker's default 1800-token chunk, so ordinary chunks are never
# touched, while an un-chunked body still gets bounded instead of 400-ing.
DEFAULT_MAX_INPUT_TOKENS = 8192

# Our tokenizer is a proxy for the model's own, so spend only this share of the
# advertised window; a model that tokenizes ~10% denser than cl100k still fits.
TOKENIZER_HEADROOM = 0.9

# Stand-in for an input carrying no text. Measured against
# openrouter.ai/api/v1/embeddings (qwen/qwen3-embedding-8b) on 2026-08-27: "" is
# rejected by the gateway itself (zod `too_small`, minimum 1) and whitespace-only
# inputs are rejected by the upstream behind it ("Prompt must not be empty"), the
# latter intermittently — `[" "]` answered 400/400/200 over three runs, depending
# on which provider the gateway routed to. A single printable character is the
# smallest input no layer treats as absent, and it keeps the slot addressable so
# vectors stay aligned with inputs.
EMPTY_INPUT_PLACEHOLDER = "."


@lru_cache(maxsize=1)
def _tokenizer() -> Callable[[str], list]:
    from llama_index.core.utils import get_tokenizer

    return get_tokenizer()


def token_count(text: str) -> int:
    """Token length of `text` under the pipeline's tokenizer."""
    return len(_tokenizer()(text))


def resolve_max_input_tokens(model: str | None) -> int:
    """Context window of `model`, or a conservative default for unknown ones."""
    name = (model or "").lower()
    for key, limit in MODEL_INPUT_TOKENS.items():
        if key in name:
            return limit
    return DEFAULT_MAX_INPUT_TOKENS


def bound_inputs(
    texts: list[str], max_input_tokens: int, *, label: str = "embed",
) -> list[str]:
    """Return `texts` with every input reshaped to something the route accepts.

    One vector per input is part of the EmbedProvider contract, so an oversized
    input is truncated rather than split, and an input with no text is
    substituted rather than dropped — dropping the slot would silently misalign
    every later vector in the batch. Nothing indexed is lost by the truncation
    in practice: the long un-chunked bodies are whole-document or whole-context
    summaries whose text is also indexed through the normal chunked path.

    `max_input_tokens <= 0` disables the context-window bound only. The route's
    refusal of a text-free input is not a tunable, so it always applies.
    """
    limit = max(1, int(max_input_tokens * TOKENIZER_HEADROOM)) if max_input_tokens > 0 else 0

    bounded: list[str] = []
    for text in texts:
        fitted = text
        # Every token is at least one character, so a text no longer than the
        # limit in characters cannot exceed it in tokens — skip the tokenizer.
        if limit and len(text) > limit:
            measured = token_count(text)
            if measured > limit:
                fitted = _truncate_to_tokens(text, limit)
                logger.warning(
                    "%s: input of %d tokens exceeds the model's %d-token context — "
                    "truncated to %d tokens (%d of %d chars)",
                    label, measured, max_input_tokens, limit, len(fitted), len(text),
                )
        # Checked after truncating, not before: a text whose only content sits
        # past the limit is bounded down to whitespace, which the route refuses
        # for the same reason it refuses "".
        if not fitted.strip():
            logger.warning(
                "%s: input carries no text — substituting %r so the route does "
                "not reject the whole batch",
                label, EMPTY_INPUT_PLACEHOLDER,
            )
            fitted = EMPTY_INPUT_PLACEHOLDER
        bounded.append(fitted)
    return bounded


def _truncate_to_tokens(text: str, limit: int) -> str:
    """Longest prefix of `text` that fits in `limit` tokens.

    Binary search over characters keeps this tokenizer-agnostic: it needs only
    a length function, not a decoder, and costs ~log2(len) tokenizations of an
    input that is oversized to begin with.
    """
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(_tokenizer()(text[:mid])) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]
