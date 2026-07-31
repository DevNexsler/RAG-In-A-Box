"""Bound embedding inputs to the embed model's context window.

An embeddings request is all-or-nothing: a single input longer than the model's
context makes the provider reject the WHOLE batch with a 400, and because the
input never gets shorter, that failure is deterministic — the doc fails the
embed step on every run, forever, and is never written to the index (#0569).
Retry cannot help and quarantine only stops the bleeding; the fix is to never
send an input the model cannot accept.

Chunked text is already bounded by the chunker, but several call sites
legitimately embed a single un-chunked body (a conversation context block, a
taxonomy label, a context-only alias node). The guard therefore lives at the
provider boundary — the one place that knows which model, and so which limit,
applies — rather than being re-derived at each call site.

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
    """Return `texts` with every input truncated to fit `max_input_tokens`.

    One vector per input is part of the EmbedProvider contract, so an oversized
    input is truncated rather than split. Nothing indexed is lost by this in
    practice: the long un-chunked bodies are whole-document or whole-context
    summaries whose text is also indexed through the normal chunked path.
    """
    if max_input_tokens <= 0:
        return list(texts)
    limit = max(1, int(max_input_tokens * TOKENIZER_HEADROOM))

    bounded: list[str] = []
    for text in texts:
        # Every token is at least one character, so a text no longer than the
        # limit in characters cannot exceed it in tokens — skip the tokenizer.
        if len(text) <= limit:
            bounded.append(text)
            continue
        measured = token_count(text)
        if measured <= limit:
            bounded.append(text)
            continue
        fitted = _truncate_to_tokens(text, limit)
        logger.warning(
            "%s: input of %d tokens exceeds the model's %d-token context — "
            "truncated to %d tokens (%d of %d chars)",
            label, measured, max_input_tokens, limit, len(fitted), len(text),
        )
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
