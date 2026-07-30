"""Keep embedding inputs within model context windows."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Callable

logger = logging.getLogger(__name__)

# Match substrings so provider prefixes and model revisions share one limit.
MODEL_INPUT_TOKENS = {
    "qwen3-embedding": 40_960,
    "text-embedding-3": 8_191,
    "text-embedding-ada-002": 8_191,
    "nomic-embed-text": 8_192,
    "bge-m3": 8_192,
    "gemini-embedding": 2_048,
    "text-embedding-004": 2_048,
}
DEFAULT_MAX_INPUT_TOKENS = 8_192
TOKENIZER_HEADROOM = 0.9


@lru_cache(maxsize=1)
def _tokenizer() -> Callable[[str], list]:
    from llama_index.core.utils import get_tokenizer

    return get_tokenizer()


def resolve_max_input_tokens(model: str | None) -> int:
    """Return advertised input window for model, or conservative default."""
    model_name = (model or "").lower()
    for fragment, limit in MODEL_INPUT_TOKENS.items():
        if fragment in model_name:
            return limit
    return DEFAULT_MAX_INPUT_TOKENS


def bound_inputs(
    texts: list[str],
    max_input_tokens: int,
    *,
    label: str = "embed",
) -> list[str]:
    """Truncate each oversized input while preserving one-vector-per-input."""
    limit = max(1, int(max_input_tokens * TOKENIZER_HEADROOM))
    bounded: list[str] = []

    for text in texts:
        # Byte-level BPE cannot emit more tokens than UTF-8 input bytes. This
        # preserves the cheap path without treating Unicode characters as
        # tokens (one visible character can encode to several tokens).
        if len(text.encode("utf-8")) <= limit:
            bounded.append(text)
            continue
        measured = len(_tokenizer()(text))
        if measured <= limit:
            bounded.append(text)
            continue

        fitted = _truncate_to_tokens(text, limit, measured)
        logger.warning(
            "%s: %d-token embedding input exceeds %d-token context; "
            "truncated to at most %d tokens",
            label,
            measured,
            max_input_tokens,
            limit,
        )
        bounded.append(fitted)

    return bounded


def _truncate_to_tokens(text: str, limit: int, measured: int) -> str:
    """Return text prefix fitting token limit without repeated full scans."""
    end = max(1, int(len(text) * limit / measured))
    fitted = text[:end]
    fitted_tokens = len(_tokenizer()(fitted))
    while fitted_tokens > limit:
        end = max(1, int(end * limit / fitted_tokens))
        fitted = text[:end]
        fitted_tokens = len(_tokenizer()(fitted))
    return fitted
