"""Keep embedding inputs within model context windows."""

from __future__ import annotations

import logging

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
    """Conservatively cap each input while preserving one-vector-per-input.

    Every supported tokenizer token consumes at least one UTF-8 source byte, so
    a byte ceiling below the model token window is safe without guessing which
    tokenizer an OpenRouter route ultimately selects.
    """
    limit = max(1, int(max_input_tokens * TOKENIZER_HEADROOM))
    bounded: list[str] = []

    for text in texts:
        encoded = text.encode("utf-8")
        if len(encoded) <= limit:
            bounded.append(text)
            continue

        fitted = encoded[:limit].decode("utf-8", errors="ignore")
        logger.warning(
            "%s: %d-byte embedding input may exceed %d-token context; "
            "truncated to at most %d UTF-8 bytes",
            label,
            len(encoded),
            max_input_tokens,
            limit,
        )
        bounded.append(fitted)

    return bounded
