"""Embedding inputs must be bounded to the embed model's context window.

An embeddings request is all-or-nothing: one input longer than the model's
context makes the provider reject the whole batch with a 400, and because the
input never gets shorter that failure is permanent — the doc fails the embed
step on every run, forever (#0569: two email attachments whose conversation
context block tokenizes to ~54k tokens against a 40960-token model).

Chunked text is bounded by the chunker, but several call sites legitimately
embed a single un-chunked body (a conversation context block, a taxonomy
label), so the guard lives at the provider boundary — the only place that
knows which model, and therefore which limit, applies.
"""

from unittest.mock import patch

from providers.embed.limits import (
    DEFAULT_MAX_INPUT_TOKENS,
    bound_inputs,
    resolve_max_input_tokens,
    token_count,
)
from providers.embed.openrouter_embed import OpenRouterEmbedProvider


# --- limit resolution ---

def test_known_models_resolve_their_context_window():
    assert resolve_max_input_tokens("qwen/qwen3-embedding-8b") == 40960
    assert resolve_max_input_tokens("qwen3-embedding:4b-q8_0") == 40960
    assert resolve_max_input_tokens("text-embedding-3-large") == 8191


def test_unknown_model_falls_back_to_conservative_default():
    assert resolve_max_input_tokens("some/brand-new-embedder") == DEFAULT_MAX_INPUT_TOKENS
    assert resolve_max_input_tokens(None) == DEFAULT_MAX_INPUT_TOKENS


# --- bounding ---

def test_inputs_within_the_limit_are_untouched():
    texts = ["short", "also short", ""]
    assert bound_inputs(texts, 4096) == texts


def test_oversized_input_is_truncated_to_fit_the_limit():
    oversized = "context line about the invoice\n" * 4000
    assert token_count(oversized) > 4096

    (bounded,) = bound_inputs([oversized], 4096)

    assert token_count(bounded) <= 4096
    assert bounded and oversized.startswith(bounded)


def test_bounding_is_per_input_not_per_batch():
    small = "fine"
    oversized = "wall of text " * 5000
    bounded = bound_inputs([small, oversized], 2048)

    assert bounded[0] == small
    assert token_count(bounded[1]) <= 2048


# --- provider integration (the poison-pill path) ---

class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.headers = {}
        self.text = str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def test_provider_bounds_an_oversized_input_before_calling_the_api():
    """The #0569 repro: a single >context-window input must never be sent as-is."""
    provider = OpenRouterEmbedProvider(
        model="qwen/qwen3-embedding-8b", api_key="test-key",
    )
    # ~54k tokens — the measured size of the msg693 conversation context block.
    oversized = "BEFORE 2026-04-10 laura sanchez: please see the attached invoice\n" * 4000
    assert token_count(oversized) > 40960

    sent = {}

    def _capture(url, json=None, headers=None, timeout=None):
        sent["input"] = json["input"]
        return _FakeResponse({"data": [{"index": 0, "embedding": [0.1] * 8}]})

    with patch("providers.embed.openrouter_embed.httpx.post", side_effect=_capture):
        vectors = provider.embed_texts([oversized])

    assert len(vectors) == 1
    assert token_count(sent["input"][0]) <= 40960


def test_provider_limit_is_configurable_per_model():
    provider = OpenRouterEmbedProvider(
        model="qwen/qwen3-embedding-8b", api_key="test-key", max_input_tokens=512,
    )
    assert provider.max_input_tokens == 512

    sent = {}

    def _capture(url, json=None, headers=None, timeout=None):
        sent["input"] = json["input"]
        return _FakeResponse({"data": [{"index": 0, "embedding": [0.1] * 8}]})

    with patch("providers.embed.openrouter_embed.httpx.post", side_effect=_capture):
        provider.embed_query("what did laura send? " * 500)

    assert token_count(sent["input"][0]) <= 512
