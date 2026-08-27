"""Embedding inputs must be bounded to what the embed route accepts.

An embeddings request is all-or-nothing: one unacceptable input makes the
provider reject the whole batch, and because the input never changes that
failure is permanent — the doc fails the embed step on every run, forever.

Inputs fail at both ends. Too long is #0569 (two email attachments whose
conversation context block tokenizes to ~54k tokens against a 40960-token
model). Too short is #1687: a text-free input is rejected as hard and as
permanently, by the gateway when it is empty and by the upstream when it is
whitespace-only.

Chunked text is bounded by the chunker, but several call sites legitimately
embed a single un-chunked body (a conversation context block, a taxonomy
label), so the guard lives at the provider boundary — the only place that
knows which route, and therefore which limits, apply.
"""

from unittest.mock import patch

from providers.embed.limits import (
    DEFAULT_MAX_INPUT_TOKENS,
    EMPTY_INPUT_PLACEHOLDER,
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
    texts = ["short", "also short", "third"]
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


def test_text_free_input_is_substituted_not_dropped():
    """#1687: the slot has to survive, or every later vector shifts by one."""
    bounded = bound_inputs(["first", "", "third"], 4096)

    assert bounded == ["first", EMPTY_INPUT_PLACEHOLDER, "third"]
    assert EMPTY_INPUT_PLACEHOLDER.strip(), "a placeholder the route also refuses is no fix"


def test_whitespace_only_input_is_substituted_too():
    """The gateway refuses "" and the upstream refuses input that normalizes to
    nothing, so the guard keys on content, not on length."""
    assert bound_inputs(["   ", "\n\n", "\t"], 4096) == [EMPTY_INPUT_PLACEHOLDER] * 3


def test_substitution_still_applies_when_the_token_bound_is_disabled():
    """`max_input_tokens: 0` turns off the context window, not the route's
    refusal of a text-free input — they are independent limits."""
    assert bound_inputs(["first", "", "third"], 0) == [
        "first", EMPTY_INPUT_PLACEHOLDER, "third",
    ]


def test_input_truncated_down_to_whitespace_is_substituted():
    """Bounding can itself produce a text-free input: everything that survives
    the cut is whitespace."""
    (bounded,) = bound_inputs([" " * 5000 + "the only word"], 1)

    assert bounded == EMPTY_INPUT_PLACEHOLDER


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


def test_provider_never_sends_a_text_free_input():
    """#1687 at the provider boundary: the batch the API sees has no empty slot,
    and the caller still gets one vector per input, in order."""
    provider = OpenRouterEmbedProvider(
        model="qwen/qwen3-embedding-8b", api_key="test-key",
    )
    sent = {}

    def _capture(url, json=None, headers=None, timeout=None):
        sent["input"] = json["input"]
        return _FakeResponse(
            {"data": [
                {"index": i, "embedding": [float(i)] * 8}
                for i in range(len(json["input"]))
            ]}
        )

    with patch("providers.embed.openrouter_embed.httpx.post", side_effect=_capture):
        vectors = provider.embed_texts(["first document", "", "third one"])

    assert sent["input"] == ["first document", EMPTY_INPUT_PLACEHOLDER, "third one"]
    assert [v[0] for v in vectors] == [0.0, 1.0, 2.0]


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
