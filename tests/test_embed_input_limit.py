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

import re
from unittest.mock import patch

import httpx

from providers.embed.limits import (
    DEFAULT_MAX_INPUT_TOKENS,
    EMPTY_INPUT_PLACEHOLDER,
    bound_inputs,
    resolve_max_input_tokens,
    token_count,
)
from providers.embed.openrouter_embed import (
    DEFAULT_MAX_INPUT_CHARS,
    OpenRouterEmbedProvider,
    _fetch_route_input_tokens,
)


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


# --- the route's character cap (#1655) ---

def _english_context_block(messages: int) -> str:
    """A conversation context block of ordinary English prose.

    Prose tokenizes at ~4.9 characters per token, so an input bounded to the
    model's 40960-token window is still ~180k characters — the shape that made
    #1655 fail permanently.
    """
    body = (
        "Thank you for reaching out about the property. We have received your "
        "message and will respond with the requested documents as soon as the "
        "office reopens on Monday morning. Please let us know if anything else "
        "is needed in the meantime. "
    )
    return "\n".join(f"BEFORE message {i}: {body}" for i in range(messages))


def test_char_cap_is_enforced_alongside_the_token_bound():
    oversized = _english_context_block(900)
    assert token_count(oversized) > 40960

    (bounded,) = bound_inputs([oversized], 40960, max_input_chars=131072)

    assert len(bounded) <= 131072
    assert token_count(bounded) <= 40960
    assert bounded and oversized.startswith(bounded)


def test_char_cap_alone_bounds_an_input_inside_the_token_window():
    """Characters and tokens are not interchangeable: an input can sit inside
    the token window and still overrun the route's character cap."""
    inside_window = "The tenant submitted a maintenance request. " * 5000
    assert token_count(inside_window) < 40960
    assert len(inside_window) > 131072

    (bounded,) = bound_inputs([inside_window], 40960, max_input_chars=131072)

    assert len(bounded) == 131072


def test_no_char_cap_leaves_the_token_bound_alone():
    inside_window = "The tenant submitted a maintenance request. " * 5000

    (bounded,) = bound_inputs([inside_window], 40960)

    assert bounded == inside_window


def test_provider_bounds_an_input_that_only_the_char_cap_rejects():
    """The #1655 repro: DeepInfra (the qwen3-embedding upstream behind
    OpenRouter) rejects an input over 131072 characters with a permanent 422,
    even though it is well inside the model's 40960-token window."""
    provider = OpenRouterEmbedProvider(
        model="qwen/qwen3-embedding-8b", api_key="test-key",
    )
    oversized = _english_context_block(900)
    assert token_count(oversized) > 40960

    sent = {}

    def _capture(url, json=None, headers=None, timeout=None):
        sent["input"] = json["input"]
        return _FakeResponse({"data": [{"index": 0, "embedding": [0.1] * 8}]})

    with patch("providers.embed.openrouter_embed.httpx.post", side_effect=_capture):
        vectors = provider.embed_texts([oversized])

    assert len(vectors) == 1
    assert len(sent["input"][0]) <= DEFAULT_MAX_INPUT_CHARS
    assert token_count(sent["input"][0]) <= 40960


def test_provider_char_cap_is_configurable():
    provider = OpenRouterEmbedProvider(
        model="qwen/qwen3-embedding-8b", api_key="test-key", max_input_chars=4096,
    )
    assert provider.max_input_chars == 4096

    sent = {}

    def _capture(url, json=None, headers=None, timeout=None):
        sent["input"] = json["input"]
        return _FakeResponse({"data": [{"index": 0, "embedding": [0.1] * 8}]})

    with patch("providers.embed.openrouter_embed.httpx.post", side_effect=_capture):
        provider.embed_texts(["a much longer body " * 2000])

    assert len(sent["input"][0]) == 4096


# --- counting tokens the way the model counts them (#1344) ---

def _qwen_token_floor(text: str) -> int:
    """A floor on Qwen's token count that does not use the bound's tokenizer.

    Qwen's published pre-tokenizer (tokenizer.json) makes every digit its own
    piece and never merges whitespace into a digit, and every piece is at least
    one token. cl100k_base groups up to three digits per token, so digit-dense
    text costs Qwen far more than the pipeline tokenizer measures.
    """
    return len(re.findall(r"\d|\s+", text))


def _structured_snapshot(records: int) -> str:
    """A machine-generated status snapshot: ids, timestamps and counters.

    The shape of the #1344 doc — a hypercare snapshot whose context block the
    bound cut to 36864 cl100k tokens, which qwen3-embedding counted past its
    40960-token window, so OpenRouter answered 400 on every run.
    """
    return "".join(
        f"{i:08d} 2026-10-04 17:05:{i % 60:02d} {i * 7919 % 100000:05d} {i % 997:03d}\n"
        for i in range(records)
    )


def test_bound_counts_digits_the_way_a_qwen_embedding_model_does():
    snapshot = _structured_snapshot(2000)
    assert len(snapshot) < 131072, "the token window, not the char cap, must bind"
    assert _qwen_token_floor(snapshot) > 40960

    (bounded,) = bound_inputs(
        [snapshot], 40960, max_input_chars=131072, model="qwen/qwen3-embedding-8b",
    )

    assert _qwen_token_floor(bounded) <= 40960
    assert bounded and snapshot.startswith(bounded)


def test_provider_never_sends_an_input_its_model_counts_past_the_window():
    """The #1344 repro at the provider boundary."""
    provider = OpenRouterEmbedProvider(
        model="qwen/qwen3-embedding-8b", api_key="test-key",
    )
    snapshot = _structured_snapshot(2000)
    sent = {}

    def _capture(url, json=None, headers=None, timeout=None):
        sent["input"] = json["input"]
        return _FakeResponse({"data": [{"index": 0, "embedding": [0.1] * 8}]})

    with patch("providers.embed.openrouter_embed.httpx.post", side_effect=_capture):
        vectors = provider.embed_texts([snapshot])

    assert len(vectors) == 1
    assert _qwen_token_floor(sent["input"][0]) <= 40960


def test_unmapped_model_keeps_the_pipeline_tokenizer():
    """A model whose tokenizer the module does not know is still bounded by the
    pipeline tokenizer and its headroom."""
    oversized = "context line about the invoice\n" * 4000

    assert bound_inputs([oversized], 4096, model="some/brand-new-embedder") == (
        bound_inputs([oversized], 4096)
    )


# --- the route's per-upstream token window (#4415) ---

class _EndpointsResponse(_FakeResponse):
    """OpenRouter's GET /models/{model}/endpoints listing."""

    def __init__(self, windows):
        super().__init__({"data": {"endpoints": [
            {"provider_name": f"upstream-{i}", "context_length": window}
            for i, window in enumerate(windows)
        ]}})


def _no_route_listing(url, timeout=None):
    raise httpx.ConnectError("no route listing")


def _provider_listing(windows):
    _fetch_route_input_tokens.cache_clear()
    with patch(
        "providers.embed.openrouter_embed.httpx.get",
        return_value=_EndpointsResponse(windows),
    ):
        return OpenRouterEmbedProvider(
            model="qwen/qwen3-embedding-8b", api_key="test-key",
        )


def test_provider_never_sends_an_input_past_the_smallest_upstream_window():
    """The #4415 repro: a digit-dense snapshot bounded to qwen3-embedding's own
    40960-token window (36864 with headroom) still 400'd on every run, because
    OpenRouter routed it to SiliconFlow, which serves the model with 32768."""
    provider = _provider_listing([40960, 32768, 32768])
    # Shaped like the production doc: past the route's char cap, and still past
    # the model's own window once cut to it.
    snapshot = _structured_snapshot(4500)
    assert len(snapshot) > DEFAULT_MAX_INPUT_CHARS
    assert token_count(snapshot[:DEFAULT_MAX_INPUT_CHARS], "qwen/qwen3-embedding-8b") > 40960
    sent = {}

    def _capture(url, json=None, headers=None, timeout=None):
        sent["input"] = json["input"]
        return _FakeResponse({"data": [{"index": 0, "embedding": [0.1] * 8}]})

    with patch("providers.embed.openrouter_embed.httpx.post", side_effect=_capture):
        vectors = provider.embed_texts([snapshot])

    assert len(vectors) == 1
    # The upstream appends an end-of-sequence token to what the model counts.
    assert token_count(sent["input"][0], "qwen/qwen3-embedding-8b") + 1 <= 32768
    assert snapshot.startswith(sent["input"][0])


def test_provider_falls_back_to_the_model_window_without_a_route_listing():
    """A base_url that lists no upstreams (a proxy, the staging simulator) or a
    listing that cannot be read keeps the model's own window, as before #4415 —
    and a failed lookup is retried by the next provider rather than cached."""
    _fetch_route_input_tokens.cache_clear()
    with patch(
        "providers.embed.openrouter_embed.httpx.get", side_effect=_no_route_listing,
    ):
        provider = OpenRouterEmbedProvider(
            model="qwen/qwen3-embedding-8b", api_key="test-key",
        )
    assert provider.max_input_tokens == 40960

    assert _provider_listing([40960, 32768]).max_input_tokens == 32768
