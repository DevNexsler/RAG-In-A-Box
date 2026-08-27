"""Tests for the OpenRouter embedding provider's error handling.

OpenRouter can wrap upstream provider failures (e.g. Nebius 429 quota) in an
HTTP 200 response whose body is {"error": {...}} with no "data" key. The
provider must retry those like transport-level 429s instead of crashing with
an opaque KeyError.
"""

from functools import partial
from unittest.mock import patch

import httpx
import pytest

from core.resilience import call_with_retry
from providers.embed.openrouter_embed import MAX_RETRIES, OpenRouterEmbedProvider


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.headers = {}
        self.text = str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass  # status 200 — never raises


def _provider():
    return OpenRouterEmbedProvider(model="qwen/qwen3-embedding-8b", api_key="test-key")


def _ok_payload(n=1):
    return {"data": [{"index": i, "embedding": [0.1] * 8} for i in range(n)]}


def test_quota_429_in_200_body_retries_then_succeeds():
    """A 200 response carrying an upstream 429 error body should be retried."""
    quota_err = {"error": {"message": "HTTP 429: tokens quota exceeded", "code": 429}}
    responses = [_FakeResponse(quota_err), _FakeResponse(_ok_payload())]

    with patch("providers.embed.openrouter_embed.httpx.post", side_effect=responses):
        with patch(
            "providers.embed.openrouter_embed.call_with_retry",
            new=partial(call_with_retry, sleep=lambda _: None),
        ):
            vectors = _provider()._call_embeddings(["hello"])

    assert len(vectors) == 1
    assert len(vectors[0]) == 8


def test_quota_429_in_200_body_uses_rate_limit_backoff():
    quota_err = {"error": {"message": "HTTP 429: tokens quota exceeded", "code": 429}}
    responses = [_FakeResponse(quota_err), _FakeResponse(_ok_payload())]
    slept = []

    with patch("providers.embed.openrouter_embed.httpx.post", side_effect=responses):
        with patch("core.resilience.random.uniform", return_value=0.5):
            with patch(
                "providers.embed.openrouter_embed.call_with_retry",
                new=partial(call_with_retry, sleep=slept.append),
            ):
                _provider()._call_embeddings(["hello"])

    assert slept == [2.5]


def test_http_429_honors_retry_after_header():
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/embeddings")
    responses = [
        httpx.Response(429, headers={"Retry-After": "37"}, request=request),
        httpx.Response(200, json=_ok_payload(), request=request),
    ]
    slept = []

    with patch("providers.embed.openrouter_embed.httpx.post", side_effect=responses):
        with patch(
            "providers.embed.openrouter_embed.call_with_retry",
            new=partial(call_with_retry, sleep=slept.append),
        ):
            _provider()._call_embeddings(["hello"])

    assert slept == [37.0]


def test_quota_429_exhausting_retries_raises_clear_error():
    """Persistent upstream quota errors should surface a readable error, not KeyError."""
    quota_err = {"error": {"message": "HTTP 429: tokens quota exceeded", "code": 429}}

    with patch(
        "providers.embed.openrouter_embed.httpx.post",
        return_value=_FakeResponse(quota_err),
    ):
        with patch(
            "providers.embed.openrouter_embed.call_with_retry",
            new=partial(call_with_retry, sleep=lambda _: None),
        ):
            with pytest.raises(RuntimeError, match="429"):
                _provider()._call_embeddings(["hello"])


def test_non_retryable_error_body_raises_immediately():
    """Non-retryable embedded errors (e.g. 400) should raise without retry."""
    bad_req = {"error": {"message": "invalid model", "code": 400}}
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(1)
        return _FakeResponse(bad_req)

    with patch("providers.embed.openrouter_embed.httpx.post", side_effect=fake_post):
        with pytest.raises(RuntimeError, match="invalid model"):
            _provider()._call_embeddings(["hello"])

    assert len(calls) == 1


# The body OpenRouter actually returned for the #1655 oversized input. The status
# line alone ("422 Unprocessable Entity") names none of it.
OVERSIZED_INPUT_BODY = (
    '{"error":{"message":"HTTP 422: {\\"error\\":{\\"message\\":\\"Value error, '
    "The input sequence should have less than 131072 characters. "
    'Input length: 180439\\"}}"}}'
)


def test_permanent_http_4xx_surfaces_upstream_reason():
    """#1657: a permanent 4xx skips the doc forever, so it must log why.

    httpx builds HTTPStatusError's message from the status line alone, so before
    this the only thing reaching indexer.log was "Client error '422 Unprocessable
    Entity' for url '...'" — diagnosing it meant re-probing the live endpoint.
    """
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/embeddings")
    response = httpx.Response(422, text=OVERSIZED_INPUT_BODY, request=request)
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(1)
        return response

    with patch("providers.embed.openrouter_embed.httpx.post", side_effect=fake_post):
        with pytest.raises(httpx.HTTPStatusError) as caught:
            _provider()._call_embeddings(["hello"])

    assert "less than 131072 characters" in str(caught.value)
    assert len(calls) == 1  # permanent: raised straight through, never retried


@pytest.mark.parametrize("status_code", [408, 425, 500, 503, 504])
def test_transient_http_status_is_still_retried(status_code):
    """The seam: reading the body on a permanent 4xx must not cost the retry
    ladder on a temporary one — classification stays the shared layer's call."""
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/embeddings")
    response = httpx.Response(status_code, text="upstream busy", request=request)
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(1)
        return response

    with patch("providers.embed.openrouter_embed.httpx.post", side_effect=fake_post):
        with patch(
            "providers.embed.openrouter_embed.call_with_retry",
            new=partial(call_with_retry, sleep=lambda _: None),
        ):
            with pytest.raises(httpx.HTTPStatusError):
                _provider()._call_embeddings(["hello"])

    assert len(calls) == MAX_RETRIES
