import httpx
import pytest

from core.resilience import (
    TransientError,
    call_with_retry,
    is_transient,
)


def _http_status(code):
    req = httpx.Request("POST", "http://x")
    resp = httpx.Response(code, request=req)
    return httpx.HTTPStatusError(f"{code}", request=req, response=resp)


def test_is_transient_classification():
    assert is_transient(httpx.ConnectError("x"))
    assert is_transient(httpx.ReadTimeout("x"))
    assert is_transient(TimeoutError("x"))
    assert is_transient(TransientError("forced"))
    assert is_transient(_http_status(504))
    assert is_transient(_http_status(429))
    assert is_transient(_http_status(503))
    # permanent
    assert not is_transient(_http_status(400))
    assert not is_transient(_http_status(404))
    assert not is_transient(ValueError("bad input"))


def test_retries_transient_then_succeeds():
    calls = {"n": 0}
    slept = []

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("blip")
        return "ok"

    out = call_with_retry(fn, attempts=3, backoff=(0.1, 0.2),
                          label="t", sleep=slept.append)
    assert out == "ok"
    assert calls["n"] == 3
    assert slept == [0.1, 0.2]          # backed off before retries 2 and 3


def test_permanent_failure_not_retried():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise _http_status(400)

    with pytest.raises(httpx.HTTPStatusError):
        call_with_retry(fn, attempts=4, sleep=lambda *_: None)
    assert calls["n"] == 1               # no retry on a permanent 4xx


def test_exhaustion_reraises_original():
    def fn():
        raise httpx.ReadTimeout("still down")

    with pytest.raises(httpx.ReadTimeout):
        call_with_retry(fn, attempts=2, backoff=(0,), sleep=lambda *_: None)


def test_transient_error_forces_retry():
    """A failure surfaced in an HTTP-200 body (raised as TransientError) is retried."""
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] == 1:
            raise TransientError("upstream 429 in 200 body")
        return 42

    assert call_with_retry(fn, attempts=3, backoff=(0,), sleep=lambda *_: None) == 42
    assert calls["n"] == 2
