import io
import logging

import httpx
import pytest

from core.logging_setup import DEFAULT_FORMAT, SingleLineFormatter
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


def _capture_retry_warnings(exc, attempts=3):
    """Run a doomed call and return the raw physical lines the retry warning wrote."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(SingleLineFormatter(DEFAULT_FORMAT))
    logger = logging.getLogger("core.resilience")
    saved, saved_prop, saved_level = logger.handlers, logger.propagate, logger.level
    logger.handlers, logger.propagate = [handler], False
    logger.setLevel(logging.WARNING)
    try:
        with pytest.raises(type(exc)):
            call_with_retry(lambda: (_ for _ in ()).throw(exc), attempts=attempts,
                            backoff=(0,), label="litellm ocr_extract",
                            sleep=lambda *_: None)
    finally:
        logger.handlers, logger.propagate = saved, saved_prop
        logger.setLevel(saved_level)
    return stream.getvalue().splitlines()


_RATE_LIMIT = TransientError(
    "Error code: 429 - {'error': {'message': 'rate limit exceeded'}}\n"
    "For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/429"
)


def test_retry_warning_message_is_single_line_at_the_source(caplog):
    """#0546: the provider error's str() carries a '\\n', so the retry warning wrote
    a second untimestamped, unattributable physical line per retry — 27% of a
    nightly log capture. Sanitize where the exception is interpolated, not only in
    the formatter, so the record is one line for every handler."""
    with caplog.at_level(logging.WARNING, logger="core.resilience"):
        with pytest.raises(TransientError):
            call_with_retry(lambda: (_ for _ in ()).throw(_RATE_LIMIT), attempts=3,
                            backoff=(0,), label="litellm ocr_extract",
                            sleep=lambda *_: None)

    records = [r for r in caplog.records if r.name == "core.resilience"]
    assert len(records) == 2                          # one per retried attempt
    for record in records:
        message = record.getMessage()
        assert "\n" not in message
        # error text intact-but-collapsed; the retry suffix is no longer the thing
        # that gets cut (the old code truncated the formatted message mid-URL)
        assert "For more information check" in message
        assert message.endswith("retrying in 0s")


def test_retry_warning_renders_as_exactly_one_physical_line(caplog):
    lines = _capture_retry_warnings(_RATE_LIMIT, attempts=3)
    assert len(lines) == 2
    for line in lines:
        assert line.startswith("20")                  # every line is timestamped
        assert "WARNING core.resilience" in line
        assert line.endswith("retrying in 0s")


def test_retry_warning_truncates_a_huge_error_body_after_collapsing(caplog):
    exc = TransientError("Error code: 429 - " + "x" * 5000 + "\ntail")
    (line,) = _capture_retry_warnings(exc, attempts=2)
    assert "..." in line and line.endswith("retrying in 0s")
    assert len(line) < 1000


def test_repeated_retries_carry_a_dedup_key(caplog):
    """The flood is collapsible only if the record says what "the same failure"
    means: (label, error class) — not the attempt number or the backoff delay."""
    with caplog.at_level(logging.WARNING, logger="core.resilience"):
        with pytest.raises(TransientError):
            call_with_retry(lambda: (_ for _ in ()).throw(_RATE_LIMIT), attempts=3,
                            backoff=(0,), label="litellm ocr_extract",
                            sleep=lambda *_: None)

    keys = {getattr(r, "dedup_key", None) for r in caplog.records
            if r.name == "core.resilience"}
    assert keys == {("core.resilience.retry", "litellm ocr_extract", "TransientError")}


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
