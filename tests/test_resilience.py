import io
import logging
import random

import httpx
import pytest

from core.logging_setup import DEFAULT_FORMAT, SingleLineFormatter
from core.resilience import (
    TRANSIENT_STATUSES,
    TransientError,
    call_with_retry,
    is_transient,
    raise_for_status,
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


def test_rate_limit_honors_retry_after_seconds():
    req = httpx.Request("POST", "https://openrouter.ai/api/v1/embeddings")
    resp = httpx.Response(429, headers={"Retry-After": "37"}, request=req)
    rate_limit = httpx.HTTPStatusError(
        "429 Too Many Requests", request=req, response=resp,
    )
    calls = {"n": 0}
    slept = []

    def fn():
        calls["n"] += 1
        if calls["n"] == 1:
            raise rate_limit
        return "ok"

    assert call_with_retry(
        fn, attempts=2, backoff=(2.0,), sleep=slept.append,
    ) == "ok"
    assert slept == [37.0]


def test_rate_limit_without_retry_after_uses_exponential_backoff_with_jitter(
    monkeypatch,
):
    rate_limit = _http_status(429)
    calls = {"n": 0}
    slept = []
    jitter_bounds = []

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise rate_limit
        return "ok"

    def fixed_jitter(lower, upper):
        jitter_bounds.append((lower, upper))
        return upper

    monkeypatch.setattr(random, "uniform", fixed_jitter)

    assert call_with_retry(
        fn, attempts=3, backoff=(2.0, 5.0), sleep=slept.append,
    ) == "ok"
    assert jitter_bounds == [(0.0, 0.5), (0.0, 1.0)]
    assert slept == [2.5, 5.0]


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


# --- raise_for_status: keep the reason on a permanent rejection (#1657) ------


def _response(code, text="", url="https://openrouter.ai/api/v1/embeddings"):
    return httpx.Response(code, text=text, request=httpx.Request("POST", url))


def test_raise_for_status_attaches_the_response_body_to_a_permanent_error():
    """The reason a permanent 4xx skipped the doc lives in the body, not the
    status line — and a permanent error is never retried, so this log line is
    the only account of it anyone gets (#1655 needed a live probe to recover it)."""
    resp = _response(
        422,
        '{"error":{"message":"Value error, The input sequence should have less '
        'than 131072 characters. Input length: 180439"}}',
    )

    with pytest.raises(httpx.HTTPStatusError) as caught:
        raise_for_status(resp)

    message = str(caught.value)
    assert "less than 131072 characters" in message
    # httpx's own text is the prefix: `except` clauses and the log-derived HTTP
    # status parsing in deep health keep matching what they matched before.
    assert "Client error '422 Unprocessable Entity'" in message
    assert caught.value.response is resp
    assert not is_transient(caught.value)


@pytest.mark.parametrize("code", sorted(TRANSIENT_STATUSES))
def test_raise_for_status_leaves_a_transient_error_alone(code):
    """Retryable statuses are about to be retried; their body is noise until the
    ladder runs out, and they must stay classifiable as transient."""
    with pytest.raises(httpx.HTTPStatusError) as caught:
        raise_for_status(_response(code, "upstream is busy, try later"))

    assert "upstream is busy" not in str(caught.value)
    assert is_transient(caught.value)


def test_raise_for_status_truncates_and_flattens_an_unbounded_body():
    """A provider body is untrusted text: one physical line, bounded length (#0546)."""
    with pytest.raises(httpx.HTTPStatusError) as caught:
        raise_for_status(_response(400, "line one\nline two " + "x" * 5000))

    message = str(caught.value)
    assert "line one line two" in message
    assert "\n" not in message.split("response body: ", 1)[1]
    assert len(message) < 1000


def test_raise_for_status_keeps_the_original_error_when_the_body_is_empty():
    with pytest.raises(httpx.HTTPStatusError) as caught:
        raise_for_status(_response(404))

    assert str(caught.value).endswith(
        "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/404"
    )


def test_raise_for_status_passes_a_success_through():
    assert raise_for_status(_response(200, "{}")) is None


def test_enriched_message_still_parses_to_its_true_http_status():
    """The enriched line lands in indexer.log, which deep health parses with an
    ordered tuple of regexes — one of which reads `"code": NNN` out of JSON. A
    body can therefore carry a *different* number than the status; httpx's own
    `Client error 'NNN'` text must stay in the message and keep winning, or
    #0705's carve-out (400/413/422 = one bad document, not a provider outage)
    silently stops applying."""
    import mcp_server

    resp = _response(
        422,
        '{"error":{"message":"HTTP 500: upstream said 503","code":500}}',
    )
    with pytest.raises(httpx.HTTPStatusError) as caught:
        raise_for_status(resp)

    line = (
        "2026-08-26 12:00:00,000 ERROR prefect.flow_runs: Skipping "
        f"comm_messages::zoho_mail/x after retries exhausted: {caught.value}"
    )
    assert mcp_server._http_status_from_log_line(line) == 422
    assert mcp_server._provider_failure_kind(line) is None
