"""Per-endpoint circuit breaker (#0619).

A connection-refused is not per-document information — it is per-provider. The
00:31–00:33 outage burned 209 connection-refused retry warnings against a socket
that was refusing instantly, re-discovering the same fact once per document.
After N consecutive connection-level failures to one base_url the endpoint is
short-circuited for a cooldown; the calls fail instantly and transiently, so the
degraded lane still owns the retry.
"""

import httpx
import pytest

from core.resilience import (
    CircuitOpenError,
    EndpointCircuits,
    call_with_retry,
    is_transient,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _circuits(clock, threshold=3, cooldown=60.0):
    return EndpointCircuits(threshold=threshold, cooldown=cooldown, clock=clock)


def _refuse():
    raise httpx.ConnectError("[Errno 111] Connection refused")


def test_consecutive_connection_failures_trip_the_cooldown():
    clock = _Clock()
    circuits = _circuits(clock)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        _refuse()

    for _ in range(3):
        with pytest.raises(httpx.ConnectError):
            with circuits.guard("http://host.docker.internal:4000"):
                fn()
    assert calls["n"] == 3

    with pytest.raises(CircuitOpenError):
        with circuits.guard("http://host.docker.internal:4000"):
            fn()
    assert calls["n"] == 3  # the socket was not touched again


def test_open_circuit_failure_is_transient():
    # The doc must degrade + self-heal, never be capped as a permanent failure.
    clock = _Clock()
    circuits = _circuits(clock, threshold=1)
    with pytest.raises(httpx.ConnectError):
        with circuits.guard("http://p:4000"):
            _refuse()
    with pytest.raises(CircuitOpenError) as excinfo:
        with circuits.guard("http://p:4000"):
            _refuse()
    assert is_transient(excinfo.value)


def test_cooldown_expiry_admits_one_probe_and_closes_on_success():
    clock = _Clock()
    circuits = _circuits(clock, threshold=2, cooldown=60.0)
    for _ in range(2):
        with pytest.raises(httpx.ConnectError):
            with circuits.guard("http://p:4000"):
                _refuse()

    clock.advance(59.0)
    with pytest.raises(CircuitOpenError):
        with circuits.guard("http://p:4000"):
            _refuse()

    clock.advance(2.0)  # cooldown elapsed -> exactly one probe is admitted
    probed = {"n": 0}
    with circuits.guard("http://p:4000"):
        probed["n"] += 1
    assert probed["n"] == 1

    # Provider is back: the circuit is closed again, calls flow normally.
    with circuits.guard("http://p:4000"):
        probed["n"] += 1
    assert probed["n"] == 2


def test_failed_probe_reopens_for_a_fresh_cooldown():
    clock = _Clock()
    circuits = _circuits(clock, threshold=1, cooldown=30.0)
    with pytest.raises(httpx.ConnectError):
        with circuits.guard("http://p:4000"):
            _refuse()
    clock.advance(31.0)
    with pytest.raises(httpx.ConnectError):
        with circuits.guard("http://p:4000"):
            _refuse()
    with pytest.raises(CircuitOpenError):
        with circuits.guard("http://p:4000"):
            _refuse()


def test_circuit_is_per_endpoint():
    clock = _Clock()
    circuits = _circuits(clock, threshold=1)
    with pytest.raises(httpx.ConnectError):
        with circuits.guard("http://down:4000"):
            _refuse()

    healthy = {"n": 0}
    with circuits.guard("http://healthy:4000"):
        healthy["n"] += 1
    assert healthy["n"] == 1


def test_success_resets_the_failure_run():
    clock = _Clock()
    circuits = _circuits(clock, threshold=3)
    for _ in range(2):
        with pytest.raises(httpx.ConnectError):
            with circuits.guard("http://p:4000"):
                _refuse()
    with circuits.guard("http://p:4000"):
        pass
    with pytest.raises(httpx.ConnectError):
        with circuits.guard("http://p:4000"):
            _refuse()
    # Only one failure since the success — still closed.
    with circuits.guard("http://p:4000"):
        pass


def test_application_level_failures_do_not_trip_the_circuit():
    # A 500 or a bad payload means the endpoint ANSWERED; that is per-request
    # information, not "the provider is gone".
    clock = _Clock()
    circuits = _circuits(clock, threshold=2)
    request = httpx.Request("POST", "http://p:4000/v1/chat/completions")
    error = httpx.HTTPStatusError(
        "500", request=request, response=httpx.Response(500, request=request)
    )
    for _ in range(5):
        with pytest.raises(httpx.HTTPStatusError):
            with circuits.guard("http://p:4000"):
                raise error
    with circuits.guard("http://p:4000"):
        pass


def test_call_with_retry_short_circuits_without_burning_backoff():
    clock = _Clock()
    circuits = _circuits(clock, threshold=2)
    slept: list[float] = []
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        _refuse()

    with pytest.raises(httpx.ConnectError):
        call_with_retry(
            fn, attempts=3, backoff=(0.1, 0.2), label="t",
            sleep=slept.append, circuit_key="http://p:4000", circuits=circuits,
        )
    assert calls["n"] == 2  # tripped mid-retry: 2 real attempts, then open
    assert slept == [0.1]

    slept.clear()
    with pytest.raises(CircuitOpenError):
        call_with_retry(
            fn, attempts=3, backoff=(0.1, 0.2), label="t",
            sleep=slept.append, circuit_key="http://p:4000", circuits=circuits,
        )
    assert calls["n"] == 2  # no socket touched
    assert slept == []      # and no backoff burned on a known-down provider
