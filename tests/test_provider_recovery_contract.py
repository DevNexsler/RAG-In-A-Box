"""Recovery must prove success before provider health turns green."""

import httpx
import pytest
import threading
from concurrent.futures import ThreadPoolExecutor

from core.resilience import CircuitOpenError, EndpointCircuits, raise_for_status


def test_failed_recovery_probe_cannot_clear_account_outage():
    now = [100.0]
    circuits = EndpointCircuits(cooldown=10, refusal_cooldown=30, clock=lambda: now[0])
    endpoint = "https://provider.example"
    request = httpx.Request("POST", endpoint)
    with pytest.raises(httpx.HTTPStatusError):
        with circuits.guard(endpoint):
            raise_for_status(httpx.Response(402, request=request))

    now[0] = 131.0
    with pytest.raises(httpx.HTTPStatusError):
        with circuits.guard(endpoint):
            raise_for_status(httpx.Response(503, request=request))

    assert endpoint in circuits.tripped(), "an unsuccessful probe is not recovery"
    with pytest.raises(CircuitOpenError):
        with circuits.guard(endpoint):
            pytest.fail("provider called again during recovery cooldown")

    now[0] = 142.0
    with circuits.guard(endpoint):
        pass
    assert circuits.tripped() == {}


@pytest.mark.parametrize("outcome", [None, httpx.ConnectError("stale failure")])
@pytest.mark.parametrize("reset", [False, True])
def test_old_request_cannot_change_new_recovery_probe(outcome, reset):
    now = [0.0]
    circuits = EndpointCircuits(threshold=1, cooldown=10, clock=lambda: now[0])
    endpoint = "https://provider.example"
    old = circuits.guard(endpoint)
    old.__enter__()
    if reset:
        circuits.reset()
    with pytest.raises(httpx.ConnectError):
        with circuits.guard(endpoint):
            raise httpx.ConnectError("current outage")
    now[0] = 11.0
    with circuits.guard(endpoint):
        before = circuits.tripped()
        old.__exit__(type(outcome) if outcome else None, outcome, None)
        assert circuits.tripped() == before
        now[0] = 100.0
        with pytest.raises(CircuitOpenError):
            with circuits.guard(endpoint):
                pytest.fail("stale completion released the active probe")
    assert circuits.tripped() == {}


def test_probe_body_circuit_error_releases_probe_for_next_recovery():
    now = [0.0]
    circuits = EndpointCircuits(threshold=1, cooldown=10, clock=lambda: now[0])
    endpoint = "https://provider.example"
    for error in (httpx.ConnectError("down"), CircuitOpenError("downstream unavailable")):
        with pytest.raises(type(error)):
            with circuits.guard(endpoint):
                raise error
        now[0] += 11
    with circuits.guard(endpoint):
        pass
    assert circuits.tripped() == {}


def test_slow_recovery_probe_keeps_concurrent_callers_out_after_cooldown():
    now = [0.0]
    circuits = EndpointCircuits(threshold=1, cooldown=10, clock=lambda: now[0])
    endpoint = "https://provider.example"
    with pytest.raises(httpx.ConnectError):
        with circuits.guard(endpoint):
            raise httpx.ConnectError("down")
    now[0] = 11.0
    entered, release = threading.Event(), threading.Event()

    def probe():
        with circuits.guard(endpoint):
            entered.set()
            assert release.wait(5), "test failed to release probe"

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(probe)
        try:
            assert entered.wait(5)
            now[0] = 100.0  # request outlives the cooldown; still only one probe
            with pytest.raises(CircuitOpenError):
                with circuits.guard(endpoint):
                    pass
        finally:
            release.set()
        future.result(timeout=5)
    assert circuits.tripped() == {}
