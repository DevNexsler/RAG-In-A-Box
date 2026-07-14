import pytest
from core.fallback import resolve_with_fallback
from core.resilience import TransientError


def _raises_transient():
    raise TransientError("unreachable")


def test_primary_text_passthrough_no_fallback_called():
    calls = {"fb": 0}
    def fb():
        calls["fb"] += 1
        return "fallback"
    assert resolve_with_fallback(lambda: "primary text", fb) == "primary text"
    assert calls["fb"] == 0  # cost guard: fallback not called when primary succeeds


def test_primary_unreachable_propagates_and_fallback_not_called():
    calls = {"fb": 0}
    def fb():
        calls["fb"] += 1
        return "fallback"
    with pytest.raises(TransientError):
        resolve_with_fallback(_raises_transient, fb)
    assert calls["fb"] == 0  # cost guard: no fallback on unreachable primary


def test_dark_mode_empty_raises_transient():
    # fallback is None -> unconfirmed empty must retry, never be treated as clean
    with pytest.raises(TransientError):
        resolve_with_fallback(lambda: "", None)


def test_reachable_empty_fallback_text_recovers():
    assert resolve_with_fallback(lambda: "   ", lambda: "recovered") == "recovered"


def test_reachable_empty_fallback_empty_confirms_blank():
    assert resolve_with_fallback(lambda: "", lambda: "") == ""


def test_reachable_empty_fallback_unreachable_raises_transient():
    with pytest.raises(TransientError):
        resolve_with_fallback(lambda: "", _raises_transient)
