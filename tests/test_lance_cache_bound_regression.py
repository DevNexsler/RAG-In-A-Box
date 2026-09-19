"""Deterministic guards for the native cache-bound integration probe (#2188).

Native eviction timing varies with scheduling. Replay cache readings at the
workload boundary so a single-reading assertion cannot silently return.
The integration tier still exercises the real index and native sessions.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


MB = 1024 * 1024


@pytest.fixture
def cache_probe(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "cache_bound_probe", Path(__file__).with_name("test_lance_session_bounds.int.test.py")
    )
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)

    def run(readings, *, calibration_bytes=20 * MB, items=1):
        session = SimpleNamespace(size_bytes=0, approx_num_items=items)
        calibration = SimpleNamespace(size_bytes=calibration_bytes)
        monkeypatch.setattr(probe, "lancedb", Mock())
        probe.lancedb.Session.default.return_value = calibration
        monkeypatch.setattr(probe, "lance_session", Mock())
        probe.lance_session.configure.return_value = session
        monkeypatch.setattr(probe, "LanceDBStore", Mock())
        monkeypatch.setattr(probe, "_build_indexed_table", Mock())
        # First workload calibrates the default session; later rounds update
        # the bounded session independently of when its metrics are sampled.
        rounds = iter([0, *readings])

        def drive_reads(*args, **kwargs):
            session.size_bytes = next(rounds)

        monkeypatch.setattr(probe, "_drive_reads", drive_reads)
        probe.test_shared_session_caps_lance_cache_growth(tmp_path)

    return run


def test_lazy_eviction_can_return_under_cap_before_final_reading(cache_probe):
    cache_probe([5 * MB, 12 * MB, 14 * MB, 13 * MB])


def test_unbounded_cache_is_rejected(cache_probe):
    with pytest.raises(AssertionError):
        cache_probe([19 * MB, 20 * MB, 20 * MB, 20 * MB])


def test_small_calibration_workload_is_rejected(cache_probe):
    with pytest.raises(AssertionError, match="workload too small"):
        cache_probe([5 * MB] * 4, calibration_bytes=6 * MB)


def test_unused_shared_session_is_rejected(cache_probe):
    with pytest.raises(AssertionError):
        cache_probe([0] * 4, items=0)
