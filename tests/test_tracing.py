"""Tests for core.tracing (OTEL SDK + JSONL span exporter).

NOTE on ordering: OTEL's global tracer provider can only be set once per
process, so the disabled test runs FIRST (pytest runs tests in definition
order) while the global provider is still the default no-op one.
If pytest-randomly is ever installed, it would shuffle definition order and
break this assumption.
"""

import json

import pytest

from core.tracing import get_tracer, setup_tracing, shutdown_tracing

pytestmark = pytest.mark.unit


def test_disabled_is_noop(tmp_path):
    setup_tracing({"tracing": {"enabled": False}}, service_name="t")
    with get_tracer("x").start_as_current_span("s"):
        pass  # must not raise, must not write anything
    assert not list(tmp_path.rglob("*.jsonl"))


def test_spans_written_as_jsonl(tmp_path):
    cfg = {"tracing": {"enabled": True, "directory": str(tmp_path)}}
    setup_tracing(cfg, service_name="test-svc")
    tracer = get_tracer("pipeline")
    with tracer.start_as_current_span("extract", attributes={"doc_id": "d1"}):
        with tracer.start_as_current_span("ocr"):
            pass
    shutdown_tracing()
    files = list(tmp_path.glob("*.jsonl"))
    assert files
    spans = [json.loads(l) for f in files for l in f.read_text().splitlines()]
    by_name = {s["name"]: s for s in spans}
    assert by_name["extract"]["attributes"]["doc_id"] == "d1"
    assert by_name["ocr"]["parent_span_id"] == by_name["extract"]["span_id"]
    assert by_name["ocr"]["trace_id"] == by_name["extract"]["trace_id"]


def test_reinit_after_shutdown_writes_spans(tmp_path):
    d1, d2 = tmp_path / "a", tmp_path / "b"
    setup_tracing({"tracing": {"enabled": True, "directory": str(d1)}}, service_name="t1")
    with get_tracer("p").start_as_current_span("first"):
        pass
    shutdown_tracing()
    setup_tracing({"tracing": {"enabled": True, "directory": str(d2)}}, service_name="t2")
    with get_tracer("p").start_as_current_span("second"):
        pass
    shutdown_tracing()
    assert list(d2.glob("*.jsonl")), "spans lost after re-setup"


def test_cached_tracer_survives_reinit(tmp_path):
    # Module-level `_tracer = get_tracer(...)` at import time must keep
    # working across shutdown/re-setup, not stay bound to a dead provider.
    t = get_tracer("cached")  # cached BEFORE any setup
    da, db = tmp_path / "a", tmp_path / "b"
    setup_tracing({"tracing": {"enabled": True, "directory": str(da)}}, service_name="t1")
    with t.start_as_current_span("first"):
        pass
    shutdown_tracing()
    setup_tracing({"tracing": {"enabled": True, "directory": str(db)}}, service_name="t2")
    with t.start_as_current_span("second"):
        pass
    shutdown_tracing()

    a_files = list(da.glob("*.jsonl"))
    assert a_files, "first span missing from dir A"
    a_names = [json.loads(l)["name"] for f in a_files for l in f.read_text().splitlines()]
    assert a_names == ["first"], f"dir A must hold only the first span, got {a_names}"

    b_files = list(db.glob("*.jsonl"))
    assert b_files, "second span lost after re-setup via cached tracer"
    b_names = [json.loads(l)["name"] for f in b_files for l in f.read_text().splitlines()]
    assert "second" in b_names


def test_setup_never_crashes_on_bad_directory(tmp_path):
    # A read-only/invalid span directory must not take the service down at boot.
    blocker = tmp_path / "f"
    blocker.write_text("not a directory")
    cfg = {"tracing": {"enabled": True, "directory": str(blocker / "sub")}}
    setup_tracing(cfg, service_name="t")  # must not raise
    with get_tracer("x").start_as_current_span("s"):
        pass  # must no-op without raising
    import core.tracing as tracing_mod
    assert tracing_mod._provider is None
    shutdown_tracing()  # harmless no-op


def test_no_writes_after_shutdown(tmp_path):
    import core.tracing as tracing_mod
    setup_tracing({"tracing": {"enabled": True, "directory": str(tmp_path)}}, service_name="t")
    # Concrete (stale) tracer bound to this provider, bypassing lazy resolution.
    stale = tracing_mod._provider.get_tracer("stale")
    with stale.start_as_current_span("before"):
        pass
    files = list(tmp_path.glob("*.jsonl"))
    assert files
    n_before = len(files[0].read_text().splitlines())
    shutdown_tracing()
    with stale.start_as_current_span("after"):
        pass
    assert len(files[0].read_text().splitlines()) == n_before, "span written after shutdown"
