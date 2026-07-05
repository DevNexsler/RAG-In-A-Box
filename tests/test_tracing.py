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
