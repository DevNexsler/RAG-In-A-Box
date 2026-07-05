"""OTEL SDK tracing with a JSONL file exporter. No collector, no UI.

Off unless config has tracing.enabled: true. Span JSONL lands in
tracing.directory (default .evals/spans), one file per process start.

Design note: SimpleSpanProcessor (synchronous) is deliberate — spans are
per-document, not per-token, and it guarantees spans are on disk when a test
asserts. If profiling ever shows overhead, switch to BatchSpanProcessor +
flush in shutdown_tracing.
"""

import json
import logging
import os
import threading
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

logger = logging.getLogger(__name__)

_provider = None
_lock = threading.Lock()


class JsonlSpanExporter(SpanExporter):
    def __init__(self, directory: str):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"spans-{os.getpid()}.jsonl"
        self._flock = threading.Lock()
        self._stopped = False

    def export(self, spans) -> SpanExportResult:
        if self._stopped:
            return SpanExportResult.SUCCESS  # no writes after shutdown
        lines = []
        for s in spans:
            ctx, parent = s.get_span_context(), s.parent
            lines.append(json.dumps({
                "name": s.name,
                "trace_id": format(ctx.trace_id, "032x"),
                "span_id": format(ctx.span_id, "016x"),
                "parent_span_id": format(parent.span_id, "016x") if parent else None,
                "start_ns": s.start_time, "end_ns": s.end_time,
                "status": s.status.status_code.name,
                "attributes": dict(s.attributes or {}),
            }, default=str))
        if not lines:
            return SpanExportResult.SUCCESS
        try:
            with self._flock, self._path.open("a") as f:
                f.write("\n".join(lines) + "\n")
            return SpanExportResult.SUCCESS
        except OSError:
            return SpanExportResult.FAILURE  # never crash the pipeline over tracing

    def shutdown(self):
        self._stopped = True


class _LazyTracer:
    """Resolves the provider per call, so module-level
    `_tracer = get_tracer(...)` survives shutdown/re-setup."""

    __slots__ = ("_name",)

    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        return getattr((_provider or trace).get_tracer(self._name), attr)


def setup_tracing(config: dict, service_name: str) -> None:
    global _provider
    tcfg = (config or {}).get("tracing") or {}
    if not tcfg.get("enabled", False):
        return
    with _lock:
        if _provider is not None:
            return
        try:
            provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
            exporter = JsonlSpanExporter(tcfg.get("directory", ".evals/spans"))
            provider.add_span_processor(SimpleSpanProcessor(exporter))
            trace.set_tracer_provider(provider)
            _provider = provider
        except Exception:
            logger.warning("tracing setup failed; continuing without tracing", exc_info=True)
            _provider = None
            return
        try:
            # Binds to the OTEL global once; a re-setup will log "already
            # instrumented" — expected, prod does a single setup per process.
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
            HTTPXClientInstrumentor().instrument()
        except Exception:
            pass  # tracing must never take the service down


def get_tracer(name: str):
    # Lazy: trace.set_tracer_provider() refuses to override the OTEL global
    # once set, so anything bound eagerly (global proxy or a cached concrete
    # tracer) would keep pointing at an old, shut-down provider.
    return _LazyTracer(name)


def shutdown_tracing() -> None:
    global _provider
    with _lock:
        if _provider is not None:
            _provider.shutdown()
            _provider = None
