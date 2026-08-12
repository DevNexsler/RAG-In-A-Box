#!/usr/bin/env python3
"""Which phase of the bounded widening actually allocates 3.8 GiB?

Same production-shaped table as shadow_widening_probe.py, but with an RSS
sampler running continuously and markers recorded around create_table (the
streaming write) and _ensure_scalar_index (the index rebuild).
"""
from __future__ import annotations

import json, os, shutil, sys, tempfile, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shadow_widening_probe as swp   # reuse build_table / make_store

PAGE = os.sysconf("SC_PAGE_SIZE")
def rss() -> int:
    return int(Path("/proc/self/statm").read_text().split()[1]) * PAGE

samples: list[tuple[float, int]] = []
marks: list[tuple[str, float, int]] = []
stop = threading.Event()

def sampler():
    while not stop.is_set():
        samples.append((time.monotonic(), rss()))
        time.sleep(0.02)

def mark(name: str):
    marks.append((name, time.monotonic(), rss()))

def peak_between(t0: float, t1: float) -> int:
    vals = [r for t, r in samples if t0 <= t <= t1]
    return max(vals) if vals else 0

base = Path(tempfile.mkdtemp(prefix="shadow-phase-", dir="/home/danpark/tmp"))
root = base / "index"; root.mkdir(parents=True)
try:
    swp.build_table(root)
    store, ls = swp.make_store(root)

    import lancedb as _l
    real_create = None
    orig_connect = _l.connect

    class _Wrapped:
        def __init__(self, inner): self._inner = inner
        def __getattr__(self, n): return getattr(self._inner, n)
        def create_table(self, *a, **kw):
            mark("create_table:start")
            try:
                return self._inner.create_table(*a, **kw)
            finally:
                mark("create_table:end")

    _l.connect = lambda uri, *a, **kw: _Wrapped(orig_connect(uri, *a, **kw))

    orig_idx = type(store)._ensure_scalar_index
    def wrapped_idx(self):
        mark("scalar_index:start")
        try:
            return orig_idx(self)
        finally:
            mark("scalar_index:end")
    type(store)._ensure_scalar_index = wrapped_idx

    t = threading.Thread(target=sampler, daemon=True); t.start()
    baseline = rss()
    mark("migration:start")
    store._evolve_metadata_schema({"content_status", "content_failure_reasons"})
    mark("migration:end")
    stop.set(); t.join(timeout=5)
    _l.connect = orig_connect

    m = dict((n, (ts, r)) for n, ts, r in marks)
    out = {"baseline_gib": round(baseline / 2**30, 2)}
    for phase, a, b in (
        ("streaming_write_create_table", "create_table:start", "create_table:end"),
        ("scalar_index_rebuild", "scalar_index:start", "scalar_index:end"),
        ("whole_migration", "migration:start", "migration:end"),
    ):
        if a in m and b in m:
            pk = peak_between(m[a][0], m[b][0])
            out[phase] = {
                "seconds": round(m[b][0] - m[a][0], 1),
                "peak_rss_gib": round(pk / 2**30, 2),
                "delta_over_baseline_gib": round((pk - baseline) / 2**30, 2),
            }
    print(json.dumps(out, indent=2))
finally:
    shutil.rmtree(base, ignore_errors=True)
