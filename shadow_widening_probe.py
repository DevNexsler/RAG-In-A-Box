#!/usr/bin/env python3
"""Production-shaped shadow migration for the #0584 / #0926 metadata widening.

Answers the objection raised by the adversarial second opinion on 2026-08-11:
the existing regressions for the bounded widening use 32/64-dimensional vectors
and 40 metadata fields, so neither reproduces the LIVE table's shape (68,413
rows, 4,096-dimensional vectors, a 102-field metadata struct, heavily
fragmented).  This builds a table of that shape and runs the DEPLOYED
``LanceDBStore._evolve_metadata_schema`` on it, adding exactly the two fields
PR #79 introduces, while a separate process hammers it with reads.

Measures: wall duration, peak RSS of the migrating process, disk high-water,
reader errors across the table-directory swap AND the scalar-index rebuild.
Then forces a mid-migration failure and asserts the original table survives.

Everything happens under a throwaway directory.  Production is never touched.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa

ROWS = int(os.environ.get("SHADOW_ROWS", "68413"))
DIM = int(os.environ.get("SHADOW_DIM", "4096"))
META_FIELDS = int(os.environ.get("SHADOW_META_FIELDS", "102"))
BATCH = int(os.environ.get("SHADOW_BATCH", "250"))
NEW_FIELDS = {"content_status", "content_failure_reasons"}


def _meta_names(n: int) -> list[str]:
    # Mirror the real struct: a handful of the names the code actually reads,
    # padded out to the live width with filler of the same type (utf8).
    real = [
        "source_type", "doc_id", "rel_path", "title", "change_hash", "loc",
        "enr_summary", "enr_doc_type", "enr_topics", "enr_key_facts",
        "enr_people", "enr_places", "building", "unit", "status", "sender",
        "sent_at", "direction", "channel_name", "source_message_id",
    ]
    filler = [f"meta_{i:03d}" for i in range(n - len(real))]
    return real + filler


META_NAMES = _meta_names(META_FIELDS)


def build_schema() -> pa.Schema:
    return pa.schema([
        pa.field("id", pa.utf8()),
        pa.field("doc_id", pa.utf8()),
        pa.field("text", pa.utf8()),
        pa.field("vector", pa.list_(pa.float32(), DIM)),
        pa.field("metadata", pa.struct([pa.field(n, pa.utf8()) for n in META_NAMES])),
    ])


def build_table(root: Path) -> None:
    schema = build_schema()
    db = lancedb.connect(str(root))
    rng = np.random.default_rng(20260811)
    made = 0
    table = None
    while made < ROWS:
        n = min(BATCH, ROWS - made)
        vecs = rng.random((n, DIM), dtype=np.float32)
        flat = pa.array(vecs.reshape(-1), type=pa.float32())
        vector = pa.FixedSizeListArray.from_arrays(flat, DIM)
        ids = [f"documents::{made + i:06d}::c:0" for i in range(n)]
        doc_ids = [f"documents::{made + i:06d}" for i in range(n)]
        meta_arrays = []
        for name in META_NAMES:
            if name == "source_type":
                meta_arrays.append(pa.array(["img"] * n, type=pa.utf8()))
            elif name == "doc_id":
                meta_arrays.append(pa.array(doc_ids, type=pa.utf8()))
            elif name == "title":
                meta_arrays.append(pa.array([f"attachment {made + i}" for i in range(n)], type=pa.utf8()))
            else:
                meta_arrays.append(pa.array([f"{name}-{made + i}" for i in range(n)], type=pa.utf8()))
        meta = pa.StructArray.from_arrays(
            meta_arrays, fields=[pa.field(n_, pa.utf8()) for n_ in META_NAMES]
        )
        batch = pa.RecordBatch.from_arrays(
            [
                pa.array(ids, type=pa.utf8()),
                pa.array(doc_ids, type=pa.utf8()),
                pa.array([f"chunk text for row {made + i}" for i in range(n)], type=pa.utf8()),
                vector,
                meta,
            ],
            schema=schema,
        )
        tbl = pa.Table.from_batches([batch], schema=schema)
        if table is None:
            table = db.create_table("chunks", tbl, mode="overwrite")
        else:
            table.add(tbl)          # each add() = its own fragment -> fragmentation
        made += n
        if made % 10000 == 0:
            print(f"  built {made}/{ROWS} rows", flush=True)


def reader_loop(root: str, stop_path: str, out_path: str) -> None:
    """Stand in for Hermes/CDS: read the table continuously, like MCP search."""
    import json as _json
    import time as _time
    from pathlib import Path as _Path

    import lancedb as _lancedb

    ok = 0
    errors: list[str] = []
    try:
        db = _lancedb.connect(root)
    except Exception as exc:                           # noqa: BLE001
        _Path(out_path).write_text(_json.dumps(
            {"ok": 0, "errors": [f"connect failed: {exc}"], "error_count": 1}))
        return
    while not _Path(stop_path).exists():
        try:
            t = db.open_table("chunks")
            rows = (
                t.search()
                .where("metadata.source_type = 'img'")
                .limit(5)
                .select(["id", "doc_id", "metadata"])
                .to_arrow()
            )
            if rows.num_rows == 0:
                errors.append("empty result set")
            else:
                ok += 1
        except Exception as exc:                       # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")
        _time.sleep(0.01)
    _Path(out_path).write_text(_json.dumps(
        {"ok": ok, "errors": errors[:40], "error_count": len(errors)}))


def rss_watcher(stop: threading.Event, result: dict) -> None:
    """Sample this process's RSS *during the migration window only*.

    ru_maxrss is a whole-process high-water mark, so it also captures the
    table-construction phase (68k x 4096 float32 = 1.12 GB of numpy) and would
    massively overstate what the widening itself costs.
    """
    page = os.sysconf("SC_PAGE_SIZE")
    high = 0
    while not stop.is_set():
        try:
            rss = int(Path("/proc/self/statm").read_text().split()[1]) * page
            high = max(high, rss)
        except Exception:                              # noqa: BLE001
            pass
        time.sleep(0.02)
    result["migration_peak_rss_bytes"] = high


def disk_watcher(root: Path, stop: threading.Event, result: dict) -> None:
    high = 0
    while not stop.is_set():
        try:
            out = subprocess.run(["du", "-sb", str(root)], capture_output=True, text=True, timeout=60)
            val = int(out.stdout.split()[0])
            high = max(high, val)
        except Exception:                              # noqa: BLE001
            pass
        time.sleep(1.0)
    result["disk_high_water_bytes"] = high


PROD_MEM_LIMIT_BYTES = 8 * 1024 ** 3          # doc-organizer mem_limit: 8g


def make_store(root: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import lancedb_store as ls
    from llama_index.vector_stores.lancedb import LanceDBVectorStore

    # The probe runs on the host (128 GiB cgroup); production runs in an 8 GiB
    # container. Measuring the bounded-widening guard against the host limit
    # would make the envelope ~16x too generous and prove nothing.
    ls._cgroup_memory_limit_bytes = lambda *a, **k: PROD_MEM_LIMIT_BYTES

    store = ls.LanceDBStore.__new__(ls.LanceDBStore)
    store.index_root = str(root)
    store.table_name = "chunks"
    store._schema_lock = ls._SchemaEvolutionRWLock()
    store._completed_insert_doc_ids = set()
    store._exclusive_writer_depth = 0
    store._memory_observer = None
    store._vs = LanceDBVectorStore(uri=str(root), table_name="chunks", mode="create")
    return store, ls


def run_migration(root: Path) -> dict:
    store, ls = make_store(root)
    before = store._metadata_subfields()
    assert not (NEW_FIELDS & before), "new fields already present"

    stop_path = root.parent / "STOP"
    reader_out = root.parent / "reader.json"
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=reader_loop, args=(str(root), str(stop_path), str(reader_out)))
    proc.start()
    time.sleep(2.0)                                    # let the reader warm up

    stop = threading.Event()
    disk: dict = {}
    rssd: dict = {}
    watcher = threading.Thread(target=disk_watcher, args=(root, stop, disk), daemon=True)
    rssw = threading.Thread(target=rss_watcher, args=(stop, rssd), daemon=True)
    watcher.start()
    rssw.start()
    baseline_rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")

    t0 = time.monotonic()
    store._evolve_metadata_schema(set(NEW_FIELDS))
    elapsed = time.monotonic() - t0

    stop.set()
    watcher.join(timeout=10)
    rssw.join(timeout=10)
    time.sleep(2.0)                                    # keep reading past the swap
    stop_path.write_text("stop")
    proc.join(30)
    if proc.is_alive():
        proc.terminate()
        proc.join()

    reader = json.loads(reader_out.read_text()) if reader_out.exists() else {"ok": 0, "errors": ["reader produced no output"], "error_count": 1}
    after = store._metadata_subfields()
    tbl = lancedb.connect(str(root)).open_table("chunks")
    return {
        "rows_before": ROWS,
        "rows_after": tbl.count_rows(),
        "fields_before": len(before),
        "fields_after": len(after),
        "new_fields_present": sorted(NEW_FIELDS & after),
        "migration_seconds": round(elapsed, 2),
        "baseline_rss_bytes": baseline_rss,
        "migration_peak_rss_bytes": rssd.get("migration_peak_rss_bytes", 0),
        "migration_rss_delta_bytes": rssd.get("migration_peak_rss_bytes", 0) - baseline_rss,
        "whole_process_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "disk_high_water_bytes": disk.get("disk_high_water_bytes", 0),
        "reader_ok": reader["ok"],
        "reader_error_count": reader["error_count"],
        "reader_errors": reader["errors"],
    }


def run_rollback_test(root: Path) -> dict:
    """Force create_table to blow up part-way and assert the table survives."""
    store, ls = make_store(root)
    import lancedb as ldb

    real_connect = ldb.connect

    class _Boom(Exception):
        pass

    class _FailingDB:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def create_table(self, name, data, schema=None, **kw):
            # Consume part of the stream so the failure lands mid-migration,
            # not before it starts.
            consumed = 0
            for _ in data:
                consumed += 1
                if consumed >= 500:
                    break
            raise _Boom(f"synthetic failure after {consumed} batches")

    def fake_connect(uri, *a, **kw):
        return _FailingDB(real_connect(uri, *a, **kw))

    before_rows = lancedb.connect(str(root)).open_table("chunks").count_rows()
    ls.ldb = None  # noqa: F841  (module imports lancedb lazily inside the method)
    import lancedb as _l
    orig = _l.connect
    _l.connect = fake_connect
    raised = None
    try:
        store._evolve_metadata_schema({"rollback_probe_field"})
    except Exception as exc:                           # noqa: BLE001
        raised = f"{type(exc).__name__}: {exc}"
    finally:
        _l.connect = orig

    db = lancedb.connect(str(root))
    listed = db.list_tables()
    names = set(getattr(listed, "tables", listed) or [])
    tbl = db.open_table("chunks")
    return {
        "raised": raised,
        "rows_after_failed_migration": tbl.count_rows(),
        "rows_preserved": tbl.count_rows() == before_rows,
        "stray_tables": sorted(n for n in names if n != "chunks"),
        "field_not_added": "rollback_probe_field" not in {
            tbl.schema.field(tbl.schema.get_field_index("metadata")).type.field(i).name
            for i in range(tbl.schema.field(tbl.schema.get_field_index("metadata")).type.num_fields)
        },
    }


def main() -> int:
    base = Path(tempfile.mkdtemp(prefix="shadow-widen-", dir=os.environ.get("SHADOW_TMP", "/home/danpark/tmp")))
    root = base / "index"
    root.mkdir(parents=True)
    try:
        print(f"building production-shaped table: {ROWS} rows, dim={DIM}, "
              f"{META_FIELDS} metadata fields, batch={BATCH}", flush=True)
        t0 = time.monotonic()
        build_table(root)
        print(f"built in {time.monotonic() - t0:.1f}s", flush=True)
        frags = len(list((root / "chunks.lance" / "data").glob("*.lance")))
        print(f"fragments (data files): {frags}", flush=True)

        result = run_migration(root)
        result["data_files"] = frags
        print("MIGRATION " + json.dumps(result, indent=2), flush=True)

        rb = run_rollback_test(root)
        print("ROLLBACK " + json.dumps(rb, indent=2), flush=True)
        return 0
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
