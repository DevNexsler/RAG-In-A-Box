"""Production-shaped RSS regression for Lance metadata schema widening."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa

from lancedb_store import LanceDBStore, _SchemaEvolutionRWLock
from llama_index.vector_stores.lancedb import LanceDBVectorStore


PRODUCTION_ROWS = 68_413
PRODUCTION_VECTOR_DIM = 4_096
PRODUCTION_METADATA_FIELDS = 102
PRODUCTION_DATA_FILES = 1_369
# Headroom is `mem_limit` (8 GiB) minus the doc-organizer resident baseline, and
# that baseline moves: 4.28 GiB (2026-08-08), 6.95 GiB (2026-08-06),
# 4.727 GiB (2026-08-11), 5.263 GiB (2026-08-12) — all sampled five times with
# the indexer idle. Budgeting against a single night's baseline is how #0926's
# check came to pass while the defect was live, so this asserts against the
# WORST observed baseline (6.95 GiB -> 1.05 GiB of headroom) rather than the
# most recent one. Measured cost of the bounded writer is ~525 MiB on the host
# and ~734 MiB inside an 8 GiB container, so a green run here means the widening
# fits even on production's worst observed day.
PRODUCTION_HEADROOM_BYTES = int(1.05 * 1024**3)
NEW_FIELDS = {"content_status", "content_failure_reasons"}


def _metadata_names() -> list[str]:
    real_names = [
        "source_type",
        "doc_id",
        "rel_path",
        "title",
        "change_hash",
        "loc",
        "enr_summary",
        "enr_doc_type",
        "enr_topics",
        "enr_key_facts",
        "enr_people",
        "enr_places",
        "building",
        "unit",
        "status",
        "sender",
        "sent_at",
        "direction",
        "channel_name",
        "source_message_id",
    ]
    return real_names + [
        f"meta_{index:03d}"
        for index in range(PRODUCTION_METADATA_FIELDS - len(real_names))
    ]


def _build_production_shaped_table(index_root: Path) -> None:
    metadata_names = _metadata_names()
    metadata_fields = [pa.field(name, pa.utf8()) for name in metadata_names]
    schema = pa.schema(
        [
            pa.field("id", pa.utf8()),
            pa.field("doc_id", pa.utf8()),
            pa.field("text", pa.utf8()),
            pa.field(
                "vector",
                pa.list_(pa.float32(), PRODUCTION_VECTOR_DIM),
            ),
            pa.field("metadata", pa.struct(metadata_fields)),
        ]
    )
    rows_per_file = (PRODUCTION_ROWS + PRODUCTION_DATA_FILES - 1) // PRODUCTION_DATA_FILES
    rng = np.random.default_rng(20260811)
    db = lancedb.connect(str(index_root))
    table = None
    made = 0
    while made < PRODUCTION_ROWS:
        row_count = min(rows_per_file, PRODUCTION_ROWS - made)
        vectors = rng.random(
            (row_count, PRODUCTION_VECTOR_DIM),
            dtype=np.float32,
        )
        vector_array = pa.FixedSizeListArray.from_arrays(
            pa.array(vectors.reshape(-1), type=pa.float32()),
            PRODUCTION_VECTOR_DIM,
        )
        ids = [f"documents::{made + offset:06d}" for offset in range(row_count)]
        metadata_arrays = []
        for name in metadata_names:
            if name == "source_type":
                values = ["img"] * row_count
            elif name == "doc_id":
                values = ids
            else:
                values = [f"{name}-{made + offset}" for offset in range(row_count)]
            metadata_arrays.append(pa.array(values, type=pa.utf8()))
        metadata = pa.StructArray.from_arrays(
            metadata_arrays,
            fields=metadata_fields,
        )
        batch = pa.RecordBatch.from_arrays(
            [
                pa.array([f"{doc_id}::c:0" for doc_id in ids]),
                pa.array(ids),
                pa.array([f"chunk text {made + offset}" for offset in range(row_count)]),
                vector_array,
                metadata,
            ],
            schema=schema,
        )
        if table is None:
            table = db.create_table("chunks", batch)
        else:
            table.add(batch)
        made += row_count


def _migration_store(index_root: Path) -> LanceDBStore:
    store = LanceDBStore.__new__(LanceDBStore)
    store.index_root = str(index_root)
    store.table_name = "chunks"
    store._schema_lock = _SchemaEvolutionRWLock()
    store._completed_insert_doc_ids = set()
    store._exclusive_writer_depth = 0
    store._memory_observer = None
    store._vs = LanceDBVectorStore(
        uri=str(index_root),
        table_name="chunks",
        mode="create",
    )
    return store


def _current_rss_bytes() -> int:
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def test_metadata_widening_fits_production_headroom(tmp_path):
    """Exact live row/vector/struct shape must widen below measured headroom."""
    index_root = tmp_path / "index"
    index_root.mkdir()
    _build_production_shaped_table(index_root)
    store = _migration_store(index_root)
    baseline_rss = _current_rss_bytes()
    peak_rss = baseline_rss
    stop = threading.Event()

    def sample_rss() -> None:
        nonlocal peak_rss
        while not stop.is_set():
            peak_rss = max(peak_rss, _current_rss_bytes())
            time.sleep(0.02)

    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()
    try:
        store._evolve_metadata_schema(set(NEW_FIELDS))
    finally:
        stop.set()
        sampler.join(timeout=5)

    rss_delta = peak_rss - baseline_rss
    print(
        "metadata_widening_memory "
        f"rows={PRODUCTION_ROWS} vector_dim={PRODUCTION_VECTOR_DIM} "
        f"metadata_fields={PRODUCTION_METADATA_FIELDS} "
        f"rss_delta_bytes={rss_delta} headroom_bytes={PRODUCTION_HEADROOM_BYTES}"
    )
    assert rss_delta <= PRODUCTION_HEADROOM_BYTES, (
        "metadata widening exceeded production headroom: "
        f"rss_delta={rss_delta} headroom={PRODUCTION_HEADROOM_BYTES}"
    )
    table = lancedb.connect(str(index_root)).open_table("chunks")
    assert table.count_rows() == PRODUCTION_ROWS
    metadata_type = table.schema.field("metadata").type
    assert NEW_FIELDS <= {field.name for field in metadata_type}
