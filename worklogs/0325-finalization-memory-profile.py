#!/usr/bin/env python3
"""Profile doc-organizer finalization against an isolated Lance snapshot.

This operator harness intentionally calls the same private maintenance steps as
``LanceDBStore._optimize_and_prune`` so phase-level cgroup peaks identify the
responsible boundary. It must never be pointed at production state.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import lance

from lancedb_store import LanceDBStore
from taxonomy_store import TaxonomyStore


def _read_int(path: Path) -> int:
    return int(path.read_text().strip())


def _memory_events(cgroup: Path) -> dict[str, int]:
    values = dict(line.split() for line in (cgroup / "memory.events").read_text().splitlines())
    return {name: int(values.get(name, 0)) for name in ("max", "oom", "oom_kill")}


def _rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


class PhaseSampler:
    def __init__(self, cgroup: Path, interval: float) -> None:
        self._cgroup = cgroup
        self._interval = interval
        self._phase = "startup"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.peaks: dict[str, dict[str, int]] = defaultdict(
            lambda: {"memory_current": 0, "rss": 0, "pids": 0}
        )
        self.events_before = _memory_events(cgroup)

    def start(self) -> None:
        self._thread.start()

    def phase(self, name: str) -> None:
        with self._lock:
            self._phase = name
        self.sample()

    def sample(self) -> None:
        with self._lock:
            phase = self._phase
        current = {
            "memory_current": _read_int(self._cgroup / "memory.current"),
            "rss": _rss_bytes(),
            "pids": _read_int(self._cgroup / "pids.current"),
        }
        peak = self.peaks[phase]
        for key, value in current.items():
            peak[key] = max(peak[key], value)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self.sample()

    def finish(self) -> dict[str, Any]:
        self.sample()
        self._stop.set()
        self._thread.join(timeout=2)
        events_after = _memory_events(self._cgroup)
        return {
            "phase_peaks": dict(self.peaks),
            "events_before": self.events_before,
            "events_after": events_after,
            "event_delta": {
                key: events_after[key] - self.events_before[key] for key in events_after
            },
        }


def _taxonomy_ids_and_dimension(index_root: Path) -> tuple[list[str], int]:
    dataset = lance.dataset(str(index_root / "taxonomy.lance"))
    ids = [str(value) for value in dataset.to_table(columns=["id"])["id"].to_pylist()]
    vector_type = dataset.schema.field("vector").type
    dimension = int(getattr(vector_type, "list_size"))
    return ids, dimension


def _profile(args: argparse.Namespace) -> dict[str, Any]:
    index_root = Path(args.index_root)
    marker = index_root / "chunks.lance.last-compaction"
    if args.current_marker:
        marker.write_text(date.today().isoformat())

    ids, dimension = _taxonomy_ids_and_dimension(index_root)
    embed_calls = 0

    def embed(_text: str) -> list[float]:
        nonlocal embed_calls
        embed_calls += 1
        return [0.0] * dimension

    cgroup = Path("/sys/fs/cgroup")
    sampler = PhaseSampler(cgroup, args.sample_interval)
    sampler.start()
    started = time.monotonic()
    result: dict[str, Any] = {
        "pid": os.getpid(),
        "index_root": str(index_root),
        "marker_before": marker.read_text().strip() if marker.exists() else None,
        "taxonomy_ids": len(ids),
    }
    try:
        sampler.phase("taxonomy_usage")
        taxonomy = TaxonomyStore(index_root, "taxonomy", embed_fn=embed)
        taxonomy.increment_usage_many({entry_id: 1 for entry_id in ids})
        result["taxonomy_embed_calls"] = embed_calls

        sampler.phase("store_open")
        store = LanceDBStore(index_root, "chunks")
        table = store._vs.table

        sampler.phase("prune_pre")
        store._prune_versions("pre-maintenance")

        today = date.today()
        if store._compaction_due(today):
            sampler.phase("daily_compaction")
            table.checkout_latest()
            if args.compaction_threads is None:
                table.optimize()
            else:
                dataset = lance.dataset(store._dataset_path())
                dataset.optimize.compact_files(
                    num_threads=args.compaction_threads,
                    batch_size=args.compaction_batch_size,
                )
                sampler.phase("index_delta_merge")
                store._merge_index_deltas()
            store._record_compaction(today)
            result["compaction"] = (
                "due_default"
                if args.compaction_threads is None
                else "due_bounded"
            )
        else:
            sampler.phase("index_delta_merge")
            store._merge_index_deltas()
            result["compaction"] = "skipped_current_marker"

        sampler.phase("restore_points")
        store._manage_restore_points(table, today)

        sampler.phase("prune_post")
        store._prune_versions("post-expiry")

        sampler.phase("final_counts")
        result["doc_count"] = len(store.list_doc_ids())
        result["chunk_count"] = store.count_chunks()
        result["marker_after"] = marker.read_text().strip() if marker.exists() else None
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        return result
    finally:
        result.update(sampler.finish())
        print("PROFILE_SUMMARY=" + json.dumps(result, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-root", default="/data/index")
    parser.add_argument("--sample-interval", type=float, default=0.1)
    parser.add_argument("--current-marker", action="store_true")
    parser.add_argument("--compaction-threads", type=int)
    parser.add_argument("--compaction-batch-size", type=int, default=1024)
    args = parser.parse_args()
    _profile(args)


if __name__ == "__main__":
    main()
