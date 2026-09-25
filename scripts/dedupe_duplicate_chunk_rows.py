#!/usr/bin/env python3
"""One-time maintenance: collapse duplicate physical rows that share a chunk id.

Background (#3174)
------------------
When the mid-sweep queue lane upserted a document and the sweep lane later
inserted the same chunk id again (#3143), LanceDB held two live rows with
identical primary keys and different enrichment text. The write-path fix stops
new duplicates, but existing production rows must be compacted manually.

This script scans the ``chunks`` table for duplicate chunk ids, keeps the
newest physical row per id (highest Lance ``_rowid``), and rewrites each
affected document through the normal upsert path so exactly one row remains per
chunk id.

Usage
-----
    # inside the doc-organizer container (index at /data/index):
    python3 scripts/dedupe_duplicate_chunk_rows.py --config /app/config.yaml
    python3 scripts/dedupe_duplicate_chunk_rows.py --config /app/config.yaml --apply
    python3 scripts/dedupe_duplicate_chunk_rows.py --config /app/config.yaml \\
        --doc-id documents::002V2 --doc-id documents::002V3 --apply

Without ``--apply`` the script reports the census and planned compaction only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core import lance_session  # noqa: E402
from core.config import load_config  # noqa: E402
from core.index_write_lock import IndexWriteLockBusy, index_write_lock  # noqa: E402
from lancedb_store import LanceDBStore  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=Path("/app/config.yaml"))
    parser.add_argument("--index-root", type=Path, default=None, help="override config index_root")
    parser.add_argument("--table", default=None, help="override config lancedb.table")
    parser.add_argument(
        "--doc-id",
        action="append",
        default=[],
        help="limit compaction to specific doc ids (repeatable)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="rewrite affected documents; default is dry-run census only",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="exit 3 if another index writer holds the table",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    index_root = Path(args.index_root or config["index_root"])
    table_name = args.table or config.get("lancedb", {}).get("table", "chunks")
    lance_session.configure_from_config(config)

    report: dict = {
        "index_root": str(index_root),
        "table": table_name,
        "apply": args.apply,
    }
    try:
        with index_write_lock(index_root, table_name, blocking=not args.no_wait):
            store = LanceDBStore(index_root, table_name)
            census = store.duplicate_chunk_id_census()
            report["duplicate_chunk_ids_before"] = len(census)
            report["duplicate_chunk_ids"] = dict(sorted(census.items()))
            if args.doc_id:
                report["doc_ids"] = sorted(args.doc_id)
            result = store.compact_duplicate_chunk_rows(
                args.doc_id or None,
                dry_run=not args.apply,
            )
            report.update(result)
            if args.apply:
                report["total_rows"] = store.count_chunks()
    except IndexWriteLockBusy as exc:
        report["error"] = str(exc)
        print(json.dumps(report))
        return 3

    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
