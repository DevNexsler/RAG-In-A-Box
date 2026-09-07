#!/usr/bin/env python3
"""Build the ANN index on the chunks table's vector column outside the index run.

The index run builds this index itself (index_vault_flow -> ensure_vector_index),
but on the 88k-row production table the build peaks at ~3 GB RSS, which does not
fit beside the live server inside the container's 8 GiB memory cgroup — the
kernel would kill the server, not the builder (#1254). Run the one-time build
from a sibling container that mounts the same index volume and this repo:

    docker run --rm --memory=6g \\
        -v rag-in-a-box_doc-organizer-data:/data/index \\
        -v "$PWD":/app -w /app doc-organizer:latest \\
        python scripts/ensure_vector_index.py --config config.yaml

It takes the same table-level writer lock the sweep takes. The lock file lives
under the index root, so it serializes across containers: a running index run
finishes first, and no writer interleaves with the build. Every later index run
finds the index present and only merges new rows into it.

Prints one JSON line. ``--check`` reports without building; ``--no-wait`` exits 3
instead of waiting for a running writer.
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
from lancedb_store import LanceDBStore, vector_index_settings_from_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=Path("/app/config.yaml"))
    parser.add_argument("--index-root", type=Path, default=None, help="override config index_root")
    parser.add_argument("--table", default=None, help="override config lancedb.table")
    parser.add_argument("--check", action="store_true", help="report only; build nothing")
    parser.add_argument(
        "--no-wait", action="store_true", help="exit 3 if another index writer holds the table"
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    index_root = Path(args.index_root or config["index_root"])
    table_name = args.table or config.get("lancedb", {}).get("table", "chunks")
    lance_session.configure_from_config(config)

    report: dict = {"index_root": str(index_root), "table": table_name, "created": False}
    try:
        with index_write_lock(index_root, table_name, blocking=not args.no_wait):
            store = LanceDBStore(index_root, table_name)
            if not args.check:
                report["created"] = store.ensure_vector_index(
                    **vector_index_settings_from_config(config)
                )
            report["rows"] = store.count_chunks()
            report["vector_index_available"] = store.vector_index_available()
    except IndexWriteLockBusy as exc:
        report["error"] = str(exc)
        print(json.dumps(report))
        return 3
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
