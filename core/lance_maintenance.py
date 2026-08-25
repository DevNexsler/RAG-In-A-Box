"""Production-shaped Lance maintenance commands for short-lived workers.

Large native operations run through this module as subprocesses so all Arrow,
Lance, and filesystem cache state owned by the worker is released on exit.
"""

from __future__ import annotations

import argparse
import shutil
import time
from collections.abc import Sequence
from pathlib import Path


def compact_dataset(dataset_path: str) -> None:
    """Compact compatible fragments without decoding/re-encoding vectors."""
    import lance

    lance.dataset(dataset_path).optimize.compact_files(
        compaction_mode="try_binary_copy"
    )


def reachable_index_uuids(dataset_path: str) -> set[str]:
    """UUIDs of every index segment some retained dataset version still names.

    Delta indices are separate manifest entries under one index name, so the
    reachable set is collected per segment, not per index.
    """
    import lance

    dataset = lance.dataset(dataset_path)
    return {
        str(segment.uuid)
        for version in dataset.versions()
        for index in dataset.checkout_version(version["version"]).describe_indices()
        for segment in index.segments
    }


def prune_orphan_indices(
    dataset_path: str, *, min_age_seconds: float = 0.0
) -> tuple[int, int]:
    """Remove `_indices/<uuid>` directories no retained version can reach.

    Every index merge writes a whole new index generation and orphans the
    previous one, but `cleanup_old_versions` never deletes a file newer than
    the oldest version it retains — and a daily restore-point tag deliberately
    retains a days-old one. So each orphan generation is pinned for the whole
    restore-point window: production reached 5,774 directories holding 26 GiB
    against a 43 MiB live index (#1160). A directory outside the reachable set
    cannot be opened by any version the dataset still has, so removing it is
    the reclaim Lance itself would do without that floor.

    ``min_age_seconds`` protects an index build that has not committed its
    manifest yet — until it commits it is indistinguishable from an orphan.
    Callers pass the version-retention band, which is already sized to outlast
    the longest concurrent index operation.

    Returns ``(directories_removed, bytes_removed)``.
    """
    indices_root = Path(dataset_path) / "_indices"
    if not indices_root.is_dir():
        return (0, 0)

    reachable = reachable_index_uuids(dataset_path)
    cutoff = time.time() - max(0.0, min_age_seconds)
    removed = 0
    reclaimed = 0
    for entry in sorted(indices_root.iterdir()):
        if entry.name in reachable or not entry.is_dir():
            continue
        files = [path for path in entry.rglob("*") if path.is_file()]
        newest = max(path.stat().st_mtime for path in (entry, *files))
        if newest > cutoff:
            continue
        size = sum(path.stat().st_size for path in files)
        shutil.rmtree(entry)
        removed += 1
        reclaimed += size
    return (removed, reclaimed)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest="command", required=True)
    compact = subcommands.add_parser("compact")
    compact.add_argument("dataset_path")
    prune_indices = subcommands.add_parser("prune-indices")
    prune_indices.add_argument("dataset_path")
    prune_indices.add_argument(
        "--min-age-minutes",
        type=float,
        default=0.0,
        help="keep index directories written within this many minutes",
    )
    args = parser.parse_args(argv)

    if args.command == "compact":
        compact_dataset(args.dataset_path)
    elif args.command == "prune-indices":
        removed, reclaimed = prune_orphan_indices(
            args.dataset_path,
            min_age_seconds=args.min_age_minutes * 60,
        )
        print(f"removed {removed} orphan index directories ({reclaimed} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
