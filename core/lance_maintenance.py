"""Production-shaped Lance maintenance commands for short-lived workers.

Large native operations run through this module as subprocesses so all Arrow,
Lance, and filesystem cache state owned by the worker is released on exit.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def compact_dataset(dataset_path: str) -> None:
    """Compact compatible fragments without decoding/re-encoding vectors."""
    import lance

    lance.dataset(dataset_path).optimize.compact_files(
        compaction_mode="try_binary_copy"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest="command", required=True)
    compact = subcommands.add_parser("compact")
    compact.add_argument("dataset_path")
    args = parser.parse_args(argv)

    if args.command == "compact":
        compact_dataset(args.dataset_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
