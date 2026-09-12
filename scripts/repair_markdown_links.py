#!/usr/bin/env python3
"""Repair Markdown links broken by Doc ID filename injection.

This lightweight entry point is suitable for a host monitor when the live
Document Organizer container intentionally mounts the vault read-only.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from markdown_link_repair import build_doc_id_aliases, repair_markdown_links  # noqa: E402


def registered_paths(registry_path: Path) -> list[str]:
    """Read registry paths without initializing or migrating the live DB."""
    connection = sqlite3.connect(f"file:{registry_path}?mode=ro", uri=True)
    try:
        return [
            row[0]
            for row in connection.execute("SELECT rel_path FROM doc_registry")
        ]
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()

    aliases = build_doc_id_aliases(
        args.vault_root,
        registered_paths(args.registry),
    )
    result = repair_markdown_links(
        args.vault_root,
        aliases,
        exclude=args.exclude,
    )
    print(json.dumps({
        "status": "ok",
        "aliases": len(aliases),
        "files_changed": result.files_changed,
        "links_rewritten": result.links_rewritten,
        "changed_paths": list(result.changed_paths),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
