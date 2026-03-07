"""Seed taxonomy table from existing SQLite databases.

Reads tags.db and directory.db from ~/Documents/Primary/0-AI/directory_info/
and populates the LanceDB taxonomy table.

Idempotent: skips entries that already exist (upsert by id).

Usage:
    .venv/bin/python scripts/seed_taxonomy.py
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_config
from core.taxonomy import load_taxonomy_store


def _read_tags_db(db_path: str) -> list[dict]:
    """Read tags from tags.db SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM tags").fetchall()
    conn.close()

    entries = []
    for row in rows:
        entries.append({
            "kind": "tag",
            "name": row["name"],
            "description": row["description"] or "",
            "status": row["status"] or "active",
            "usage_count": row["usage_count"] or 0,
            "created_by": row["created_by"] or "AI",
            "ai_managed": 1,
            "aliases": "",
            "parent": "",
            "contents_type": "",
        })
    return entries


def _read_directory_db(db_path: str) -> list[dict]:
    """Read directories from directory.db SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM directories").fetchall()
    conn.close()

    entries = []
    for row in rows:
        # Combine description + purpose for richer embedding text
        desc = row["description"] or ""
        purpose = row["purpose"] or ""
        if purpose and purpose not in desc:
            desc = f"{desc}. {purpose}" if desc else purpose

        entries.append({
            "kind": "folder",
            "name": row["path"],
            "description": desc,
            "status": row["status"] or "active",
            "usage_count": 0,
            "created_by": row["created_by"] or "AI",
            "ai_managed": row["ai_managed"] if row["ai_managed"] is not None else 1,
            "contents_type": row["contents_type"] or "",
            "aliases": "",
            "parent": "",
        })
    return entries


def seed_taxonomy(config_path: str = "config.yaml") -> dict:
    """Import taxonomy data from SQLite databases. Returns import stats."""
    base_dir = os.path.expanduser("~/Documents/Primary/0-AI/directory_info")
    tags_db = os.path.join(base_dir, "tags.db")
    dir_db = os.path.join(base_dir, "directory.db")

    if not os.path.exists(tags_db):
        print(f"WARNING: tags.db not found at {tags_db}")
        tags_data = []
    else:
        tags_data = _read_tags_db(tags_db)
        print(f"Read {len(tags_data)} tags from tags.db")

    if not os.path.exists(dir_db):
        print(f"WARNING: directory.db not found at {dir_db}")
        dir_data = []
    else:
        dir_data = _read_directory_db(dir_db)
        print(f"Read {len(dir_data)} directories from directory.db")

    config = load_config(config_path)
    store = load_taxonomy_store(config)

    # Idempotent: skip existing entries
    added_tags = 0
    skipped_tags = 0
    for entry in tags_data:
        entry_id = f"tag:{entry['name']}"
        if store.get(entry_id) is not None:
            skipped_tags += 1
            continue
        store.add(**entry)
        added_tags += 1

    added_dirs = 0
    skipped_dirs = 0
    for entry in dir_data:
        entry_id = f"folder:{entry['name']}"
        if store.get(entry_id) is not None:
            skipped_dirs += 1
            continue
        store.add(**entry)
        added_dirs += 1

    # Build FTS index
    store.create_fts_index()

    stats = {
        "tags_added": added_tags,
        "tags_skipped": skipped_tags,
        "dirs_added": added_dirs,
        "dirs_skipped": skipped_dirs,
        "total_tags": store.count("tag"),
        "total_folders": store.count("folder"),
    }
    print(f"\nImport complete: {stats}")
    return stats


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    seed_taxonomy(config_path)
