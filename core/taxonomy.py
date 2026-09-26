"""Thin convenience layer for taxonomy store operations."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

from taxonomy_store import TaxonomyStore


logger = logging.getLogger(__name__)

_DATE_SEGMENT_RE = re.compile(
    r"^(\d{4}|\d{1,2}|\d{4}-\d{2}|\d{4}-\d{2}-\d{2})$"
)
_MONTH_NAMES = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}

_FOLDER_SYNC_STATE_VERSION = 1
_FOLDER_SYNC_STATE_NAME = "taxonomy_folder_sync.json"
_STALE_FOLDER_DESCRIPTION = "Folder path discovered from filesystem structure"


def load_taxonomy_store(config: dict) -> TaxonomyStore:
    """Build a TaxonomyStore from config, reusing the embed provider."""
    from providers.embed import build_embed_provider

    index_root = Path(config["index_root"])
    embed_provider = build_embed_provider(config)

    def embed_fn(text: str) -> list[float]:
        vectors = embed_provider.embed_texts([text])
        return vectors[0]

    return TaxonomyStore(index_root, table_name="taxonomy", embed_fn=embed_fn)


def _is_date_like_segment(segment: str) -> bool:
    s = segment.strip().lower()
    return s in _MONTH_NAMES or bool(_DATE_SEGMENT_RE.fullmatch(s))


def _folder_sync_state_path(store: TaxonomyStore) -> Path:
    return Path(store.index_root) / _FOLDER_SYNC_STATE_NAME


def _load_folder_sync_state(store: TaxonomyStore) -> dict[str, Any]:
    path = _folder_sync_state_path(store)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": _FOLDER_SYNC_STATE_VERSION, "roots": {}}
    if not isinstance(payload, dict):
        return {"version": _FOLDER_SYNC_STATE_VERSION, "roots": {}}
    roots = payload.get("roots")
    if not isinstance(roots, dict):
        roots = {}
    return {"version": _FOLDER_SYNC_STATE_VERSION, "roots": roots}


def _save_folder_sync_state(store: TaxonomyStore, state: dict[str, Any]) -> bool:
    """Persist sync fingerprints; False when the write failed."""
    path = _folder_sync_state_path(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    payload = {
        "version": _FOLDER_SYNC_STATE_VERSION,
        "roots": state.get("roots") or {},
    }
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        return True
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        logger.warning("Failed to save taxonomy folder sync state: %s", exc)
        return False


def _folder_set_hash(folder_paths: list[str]) -> str:
    payload = "\n".join(folder_paths)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _discover_folder_paths(root: Path, *, max_depth: int = 3) -> list[str]:
    """Return sorted unique relative folder paths kept by the sync rules."""
    if not root.exists():
        return []

    folder_paths: list[str] = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        rel = path.relative_to(root).as_posix()
        parts = rel.split("/")
        depth = len(parts)
        keep = False
        if depth <= min(max_depth, 2):
            keep = True
        elif depth == 3 and max_depth >= 3 and not _is_date_like_segment(parts[-1]):
            keep = True
        if keep:
            folder_paths.append(rel + "/")
    return sorted(set(folder_paths))


def sync_folder_taxonomy_from_filesystem(
    store: TaxonomyStore,
    root: str | Path,
    *,
    max_depth: int = 3,
) -> dict[str, int]:
    """Seed folder taxonomy entries from a real filesystem tree.

    Rules:
    - Keep depth 1 and 2 paths.
    - Keep depth 3 only when the leaf is not date-like.
    - Skip deeper paths to keep taxonomy prompt size bounded.

    When the discovered folder-name set matches the fingerprint from the last
    successful sync for this root, return immediately without per-entry Lance
    lookups. That keeps the common "nothing changed" indexer path nearly free.
    """
    root = Path(root).resolve()
    folder_paths = _discover_folder_paths(root, max_depth=max_depth)
    discovered = len(folder_paths)
    if discovered == 0:
        return {"discovered": 0, "added": 0, "existing": 0, "skipped": 0}

    digest = _folder_set_hash(folder_paths)
    state = _load_folder_sync_state(store)
    roots = state.setdefault("roots", {})
    root_key = str(root)
    previous = roots.get(root_key)
    if (
        isinstance(previous, dict)
        and previous.get("folder_set_hash") == digest
        and int(previous.get("discovered") or 0) == discovered
    ):
        return {
            "discovered": discovered,
            "added": 0,
            "existing": discovered,
            "skipped": 1,
        }

    # One list query instead of N get() round-trips (the 40s/run cost in prod).
    existing_by_id = {
        row["id"]: row for row in store.list_by_kind("folder") if row.get("id")
    }

    added = 0
    existing = 0
    for folder_path in folder_paths:
        entry_id = f"folder:{folder_path}"
        existing_entry = existing_by_id.get(entry_id)
        description = f"Filesystem folder path: {folder_path}"
        if existing_entry is not None:
            existing += 1
            if existing_entry.get("description") == _STALE_FOLDER_DESCRIPTION:
                store.update(entry_id, description=description)
            continue
        store.add(
            "folder",
            folder_path,
            description,
            contents_type="mixed",
            ai_managed=0,
            created_by="indexer",
        )
        added += 1

    roots[root_key] = {
        "folder_set_hash": digest,
        "discovered": discovered,
    }
    _save_folder_sync_state(store, state)

    return {
        "discovered": discovered,
        "added": added,
        "existing": existing,
        "skipped": 0,
    }


def sync_folder_taxonomy_from_sources(
    store: TaxonomyStore | None,
    sources: list[Any],
) -> dict[str, int]:
    """Seed folder taxonomy entries from filesystem-backed sources."""
    if store is None:
        return {
            "sources": 0,
            "discovered": 0,
            "added": 0,
            "existing": 0,
            "skipped": 0,
        }

    totals = {
        "sources": 0,
        "discovered": 0,
        "added": 0,
        "existing": 0,
        "skipped": 0,
    }
    for src in sources:
        root = getattr(src, "_root", None)
        if root is None:
            continue
        totals["sources"] += 1
        stats = sync_folder_taxonomy_from_filesystem(store, root)
        for key in ("discovered", "added", "existing", "skipped"):
            totals[key] += stats[key]
    return totals


def validate_tags(store: TaxonomyStore, tags: list[str]) -> tuple[list[str], list[str]]:
    """Check tags against taxonomy. Returns (known, unknown)."""
    known = []
    unknown = []
    for tag in tags:
        entry = store.get(f"tag:{tag}")
        if entry:
            known.append(tag)
        else:
            unknown.append(tag)
    return known, unknown


def suggest_folder(store: TaxonomyStore, folder: str) -> str | None:
    """Return canonical folder name if found in taxonomy, else None."""
    # Try exact match first
    entry = store.get(f"folder:{folder}")
    if entry:
        return entry["name"]
    # Try with trailing slash
    entry = store.get(f"folder:{folder}/")
    if entry:
        return entry["name"]
    return None
