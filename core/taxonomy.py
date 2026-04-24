"""Thin convenience layer for taxonomy store operations."""

from __future__ import annotations

import re
from typing import Any
from pathlib import Path

from taxonomy_store import TaxonomyStore


_DATE_SEGMENT_RE = re.compile(
    r"^(\d{4}|\d{1,2}|\d{4}-\d{2}|\d{4}-\d{2}-\d{2})$"
)
_MONTH_NAMES = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}


def load_taxonomy_store(config: dict) -> TaxonomyStore:
    """Build a TaxonomyStore from config, reusing the embed provider."""
    from pathlib import Path
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
    """
    root = Path(root)
    if not root.exists():
        return {"discovered": 0, "added": 0, "existing": 0}

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

    added = 0
    existing = 0
    for folder_path in sorted(set(folder_paths)):
        entry_id = f"folder:{folder_path}"
        if store.get(entry_id) is not None:
            existing += 1
            continue
        store.add(
            "folder",
            folder_path,
            "Folder path discovered from filesystem structure",
            contents_type="mixed",
            ai_managed=0,
            created_by="indexer",
        )
        added += 1

    return {
        "discovered": len(set(folder_paths)),
        "added": added,
        "existing": existing,
    }


def sync_folder_taxonomy_from_sources(
    store: TaxonomyStore | None,
    sources: list[Any],
) -> dict[str, int]:
    """Seed folder taxonomy entries from filesystem-backed sources."""
    if store is None:
        return {"sources": 0, "discovered": 0, "added": 0, "existing": 0}

    totals = {"sources": 0, "discovered": 0, "added": 0, "existing": 0}
    for src in sources:
        root = getattr(src, "_root", None)
        if root is None:
            continue
        totals["sources"] += 1
        stats = sync_folder_taxonomy_from_filesystem(store, root)
        for key in ("discovered", "added", "existing"):
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
