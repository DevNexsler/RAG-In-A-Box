"""Thin convenience layer for taxonomy store operations."""

from __future__ import annotations

from typing import Any

from taxonomy_store import TaxonomyStore


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
