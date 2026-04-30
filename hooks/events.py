"""Generic event payload builders for downstream hooks."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

INTERNAL_METADATA_KEYS = {
    "_node_content",
    "_node_type",
    "document_id",
    "ref_doc_id",
}


def public_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return metadata safe for outbound hook payloads."""
    return {
        key: value
        for key, value in metadata.items()
        if key not in INTERNAL_METADATA_KEYS and value not in (None, "")
    }


def build_document_indexed_event(
    *,
    doc_id: str,
    source_name: str,
    source_type: str,
    rel_path: str,
    abs_path: str,
    text: str,
    metadata: dict[str, Any],
    chunks: list[dict[str, Any]],
    occurred_at: str | None = None,
) -> dict[str, Any]:
    """Build a serializable document.indexed event payload."""
    return {
        "event": "document.indexed",
        "version": 1,
        "occurred_at": occurred_at or datetime.now(timezone.utc).isoformat(),
        "doc_id": doc_id,
        "source_name": source_name,
        "source_type": source_type,
        "rel_path": rel_path,
        "abs_path": abs_path,
        "metadata": public_metadata(metadata),
        "text": text,
        "chunks": chunks,
    }
