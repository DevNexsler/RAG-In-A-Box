"""Shared classifiers for generated files that are not standalone documents."""

from __future__ import annotations

import json
from pathlib import Path


_LEGACY_MESSAGE_KEYS = frozenset(
    {"message_id", "source_message_id", "source_record_id", "raw_event_id"}
)
_LEGACY_ATTACHMENT_KEYS = frozenset(
    {
        "provider_file_id",
        "filename",
        "original_filename",
        "attachment_name",
        "media_url",
        "source_media_url",
    }
)
_LEGACY_MEDIA_TYPE_KEYS = frozenset(
    {"mime", "mime_type", "media_type", "content_type"}
)
_LEGACY_RETRIEVAL_KEYS = frozenset(
    {"retrieval_status", "download_status", "retrieval_attempt"}
)


def is_communication_sidecar(path: Path, *, max_bytes: int = 2_000_000) -> bool:
    """Return true for attachment metadata consumed by communication context."""
    try:
        if path.stat().st_size > max_bytes:
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False

    modern = isinstance(payload.get("media"), dict) and any(
        key in payload for key in ("schema_version", "message", "counterparty", "channel")
    )
    if modern:
        return True

    # Older comm-review jobs emitted one flat attachment manifest beside each
    # binary.  These predate the nested ``message``/``media`` schema but carry
    # the same four independent signals: communication identity, attachment
    # identity, media type, and retrieval lifecycle.  Require all four so an
    # ordinary JSON business record with a filename or message_id is not
    # mistaken for generated sidecar metadata.
    keys = payload.keys()
    return (
        not _LEGACY_MESSAGE_KEYS.isdisjoint(keys)
        and not _LEGACY_ATTACHMENT_KEYS.isdisjoint(keys)
        and not _LEGACY_MEDIA_TYPE_KEYS.isdisjoint(keys)
        and not _LEGACY_RETRIEVAL_KEYS.isdisjoint(keys)
    )
