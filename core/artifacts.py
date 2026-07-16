"""Shared classifiers for generated files that are not standalone documents."""

from __future__ import annotations

import json
from pathlib import Path


def is_communication_sidecar(path: Path, *, max_bytes: int = 2_000_000) -> bool:
    """Return true for attachment metadata consumed by communication context."""
    try:
        if path.stat().st_size > max_bytes:
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(payload, dict) or not isinstance(payload.get("media"), dict):
        return False
    return any(
        key in payload
        for key in ("schema_version", "message", "counterparty", "channel")
    )
