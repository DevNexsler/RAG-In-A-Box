"""Canonical source type helpers shared by indexer and APIs."""

import re

SOURCE_TYPE_BY_EXTENSION = {
    "md": "md",
    "pdf": "pdf",
    "docx": "doc",
    "doc": "doc",
    "rtf": "doc",
    "xlsx": "sheet",
    "xls": "sheet",
    "pptx": "pres",
    "csv": "csv",
    "html": "html",
    "htm": "html",
    "epub": "epub",
    "txt": "txt",
    "png": "img",
    "jpg": "img",
    "jpeg": "img",
    "gif": "img",
    "webp": "img",
}

BUILTIN_SOURCE_TYPES = set(SOURCE_TYPE_BY_EXTENSION.values()) | {"other"}
SAFE_SOURCE_TYPE_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def canonical_source_type(value: str | None) -> str:
    """Map filesystem suffixes to canonical search source_type values."""
    if not value:
        return "other"
    normalized = value.lower().lstrip(".")
    return SOURCE_TYPE_BY_EXTENSION.get(normalized, normalized or "other")


def is_safe_source_type(value: str | None) -> bool:
    """Return True for source_type strings safe to use as filter values."""
    return bool(value and SAFE_SOURCE_TYPE_RE.match(value))
