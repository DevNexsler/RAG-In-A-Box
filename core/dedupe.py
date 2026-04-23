"""Helpers for exact duplicate hashing and archival."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

from blake3 import blake3

_MAX_ARCHIVE_NAME_LENGTH = 180
_ARCHIVE_DIGEST_LENGTH = 16
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


@dataclass(frozen=True)
class ExactIdentity:
    size_bytes: int
    content_hash: bytes
    hash_algo: str = "blake3"
    is_empty: bool = False


def is_zero_payload(payload: str | bytes | ExactIdentity) -> bool:
    """Return whether a payload or identity is empty."""
    if isinstance(payload, ExactIdentity):
        return payload.is_empty
    if isinstance(payload, str):
        return len(payload.encode("utf-8")) == 0
    return len(payload) == 0


def _validate_chunk_size(chunk_size: int) -> None:
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")


def compute_file_identity(path: str | Path, chunk_size: int = 1024 * 1024) -> ExactIdentity:
    """Return exact hash identity for a file using streaming BLAKE3."""
    _validate_chunk_size(chunk_size)
    file_path = Path(path)
    hasher = blake3()
    size_bytes = 0

    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            size_bytes += len(chunk)
            hasher.update(chunk)

    return ExactIdentity(
        size_bytes=size_bytes,
        content_hash=hasher.digest(),
        is_empty=size_bytes == 0,
    )


def compute_text_identity(text: str, chunk_size: int = 1024 * 1024) -> ExactIdentity:
    """Return exact hash identity for UTF-8 text payload."""
    _validate_chunk_size(chunk_size)
    hasher = blake3()
    size_bytes = 0
    stream = BytesIO(text.encode("utf-8"))
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        size_bytes += len(chunk)
        hasher.update(chunk)

    return ExactIdentity(
        size_bytes=size_bytes,
        content_hash=hasher.digest(),
        is_empty=size_bytes == 0,
    )


def _timestamp_token() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _archive_root_path(archive_root: str | Path) -> Path:
    return Path(archive_root).expanduser().resolve(strict=False)


def _ensure_within_archive_root(archive_root: Path, candidate: Path) -> Path:
    resolved_candidate = candidate.resolve(strict=False)
    if resolved_candidate != archive_root and archive_root not in resolved_candidate.parents:
        raise ValueError(f"archive path escapes archive_root: {candidate}")
    return candidate


def _require_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if value == "":
        raise ValueError(f"{field_name} must not be empty")
    return value


def _encode_archive_segment(value: str, field_name: str) -> str:
    encoded = quote(_require_non_empty_string(value, field_name), safe="-_")
    encoded = encoded.replace(".", "%2E")
    return _prefix_windows_reserved_basename(encoded)


def _prefix_windows_reserved_basename(value: str) -> str:
    if value.rstrip(" .").upper() in _WINDOWS_RESERVED_NAMES:
        return f"seg__{value}"
    return value


def _stable_archive_digest(value: str) -> str:
    return blake3(value.encode("utf-8")).hexdigest()[:_ARCHIVE_DIGEST_LENGTH]


def _bounded_archive_name(
    original_value: str,
    *,
    prefix: str = "",
    suffix: str = "",
    preserve_suffix: str = "",
) -> str:
    encoded = quote(original_value, safe="")
    full_name = f"{prefix}{encoded}{suffix}"
    if len(full_name) <= _MAX_ARCHIVE_NAME_LENGTH:
        return _make_windows_safe_archive_name(full_name)

    encoded_preserve_suffix = quote(preserve_suffix, safe="")
    encoded_prefix = encoded
    if encoded_preserve_suffix and encoded.endswith(encoded_preserve_suffix):
        encoded_prefix = encoded[: -len(encoded_preserve_suffix)]

    digest_suffix = f"__{_stable_archive_digest(original_value)}{encoded_preserve_suffix}{suffix}"
    available = _MAX_ARCHIVE_NAME_LENGTH - len(prefix) - len(digest_suffix)
    if available <= 0:
        raise ValueError("archive filename budget exhausted")

    truncated_prefix = encoded_prefix[:available].rstrip(".")
    if not truncated_prefix:
        truncated_prefix = "item"
    return _make_windows_safe_archive_name(f"{prefix}{truncated_prefix}{digest_suffix}")


def _make_windows_safe_archive_name(name: str) -> str:
    stem, dot, suffix = name.rpartition(".")
    if not dot:
        stem = name
        suffix = ""
    else:
        suffix = f"{dot}{suffix}"
    return f"{_prefix_windows_reserved_basename(stem)}{suffix}"


def _metadata_sidecar_path(archived_path: Path) -> Path:
    return archived_path.parent / f"{archived_path.name}.metadata.json"


def _write_text_atomic(target_path: Path, content: str) -> None:
    fd, tmp_name = tempfile.mkstemp(
        dir=target_path.parent,
        prefix=f".{target_path.name}.",
        suffix=".tmp",
        text=True,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_path, target_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _reserve_unique_record_snapshot_path(
    archive_root: Path,
    archive_dir: Path,
    natural_key: str,
) -> tuple[Path, int]:
    open_flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_BINARY"):
        open_flags |= os.O_BINARY

    snapshot_name = _bounded_archive_name(natural_key, suffix=".json")
    candidates = [snapshot_name]

    for _ in range(32):
        candidates.append(
            _bounded_archive_name(
                natural_key,
                suffix=f"__{_timestamp_token()}.json",
            )
        )

    for candidate_name in candidates:
        snapshot_path = _ensure_within_archive_root(archive_root, archive_dir / candidate_name)
        try:
            fd = os.open(snapshot_path, open_flags)
        except FileExistsError:
            continue
        return snapshot_path, fd

    raise FileExistsError(
        f"could not allocate unique archive snapshot for natural_key={natural_key!r}"
    )


def archive_duplicate_file(
    archive_root: str | Path,
    source_name: str,
    canonical_doc_id: str,
    source_path: str | Path,
    rel_path: str,
) -> Path:
    """Copy duplicate file into archive without touching original source."""
    archive_root_path = _archive_root_path(archive_root)
    canonical_segment = _encode_archive_segment(canonical_doc_id, "canonical_doc_id")
    _encode_archive_segment(source_name, "source_name")
    rel_path = _require_non_empty_string(rel_path, "rel_path")
    archive_dir = _ensure_within_archive_root(
        archive_root_path,
        archive_root_path / "filesystem" / canonical_segment,
    )
    archive_dir.mkdir(parents=True, exist_ok=True)

    archived_name_prefix = f"{_timestamp_token()}__"
    archived_name = _bounded_archive_name(
        rel_path,
        prefix=archived_name_prefix,
        preserve_suffix=Path(rel_path).suffix,
    )
    while True:
        archived_path = _ensure_within_archive_root(archive_root_path, archive_dir / archived_name)
        if not archived_path.exists():
            break
        archived_name_prefix = f"{_timestamp_token()}__"
        archived_name = _bounded_archive_name(
            rel_path,
            prefix=archived_name_prefix,
            preserve_suffix=Path(rel_path).suffix,
        )

    shutil.copy2(Path(source_path), archived_path)

    metadata_path = _ensure_within_archive_root(archive_root_path, _metadata_sidecar_path(archived_path))
    try:
        _write_text_atomic(
            metadata_path,
            json.dumps(
                {
                    "source_name": source_name,
                    "canonical_doc_id": canonical_doc_id,
                    "original_rel_path": rel_path,
                    "archived_path": str(archived_path),
                },
                sort_keys=True,
                indent=2,
            )
        )
    except Exception:
        archived_path.unlink(missing_ok=True)
        raise
    return archived_path


def archive_duplicate_record(
    archive_root: str | Path,
    source_name: str,
    canonical_doc_id: str,
    natural_key: str,
    record: dict,
) -> Path:
    """Write duplicate record snapshot to archive JSON."""
    archive_root_path = _archive_root_path(archive_root)
    source_segment = _encode_archive_segment(source_name, "source_name")
    canonical_segment = _encode_archive_segment(canonical_doc_id, "canonical_doc_id")
    natural_key = _require_non_empty_string(natural_key, "natural_key")
    archive_dir = _ensure_within_archive_root(
        archive_root_path,
        archive_root_path / "postgres" / source_segment / canonical_segment,
    )
    archive_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path, snapshot_fd = _reserve_unique_record_snapshot_path(
        archive_root_path,
        archive_dir,
        natural_key,
    )
    payload = dict(record)
    payload["source_name"] = source_name
    payload["natural_key"] = natural_key
    payload["canonical_doc_id"] = canonical_doc_id
    try:
        with os.fdopen(snapshot_fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str, indent=2))
    except Exception:
        snapshot_path.unlink(missing_ok=True)
        raise
    return snapshot_path


# Backward-compatible aliases for earlier Task 2 draft API.
hash_file_exact = compute_file_identity
hash_text_exact = compute_text_identity
archive_filesystem_duplicate = archive_duplicate_file
archive_postgres_duplicate = archive_duplicate_record
