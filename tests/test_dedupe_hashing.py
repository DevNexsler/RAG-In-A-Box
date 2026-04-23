import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from core.dedupe import ExactIdentity, compute_file_identity, compute_text_identity, is_zero_payload


def test_compute_file_identity_returns_blake3_identity(tmp_path):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"abc123")

    result = compute_file_identity(path)

    assert isinstance(result, ExactIdentity)
    assert result.size_bytes == 6
    assert result.hash_algo == "blake3"
    assert isinstance(result.content_hash, bytes)
    assert len(result.content_hash) == 32
    assert result.content_hash.hex() == "00307ced6a8b278d5e3a9f77b138d0e9d2209717c9d45b205f427a73565cc5fb"
    assert result.is_empty is False


def test_compute_file_identity_flags_zero_byte_file(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_bytes(b"")

    result = compute_file_identity(path)

    assert result.size_bytes == 0
    assert result.is_empty is True


def test_compute_file_identity_streams_across_multiple_chunks(tmp_path):
    path = tmp_path / "chunked.bin"
    path.write_bytes(b"abc123")

    result = compute_file_identity(path, chunk_size=2)

    assert result.size_bytes == 6
    assert result.content_hash.hex() == "00307ced6a8b278d5e3a9f77b138d0e9d2209717c9d45b205f427a73565cc5fb"


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_compute_file_identity_rejects_nonpositive_chunk_size(tmp_path, chunk_size):
    path = tmp_path / "chunked.bin"
    path.write_bytes(b"abc123")

    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        compute_file_identity(path, chunk_size=chunk_size)


def test_compute_text_identity_flags_empty_payload():
    result = compute_text_identity("")

    assert isinstance(result, ExactIdentity)
    assert result.size_bytes == 0
    assert result.hash_algo == "blake3"
    assert len(result.content_hash) == 32
    assert result.is_empty is True


def test_compute_text_identity_streams_text_chunks():
    result = compute_text_identity("abc123", chunk_size=2)

    assert result.size_bytes == 6
    assert result.content_hash.hex() == "00307ced6a8b278d5e3a9f77b138d0e9d2209717c9d45b205f427a73565cc5fb"
    assert result.is_empty is False


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_compute_text_identity_rejects_nonpositive_chunk_size(chunk_size):
    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        compute_text_identity("abc123", chunk_size=chunk_size)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (b"", True),
        ("", True),
        (ExactIdentity(size_bytes=0, content_hash=b"\x00" * 32, is_empty=True), True),
        (ExactIdentity(size_bytes=1, content_hash=b"\x00" * 32, is_empty=False), False),
        (b"x", False),
        ("x", False),
    ],
)
def test_is_zero_payload(value, expected):
    assert is_zero_payload(value) is expected
