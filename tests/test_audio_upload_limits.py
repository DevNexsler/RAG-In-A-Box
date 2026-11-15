"""Large originals are split before the per-request upload limit is enforced."""
import hashlib
from pathlib import Path
import struct
import wave

import pytest

from providers.media.chunked_audio import ChunkedAudioProvider
from providers.media.openrouter_media import MediaFileTooLargeError


@pytest.mark.parametrize("chunk_seconds", [2, 10])
def test_original_over_upload_limit_is_split_or_reencoded(tmp_path, chunk_seconds):
    original = tmp_path / "original.wav"
    with wave.open(str(original), "wb") as clip:
        clip.setnchannels(2)
        clip.setsampwidth(2)
        clip.setframerate(44100)
        clip.writeframes(struct.pack("<hh", 100, 200) * int(5.3 * 44100))
    before = hashlib.sha256(original.read_bytes()).hexdigest()
    max_mb = .2
    assert original.stat().st_size > max_mb * 1024 * 1024
    paths, durations = [], []

    class Provider:
        def transcribe_audio(self, path):
            paths.append(Path(path))
            assert Path(path).stat().st_size <= max_mb * 1024 * 1024
            with wave.open(str(path)) as clip:
                durations.append(clip.getnframes() / clip.getframerate())
            return f"words from part {len(paths)}"

    provider = ChunkedAudioProvider(Provider(), chunk_seconds=chunk_seconds,
                                    max_file_size_mb=max_mb)
    text = provider.transcribe_audio(original)
    assert len(paths) == (3 if chunk_seconds == 2 else 1)
    assert abs(sum(durations) - 5.3) < .01
    assert "words from part 1" in text
    assert all(not path.exists() for path in paths)
    assert hashlib.sha256(original.read_bytes()).hexdigest() == before


def test_oversized_derived_part_is_rejected_before_any_upload(tmp_path, monkeypatch):
    original = tmp_path / "original.wav"
    original.write_bytes(b"source")
    paths, calls = [], []

    class Provider:
        def transcribe_audio(self, path):
            calls.append(path)
            return "must not upload"

    provider = ChunkedAudioProvider(Provider(), chunk_seconds=2, max_file_size_mb=.001)
    monkeypatch.setattr(provider, "_duration", lambda _: 5)

    def split(command, *, timeout):
        for i, size in enumerate([10, 2048, 10]):
            path = Path(command[-1] % i)
            path.write_bytes(b"x" * size)
            paths.append(path)
        return ""

    monkeypatch.setattr(provider, "_run", split)
    with pytest.raises(MediaFileTooLargeError, match="segment"):
        provider.transcribe_audio(original)
    assert not calls
    assert all(not path.exists() for path in paths)
