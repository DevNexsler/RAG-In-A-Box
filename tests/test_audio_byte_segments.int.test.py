"""Real byte-sized audio files, complete frame coverage, ordered merge and cleanup."""
import hashlib
from pathlib import Path
import struct
import subprocess
import wave

import pytest

from core.resilience import TransientError
from providers.media.chunked_audio import ChunkedAudioProvider


@pytest.mark.parametrize("compressed,fail_second", [(False, False), (True, False), (False, True)])
def test_byte_chunks_cover_recording_and_never_return_partial_text(tmp_path, compressed, fail_second):
    original = tmp_path / "original.wav"
    frames = b"".join(struct.pack("<h", (i // 100) % 10000) for i in range(84800))
    with wave.open(str(original), "wb") as clip:
        clip.setnchannels(1)
        clip.setsampwidth(2)
        clip.setframerate(16000)
        clip.writeframes(frames)
    if compressed:
        mp3 = tmp_path / "original.mp3"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(original),
                        "-c:a", "libmp3lame", "-b:a", "16k", str(mp3)], check=True)
        original = mp3
        assert original.stat().st_size < 65580
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    paths, payloads, sizes = [], [], []

    class Provider:
        def transcribe_audio(self, path):
            path = Path(path)
            paths.append(path)
            sizes.append(path.stat().st_size)
            assert sizes[-1] <= 65580
            with wave.open(str(path)) as part:
                assert part.getnchannels() == 1 and part.getsampwidth() == 2
                assert part.getframerate() == 16000
                payloads.append(part.readframes(part.getnframes()))
            if fail_second and len(paths) == 2:
                raise TransientError("injected second segment failure")
            return f"complete speech part {len(paths)}"

    provider = ChunkedAudioProvider(Provider(), chunk_size_mb=65580 / 1024 / 1024,
                                    max_file_size_mb=60)
    if fail_second:
        with pytest.raises(TransientError, match="second segment"):
            provider.transcribe_audio(original)
        assert len(paths) == 2
    else:
        text = provider.transcribe_audio(original)
        assert len(paths) == 3
        assert sizes[:2] == [65580, 65580]
        assert len(b"".join(payloads)) == len(frames)
        if not compressed:
            assert b"".join(payloads) == frames
        assert text.index("speech part 1") < text.index("speech part 2") < text.index("speech part 3")
        assert "Audio segment 3/3" in text
    assert paths and all(not path.exists() for path in paths)
    assert hashlib.sha256(original.read_bytes()).hexdigest() == digest


def test_part_budget_is_checked_before_decoding_or_upload(tmp_path, monkeypatch):
    path = tmp_path / "recording.mp3"
    path.write_bytes(b"compressed original")
    provider = ChunkedAudioProvider(object(), chunk_size_mb=65580 / 1024 / 1024,
                                    max_file_size_mb=60, max_parts=2)
    monkeypatch.setattr(provider, "_duration", lambda _: 10)
    monkeypatch.setattr(provider, "_split_by_size", lambda *_: pytest.fail("must reject before decoding"))
    with pytest.raises(TransientError, match="bounded segmentation budget"):
        provider.transcribe_audio(path)
