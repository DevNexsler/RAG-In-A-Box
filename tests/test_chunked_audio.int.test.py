"""Real ffmpeg coverage, format, retry and cleanup without provider traffic."""
import hashlib
from pathlib import Path
import struct
import wave

import pytest

from core.resilience import TransientError
from providers.media.chunked_audio import ChunkedAudioProvider


@pytest.mark.parametrize("fail_second", [False, True])
def test_real_segmentation_covers_original_and_cleans_up(tmp_path, fail_second):
    original = tmp_path / "original.wav"
    with wave.open(str(original), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(22050)
        out.writeframes(struct.pack("<hh", 100, 200) * int(5.3 * 22050))
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    paths, durations = [], []

    class Provider:
        def transcribe_audio(self, path):
            paths.append(Path(path))
            with wave.open(str(path)) as clip:
                assert clip.getnchannels() == 1
                assert clip.getframerate() == 16000
                assert clip.getsampwidth() == 2
                durations.append(clip.getnframes() / clip.getframerate())
            if fail_second and len(paths) == 2:
                raise TransientError("injected segment failure")
            return f"words for part {len(paths)}"

    provider = ChunkedAudioProvider(Provider(), chunk_seconds=2, max_file_size_mb=80)
    if fail_second:
        with pytest.raises(TransientError):
            provider.transcribe_audio(original)
        assert len(paths) == 2
    else:
        text = provider.transcribe_audio(original)
        assert len(paths) == 3
        assert max(durations) <= 2.1
        assert abs(sum(durations) - 5.3) < .01
        assert "words for part 1" in text and "words for part 3" in text
    assert all(not path.exists() for path in paths)
    assert hashlib.sha256(original.read_bytes()).hexdigest() == digest
