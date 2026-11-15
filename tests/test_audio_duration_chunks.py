import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from core.resilience import TransientError
from providers.media import build_media_provider
from providers.media.chunked_audio import ChunkedAudioProvider


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan")])
def test_invalid_chunk_duration_is_rejected(seconds):
    with pytest.raises(ValueError):
        ChunkedAudioProvider(Mock(), chunk_seconds=seconds, max_file_size_mb=80)


def test_short_audio_and_video_keep_existing_routing(tmp_path, monkeypatch):
    path = tmp_path / "clip.wav"
    path.write_bytes(b"voice")
    inner = Mock()
    inner.transcribe_audio.return_value = "original transcript"
    inner.analyze_video.return_value = "original video description"
    provider = ChunkedAudioProvider(inner, chunk_seconds=540, max_file_size_mb=80)
    monkeypatch.setattr(provider, "_duration", lambda _: 12)
    assert provider.transcribe_audio(path) == "original transcript"
    inner.transcribe_audio.assert_called_once_with(path.resolve())
    assert provider.analyze_video("video.mp4") == "original video description"
    inner.analyze_video.assert_called_once_with("video.mp4")


def _fake_tools(provider, monkeypatch, *, durations=(2, 2, 1)):
    created = []

    def run(command, *, timeout):
        assert timeout > 0
        if command[0] == "ffprobe":
            path = Path(command[-1])
            seconds = (durations[int(path.stem.split("-")[-1])]
                       if path.name.startswith("part-") else 5)
            return json.dumps({"format": {"duration": seconds}})
        assert command[0] == "ffmpeg"
        assert "-nostdin" in command and "-vn" in command
        pattern = command[-1]
        for i in range(len(durations)):
            path = Path(pattern % i)
            path.write_bytes(b"segment")
            created.append(path)
        return ""

    monkeypatch.setattr(provider, "_run", run)
    return created


def test_all_segments_are_ordered_and_cleaned(tmp_path, monkeypatch):
    path = tmp_path / "original.wav"
    path.write_bytes(b"original")
    inner = Mock()
    inner.transcribe_audio.side_effect = ["first words", "middle words", "last words"]
    provider = ChunkedAudioProvider(inner, chunk_seconds=2, max_file_size_mb=80)
    created = _fake_tools(provider, monkeypatch)
    text = provider.transcribe_audio(path)
    assert text.index("first words") < text.index("middle words") < text.index("last words")
    assert "00:00:00–00:00:02" in text
    assert "00:00:04–00:00:05" in text
    assert "Speaker labels are local to each segment" in text
    assert inner.transcribe_audio.call_count == 3
    assert all(not p.exists() for p in created)
    assert path.read_bytes() == b"original"


@pytest.mark.parametrize("failure", [TransientError("provider unavailable"), ""])
def test_failed_or_empty_segment_never_returns_partial_text(tmp_path, monkeypatch, failure):
    path = tmp_path / "original.wav"
    path.write_bytes(b"original")
    inner = Mock()
    inner.transcribe_audio.side_effect = ["first words", failure, "last words"]
    provider = ChunkedAudioProvider(inner, chunk_seconds=2, max_file_size_mb=80)
    created = _fake_tools(provider, monkeypatch)
    with pytest.raises(TransientError):
        provider.transcribe_audio(path)
    assert inner.transcribe_audio.call_count == 2
    assert all(not p.exists() for p in created)
    assert path.read_bytes() == b"original"


def test_missing_tail_is_rejected_before_any_upload(tmp_path, monkeypatch):
    path = tmp_path / "original.wav"
    path.write_bytes(b"original")
    inner = Mock()
    provider = ChunkedAudioProvider(inner, chunk_seconds=2, max_file_size_mb=80)
    created = _fake_tools(provider, monkeypatch, durations=(2,))
    with pytest.raises(TransientError, match="complete recording"):
        provider.transcribe_audio(path)
    inner.transcribe_audio.assert_not_called()
    assert all(not p.exists() for p in created)


def test_duration_tool_failure_is_retryable(tmp_path, monkeypatch):
    path = tmp_path / "original.wav"
    path.write_bytes(b"original")
    provider = ChunkedAudioProvider(Mock(), chunk_seconds=2, max_file_size_mb=80)
    monkeypatch.setattr("subprocess.run", Mock(side_effect=FileNotFoundError("ffprobe")))
    with pytest.raises(TransientError):
        provider.transcribe_audio(path)


def test_factory_duration_chunking_is_opt_in(monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "qualification")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config = {"media": {"enabled": True, "provider": "litellm"}}
    assert not isinstance(build_media_provider(config), ChunkedAudioProvider)
    config["media"]["audio_chunk_seconds"] = 540
    wrapped = build_media_provider(config)
    assert isinstance(wrapped, ChunkedAudioProvider)
    assert wrapped.chunk_seconds == 540
