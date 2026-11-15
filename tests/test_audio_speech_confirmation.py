from unittest.mock import Mock

import pytest

from core.resilience import TransientError
from providers.media.chunked_audio import ChunkedAudioProvider


@pytest.mark.parametrize("marker", ["[No intelligible speech.]", "NO SPEECH DETECTED", "[No intelligible speech]"])
def test_no_speech_is_confirmed_by_configured_alternate(tmp_path, monkeypatch, marker):
    path = tmp_path / "original.wav"
    path.write_bytes(b"original")
    inner = Mock()
    inner.transcribe_audio.side_effect = [marker, "second words"]
    fallback = Mock(return_value="speech recovered from the first segment")
    provider = ChunkedAudioProvider(inner, chunk_seconds=2, max_file_size_mb=80,
                                    no_speech_fallback=fallback)
    _parts(provider, monkeypatch)
    text = provider.transcribe_audio(path)
    assert "speech recovered from the first segment" in text
    assert text.index("first segment") < text.index("second words")
    fallback.assert_called_once()
    assert not fallback.call_args.args[0].exists()


def _parts(provider, monkeypatch):
    def run(command, *, timeout):
        from pathlib import Path
        if command[0] == "ffmpeg":
            for i in range(2):
                Path(command[-1] % i).write_bytes(b"segment")
        return ""
    monkeypatch.setattr(provider, "_run", run)
    monkeypatch.setattr(provider, "_duration", lambda p: 2 if p.name.startswith("part-") else 4)


@pytest.mark.parametrize("confirmation", ["", TransientError("alternate unavailable"), None])
def test_unverified_no_speech_retains_retry_and_no_partial_output(tmp_path, monkeypatch, confirmation):
    path = tmp_path / "original.wav"
    path.write_bytes(b"original")
    inner = Mock()
    inner.transcribe_audio.side_effect = ["first words", "[No intelligible speech]"]
    fallback = None if confirmation is None else Mock(side_effect=[confirmation])
    provider = ChunkedAudioProvider(inner, chunk_seconds=2, max_file_size_mb=80,
                                    no_speech_fallback=fallback)
    _parts(provider, monkeypatch)
    with pytest.raises(TransientError):
        provider.transcribe_audio(path)
    assert path.read_bytes() == b"original"
    assert all(not call.args[0].exists() for call in inner.transcribe_audio.call_args_list)


def test_two_explicit_no_speech_responses_preserve_silent_segment(tmp_path, monkeypatch):
    path = tmp_path / "original.wav"
    path.write_bytes(b"original")
    inner = Mock()
    inner.transcribe_audio.side_effect = ["[No intelligible speech]", "spoken words"]
    provider = ChunkedAudioProvider(inner, chunk_seconds=2, max_file_size_mb=80,
                                    no_speech_fallback=Mock(return_value="[No intelligible speech]"))
    _parts(provider, monkeypatch)
    text = provider.transcribe_audio(path)
    assert "[No intelligible speech]" in text and "spoken words" in text


def test_negated_phrase_inside_transcript_does_not_trigger_confirmation(tmp_path, monkeypatch):
    path = tmp_path / "original.wav"
    path.write_bytes(b"original")
    inner = Mock()
    inner.transcribe_audio.side_effect = ["He said no intelligible speech was recorded.", "second words"]
    fallback = Mock(side_effect=AssertionError("not a sentinel"))
    provider = ChunkedAudioProvider(inner, chunk_seconds=2, max_file_size_mb=80,
                                    no_speech_fallback=fallback)
    _parts(provider, monkeypatch)
    assert "He said no intelligible speech was recorded." in provider.transcribe_audio(path)
    fallback.assert_not_called()
