"""Local-only audio routing: every request stays on LiteLLM, never OpenRouter."""

from unittest.mock import Mock

import httpx
import pytest

from core.resilience import TransientError
from providers.media import build_media_provider
from providers.media.chunked_audio import ChunkedAudioProvider

pytestmark = pytest.mark.unit


def _config(**media):
    return {"media": {"enabled": True, "provider": "litellm", "endpoint": "http://litellm/v1",
                      "audio_model": "audio", "max_file_size_mb": 60, **media}}


def test_local_route_builds_no_openrouter_audio_path(monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    provider = build_media_provider(_config(audio_transcription_route="local",
                                            audio_chunk_seconds=60))
    assert isinstance(provider, ChunkedAudioProvider)
    assert provider.chunk_seconds == 60
    assert provider.large_audio_provider is None
    assert provider.no_speech_fallback is None
    assert provider.accept_unconfirmed_silence
    audio = provider.provider._primary._audio_run.__self__
    assert audio.require_complete and audio.model == "audio"
    assert "Transcribe all speech" in audio.prompt


def test_local_route_rejects_an_openrouter_large_provider(monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    with pytest.raises(ValueError, match="conflicts"):
        build_media_provider(_config(audio_transcription_route="local", audio_chunk_size_mb=55,
                                     audio_large_provider="openrouter"))


def test_unknown_route_is_rejected(monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "test")
    with pytest.raises(ValueError, match="mixed or local"):
        build_media_provider(_config(audio_transcription_route="cloud", audio_chunk_seconds=60))


def test_mixed_route_is_unchanged_by_default(monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    provider = build_media_provider(_config(audio_chunk_seconds=60))
    assert provider.no_speech_fallback is not None
    assert not provider.accept_unconfirmed_silence


def _parts(provider, monkeypatch, count=3):
    def run(command, *, timeout):
        from pathlib import Path
        if command[0] == "ffmpeg":
            for i in range(count):
                Path(command[-1] % i).write_bytes(b"segment")
        return ""
    monkeypatch.setattr(provider, "_run", run)
    monkeypatch.setattr(provider, "_duration",
                        lambda p: 60 if p.name.startswith("part-") else 60 * count)


def test_local_blank_and_no_speech_segments_are_silence_without_cloud(tmp_path, monkeypatch):
    path = tmp_path / "original.ogg"
    path.write_bytes(b"original")
    inner = Mock()
    inner.transcribe_audio.side_effect = ["first words", "", "[No intelligible speech]"]
    provider = ChunkedAudioProvider(inner, chunk_seconds=60, max_file_size_mb=60,
                                    accept_unconfirmed_silence=True)
    _parts(provider, monkeypatch)
    monkeypatch.setattr(httpx, "post", Mock(side_effect=AssertionError("no network")))
    text = provider.transcribe_audio(path)
    assert "first words" in text
    assert text.count("[No speech detected]") == 2


def test_local_failures_still_fail_the_recording(tmp_path, monkeypatch):
    path = tmp_path / "original.ogg"
    path.write_bytes(b"original")
    inner = Mock()
    inner.transcribe_audio.side_effect = ["first words", TransientError("audio backend down")]
    provider = ChunkedAudioProvider(inner, chunk_seconds=60, max_file_size_mb=60,
                                    accept_unconfirmed_silence=True)
    _parts(provider, monkeypatch)
    with pytest.raises(TransientError, match="backend down"):
        provider.transcribe_audio(path)


def test_looping_local_part_is_marked_not_failed(tmp_path, monkeypatch):
    from providers.media.openrouter_media import AudioTranscriptTruncated
    path = tmp_path / "original.ogg"
    path.write_bytes(b"original")
    inner = Mock()
    inner.transcribe_audio.side_effect = ["first words", AudioTranscriptTruncated("length"), "last words"]
    provider = ChunkedAudioProvider(inner, chunk_seconds=60, max_file_size_mb=60,
                                    accept_unconfirmed_silence=True)
    _parts(provider, monkeypatch)
    text = provider.transcribe_audio(path)
    assert "first words" in text and "last words" in text
    assert "[Audio 00:01:00–00:02:00 not transcribed: provider output did not complete (length)]" in text
    assert inner.transcribe_audio.call_count == 3


def test_local_client_reports_a_looping_part_as_truncated(tmp_path, monkeypatch):
    from providers.fallback.litellm_fallback import LiteLLMFallback, audio_encoder
    from providers.media.openrouter_media import AudioTranscriptTruncated
    path = tmp_path / "part.wav"
    path.write_bytes(b"audio")
    monkeypatch.setattr(httpx, "post", Mock(return_value=httpx.Response(
        200, request=httpx.Request("POST", "http://audio/chat/completions"),
        json={"choices": [{"finish_reason": "length", "message": {"content": "city of the city"}}]})))
    client = LiteLLMFallback("http://audio", "audio", "transcribe", audio_encoder,
                             api_key="test", attempts=1, require_complete=True)
    with pytest.raises(AudioTranscriptTruncated, match="finish completely"):
        client.run(path)

