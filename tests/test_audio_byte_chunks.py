from unittest.mock import Mock

import httpx
import pytest

from core.resilience import TransientError
from providers.fallback.litellm_fallback import LiteLLMFallback, audio_encoder
from providers.media import build_media_provider
from providers.media.chunked_audio import ChunkedAudioProvider
from providers.media.openrouter_media import OpenRouterMediaProvider


@pytest.mark.parametrize("size", [0, -1, float("nan"), float("inf"), .00001, 61])
def test_invalid_byte_limit_is_rejected(size):
    with pytest.raises(ValueError):
        ChunkedAudioProvider(Mock(), chunk_size_mb=size, max_file_size_mb=60)


def test_chunking_modes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="exactly one"):
        ChunkedAudioProvider(Mock(), chunk_seconds=60, chunk_size_mb=55, max_file_size_mb=60)


def test_factory_byte_mode_configures_complete_audio_only(monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "qualification")
    monkeypatch.setenv("OPENROUTER_API_KEY", "qualification")
    provider = build_media_provider({"media": {"enabled": True, "provider": "litellm",
                                               "audio_chunk_size_mb": 55,
                                               "max_file_size_mb": 60}})
    assert isinstance(provider, ChunkedAudioProvider)
    assert provider.chunk_bytes == 55 * 1024 * 1024
    assert provider.chunk_seconds is None
    primary = provider.provider._primary
    assert primary._audio_run.__self__.require_complete
    assert primary._audio_run.__self__.max_tokens == 16384
    assert "Transcribe all speech" in primary._audio_run.__self__.prompt
    assert "[No intelligible speech]" not in primary._audio_run.__self__.prompt
    assert not primary._video_run.__self__.require_complete
    assert provider.no_speech_fallback.__self__.require_complete_audio


@pytest.mark.parametrize("reason", ["length", "content_filter", None])
def test_primary_refuses_partial_audio_response(tmp_path, monkeypatch, reason):
    path = tmp_path / "part.wav"
    path.write_bytes(b"audio")
    response = httpx.Response(200, request=httpx.Request("POST", "http://audio/chat/completions"),
                             json={"choices": [{"finish_reason": reason,
                                                "message": {"content": "partial words"}}]})
    monkeypatch.setattr(httpx, "post", Mock(return_value=response))
    provider = LiteLLMFallback("http://audio", "audio", "transcribe", audio_encoder,
                               api_key="test", attempts=1, require_complete=True,
                               max_tokens=16384)
    with pytest.raises(TransientError, match="finish completely"):
        provider.run(path)
    assert httpx.post.call_args.kwargs["json"]["max_tokens"] == 16384


@pytest.mark.parametrize("reason", ["stop", "end_turn"])
def test_primary_accepts_complete_audio_response(tmp_path, monkeypatch, reason):
    path = tmp_path / "part.wav"
    path.write_bytes(b"audio")
    monkeypatch.setattr(httpx, "post", Mock(return_value=httpx.Response(
        200, request=httpx.Request("POST", "http://audio/chat/completions"),
        json={"choices": [{"finish_reason": reason, "message": {"content": "complete words"}}]})))
    provider = LiteLLMFallback("http://audio", "audio", "transcribe", audio_encoder,
                               api_key="test", attempts=1, require_complete=True)
    assert provider.run(path) == "complete words"


def test_alternate_refuses_partial_audio_but_preserves_video(tmp_path, monkeypatch):
    path = tmp_path / "part.wav"
    path.write_bytes(b"audio")
    monkeypatch.setattr(httpx, "post", Mock(return_value=httpx.Response(
        200, request=httpx.Request("POST", "http://audio/chat/completions"),
        json={"choices": [{"finish_reason": "length", "message": {"content": "partial words"}}]})))
    provider = OpenRouterMediaProvider(["audio"], "video", api_key="test",
                                       require_complete_audio=True, audio_max_output_tokens=16384)
    with pytest.raises(TransientError, match="finish completely"):
        provider.transcribe_audio(path)
    assert provider.analyze_video(path) == "partial words"
    assert "max_tokens" not in httpx.post.call_args.kwargs["json"]
