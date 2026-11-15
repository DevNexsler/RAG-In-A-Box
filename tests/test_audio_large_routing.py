from unittest.mock import Mock

import httpx
import pytest

from core.resilience import TransientError
from providers.media import build_media_provider
from providers.media.chunked_audio import ChunkedAudioProvider
from providers.media.openrouter_media import OpenRouterMediaProvider


def test_short_clip_uses_local_and_long_clip_uses_configured_alternate(tmp_path, monkeypatch):
    path = tmp_path / "recording.mp3"
    path.write_bytes(b"compressed audio")
    local = Mock()
    local.transcribe_audio.return_value = "short local transcript"
    large = Mock(return_value="complete long transcript")
    provider = ChunkedAudioProvider(local, chunk_size_mb=55, max_file_size_mb=60,
                                    large_audio_provider=large, max_local_audio_seconds=60)
    monkeypatch.setattr(provider, "_duration", lambda _: 30)
    assert provider.transcribe_audio(path) == "short local transcript"
    monkeypatch.setattr(provider, "_duration", lambda _: 1200)
    assert provider.transcribe_audio(path) == "complete long transcript"
    assert local.transcribe_audio.call_count == 1
    large.assert_called_once_with(path.resolve())


def test_large_part_failure_never_returns_a_partial_merged_transcript(tmp_path, monkeypatch):
    path = tmp_path / "recording.mp3"
    path.write_bytes(b"compressed original")
    local = Mock()
    large = Mock(side_effect=["first complete part", TransientError("last part failed")])
    provider = ChunkedAudioProvider(local, chunk_size_mb=65580 / 1024 / 1024,
                                    max_file_size_mb=60, large_audio_provider=large,
                                    max_local_audio_seconds=1)
    monkeypatch.setattr(provider, "_duration", lambda p: 2 if p.name.startswith("part-") else 4)
    paths = []

    def split(_path, directory):
        from pathlib import Path
        for i in range(2):
            part = Path(directory) / f"part-{i:05d}.wav"
            part.write_bytes(b"part")
            paths.append(part)

    monkeypatch.setattr(provider, "_split_by_size", split)
    with pytest.raises(TransientError, match="last part failed"):
        provider.transcribe_audio(path)
    local.transcribe_audio.assert_not_called()
    assert all(not p.exists() for p in paths)


def test_factory_requires_explicit_large_route_and_existing_credentials(monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "qualification")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = {"media": {"enabled": True, "provider": "litellm", "audio_chunk_size_mb": 55,
                      "audio_large_provider": "openrouter"}}
    with pytest.raises(ValueError, match="configured OpenRouter"):
        build_media_provider(cfg)
    monkeypatch.setenv("OPENROUTER_API_KEY", "qualification")
    p = build_media_provider(cfg)
    assert p.large_audio_provider is not None


def test_reasoning_and_output_budget_are_audio_only(tmp_path, monkeypatch):
    path = tmp_path / "part.wav"
    path.write_bytes(b"audio")
    monkeypatch.setattr(httpx, "post", Mock(return_value=httpx.Response(
        200, request=httpx.Request("POST", "http://provider/chat/completions"),
        json={"choices": [{"finish_reason": "stop", "message": {"content": "complete"}}]})))
    p = OpenRouterMediaProvider(["large-audio"], "video", api_key="test",
                                 require_complete_audio=True, audio_max_output_tokens=65536,
                                 audio_reasoning={"effort": "minimal"})
    assert p.transcribe_audio(path) == "complete"
    payload = httpx.post.call_args.kwargs["json"]
    assert payload["max_tokens"] == 65536 and payload["reasoning"] == {"effort": "minimal"}
    p.analyze_video(path)
    payload = httpx.post.call_args.kwargs["json"]
    assert "reasoning" not in payload and "max_tokens" not in payload
