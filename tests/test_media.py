"""Tests for audio/video media extraction via OpenRouter."""

import base64
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import yaml


def _ok_response(content: str = "transcribed text") -> httpx.Response:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": content}}]},
        request=request,
    )


def test_source_type_maps_audio_and_video_extensions():
    from core.source_types import canonical_source_type

    assert canonical_source_type("mp3") == "audio"
    assert canonical_source_type(".wav") == "audio"
    assert canonical_source_type("mp4") == "video"
    assert canonical_source_type(".mov") == "video"


def test_build_media_provider_uses_whisper_then_fallbacks(monkeypatch):
    from providers.media import build_media_provider

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    provider = build_media_provider({"media": {"enabled": True}})

    assert provider.audio_models == [
        "openai/whisper-1",
        "mistralai/voxtral-small-24b-2507",
        "google/gemini-2.5-flash-lite",
    ]
    assert provider.video_model == "qwen/qwen3.5-397b-a17b"


@pytest.mark.parametrize(
    "config_path",
    ["config.yaml.example", "config.vps.yaml.example", "config.local.yaml.example"],
)
def test_example_configs_include_media_scan_and_provider(config_path: str):
    raw = yaml.safe_load(Path(config_path).read_text())
    include = raw["scan"]["include"]

    for pattern in ("**/*.mp3", "**/*.wav", "**/*.m4a", "**/*.mp4", "**/*.mov", "**/*.webm"):
        assert pattern in include

    media = raw["media"]
    assert media["enabled"] is True
    assert media["provider"] == "openrouter"
    assert media["audio_model"] == "openai/whisper-1"
    assert media["fallback_audio_models"] == [
        "mistralai/voxtral-small-24b-2507",
        "google/gemini-2.5-flash-lite",
    ]
    assert media["video_model"] == "qwen/qwen3.5-397b-a17b"


def test_openrouter_media_provider_sends_audio_input_payload(tmp_path: Path):
    from providers.media.openrouter_media import OpenRouterMediaProvider

    audio = tmp_path / "voice.mp3"
    audio.write_bytes(b"fake-audio")
    provider = OpenRouterMediaProvider(
        api_key="sk-test",
        audio_models=["openai/whisper-1"],
        video_model="google/gemini-2.5-flash-lite",
        max_file_size_mb=1,
    )

    with patch("providers.media.openrouter_media.httpx.post", return_value=_ok_response()) as post:
        result = provider.transcribe_audio(audio)

    assert result == "transcribed text"
    payload = post.call_args.kwargs["json"]
    assert payload["model"] == "openai/whisper-1"
    content = payload["messages"][0]["content"]
    assert content[1]["type"] == "input_audio"
    assert content[1]["input_audio"]["format"] == "mp3"
    assert content[1]["input_audio"]["data"] == base64.b64encode(b"fake-audio").decode("ascii")


def test_openrouter_media_provider_falls_back_between_audio_models(tmp_path: Path):
    from providers.media.openrouter_media import OpenRouterMediaProvider

    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"fake-audio")
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    failed = httpx.Response(404, json={"error": {"message": "model not found"}}, request=request)

    def fake_post(*args, **kwargs):
        if kwargs["json"]["model"] == "openai/whisper-1":
            raise httpx.HTTPStatusError("missing", request=request, response=failed)
        return _ok_response("fallback transcript")

    provider = OpenRouterMediaProvider(
        api_key="sk-test",
        audio_models=["openai/whisper-1", "mistralai/voxtral-small-24b-2507"],
        video_model="google/gemini-2.5-flash-lite",
        max_file_size_mb=1,
    )

    with patch("providers.media.openrouter_media.httpx.post", side_effect=fake_post) as post:
        result = provider.transcribe_audio(audio)

    assert result == "fallback transcript"
    assert [call.kwargs["json"]["model"] for call in post.call_args_list] == [
        "openai/whisper-1",
        "mistralai/voxtral-small-24b-2507",
    ]


def test_openrouter_media_provider_all_audio_models_fail_is_transient(tmp_path: Path):
    from core.resilience import is_transient
    from providers.media.openrouter_media import OpenRouterMediaProvider

    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"fake-audio")

    def fake_post(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    provider = OpenRouterMediaProvider(
        api_key="sk-test",
        audio_models=["openai/whisper-1", "mistralai/voxtral-small-24b-2507"],
        video_model="google/gemini-2.5-flash-lite",
        max_file_size_mb=1,
    )

    with patch("providers.media.openrouter_media.httpx.post", side_effect=fake_post):
        with pytest.raises(RuntimeError) as excinfo:
            provider.transcribe_audio(audio)

    assert is_transient(excinfo.value) is True


def test_openrouter_media_provider_sends_video_data_url(tmp_path: Path):
    from providers.media.openrouter_media import OpenRouterMediaProvider

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    provider = OpenRouterMediaProvider(
        api_key="sk-test",
        audio_models=["openai/whisper-1"],
        video_model="google/gemini-2.5-flash-lite",
        max_file_size_mb=1,
    )

    with patch("providers.media.openrouter_media.httpx.post", return_value=_ok_response("video notes")) as post:
        result = provider.analyze_video(video)

    assert result == "video notes"
    payload = post.call_args.kwargs["json"]
    assert payload["model"] == "google/gemini-2.5-flash-lite"
    content = payload["messages"][0]["content"]
    prompt = content[0]["text"]
    assert "Chronological walkthrough" in prompt
    assert "Condition evaluation" in prompt
    assert "Must fix before turnover/rent/closeout" in prompt
    assert content[1]["type"] == "video_url"
    assert content[1]["video_url"]["url"] == (
        "data:video/mp4;base64," + base64.b64encode(b"fake-video").decode("ascii")
    )


def test_openrouter_media_provider_rejects_oversize_files(tmp_path: Path):
    from providers.media.openrouter_media import MediaFileTooLargeError, OpenRouterMediaProvider

    audio = tmp_path / "large.mp3"
    audio.write_bytes(b"x" * 2048)
    provider = OpenRouterMediaProvider(
        api_key="sk-test",
        audio_models=["openai/whisper-1"],
        video_model="google/gemini-2.5-flash-lite",
        max_file_size_mb=0.001,
    )

    try:
        provider.transcribe_audio(audio)
    except MediaFileTooLargeError as exc:
        assert "large.mp3" in str(exc)
    else:
        raise AssertionError("Expected MediaFileTooLargeError")


def test_extract_text_dispatches_audio_and_video(tmp_path: Path):
    from extractors import extract_text

    class FakeMediaProvider:
        def transcribe_audio(self, file_path):
            return f"audio transcript from {Path(file_path).name}"

        def analyze_video(self, file_path):
            return f"video notes from {Path(file_path).name}"

    audio = tmp_path / "call.mp3"
    audio.write_bytes(b"audio")
    video = tmp_path / "walkthrough.mp4"
    video.write_bytes(b"video")

    audio_result = extract_text(audio, ext="mp3", media_provider=FakeMediaProvider())
    video_result = extract_text(video, ext="mp4", media_provider=FakeMediaProvider())

    assert audio_result.full_text == "audio transcript from call.mp3"
    assert audio_result.frontmatter["media_type"] == "audio"
    assert video_result.full_text == "video notes from walkthrough.mp4"
    assert video_result.frontmatter["media_type"] == "video"


def test_filesystem_source_injects_media_provider(tmp_path: Path):
    from doc_id_store import DocIDStore
    from sources.filesystem import FilesystemSource

    class FakeMediaProvider:
        def transcribe_audio(self, file_path):
            return f"audio transcript from {Path(file_path).name}"

        def analyze_video(self, file_path):
            return ""

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "call.mp3").write_bytes(b"audio")
    source = FilesystemSource(
        name="documents",
        root=vault,
        scan_config={"include": ["**/*.mp3"], "exclude": []},
        registry=DocIDStore(tmp_path / "reg.db"),
    )
    source.set_media_provider(FakeMediaProvider())

    [record] = list(source.scan())
    result = source.extract(record)

    assert record.source_type == "audio"
    assert result.full_text.startswith("audio transcript")
