"""Media providers: extract searchable text from audio/video files."""

import os
from pathlib import Path

from providers.media.base import MediaProvider

__all__ = ["MediaProvider", "build_media_provider", "DEFAULT_VIDEO_MODEL"]

DEFAULT_VIDEO_MODEL = "qwen/qwen3.5-397b-a17b"

_VIDEO_PROMPT = "Describe this video's visual content in detail for document search."
_AUDIO_PROMPT = "Transcribe this audio faithfully for document search."


def _dedupe_models(models: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for model in models:
        model = str(model).strip()
        if model and model not in seen:
            seen.add(model)
            result.append(model)
    return result


def _model_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


class _LiteLLMMediaProvider:
    """MediaProvider that routes describe/transcribe through the LiteLLM proxy.

    Wraps the shared, proven LiteLLMFallback client (correct LITELLM_* key
    handling; reachable-but-blank -> "" rather than the string "None") for both
    modalities, enforcing the same max-file-size guard the OpenRouter provider
    applies before uploading."""

    def __init__(self, video_run, audio_run, *, max_file_size_mb: float = 50.0):
        self._video_run = video_run
        self._audio_run = audio_run
        self._max_bytes = int(max_file_size_mb * 1024 * 1024)

    def _check_size(self, file_path) -> None:
        from providers.media.openrouter_media import MediaFileTooLargeError

        size = Path(file_path).stat().st_size
        if self._max_bytes >= 0 and size > self._max_bytes:
            raise MediaFileTooLargeError(
                f"{Path(file_path).name} is {size} bytes, exceeds "
                f"media.max_file_size_mb={self._max_bytes / 1024 / 1024:g}"
            )

    def analyze_video(self, file_path) -> str:
        self._check_size(file_path)
        return self._video_run(file_path)

    def transcribe_audio(self, file_path) -> str:
        self._check_size(file_path)
        return self._audio_run(file_path)


def _build_litellm_media_provider(media_cfg: dict) -> MediaProvider:
    """LiteLLM-primary media (video/audio), with OpenRouter demoted to fallback.

    Mirrors the OCR block's litellm-primary shape: the LiteLLM proxy is the
    single router for describe/transcribe, and OpenRouter stays reachable as a
    per-modality fallback (via the same transient-vs-blank decision rule) for
    when the proxy is down."""
    from providers.fallback.litellm_fallback import (
        LiteLLMFallback,
        audio_encoder,
        video_encoder,
    )
    from providers.media.openrouter_media import (
        _AUDIO_TRANSCRIBE_PROMPT,
        _VIDEO_ANALYZE_PROMPT,
        OpenRouterMediaProvider,
    )

    endpoint = media_cfg.get("endpoint", "http://192.168.68.87:4000/v1")
    timeout = media_cfg.get("timeout", 300.0)
    api_key = media_cfg.get("api_key")  # else LiteLLMFallback reads LITELLM_*_KEY
    max_mb = media_cfg.get("max_file_size_mb", 50.0)

    video = LiteLLMFallback(
        endpoint, media_cfg.get("video_model", "video"),
        _VIDEO_ANALYZE_PROMPT, video_encoder, api_key=api_key, timeout=timeout,
    )
    audio = LiteLLMFallback(
        endpoint, media_cfg.get("audio_model", "transcribe"),
        _AUDIO_TRANSCRIBE_PROMPT, audio_encoder, api_key=api_key, timeout=timeout,
    )
    primary = _LiteLLMMediaProvider(video.run, audio.run, max_file_size_mb=max_mb)

    # OpenRouter as fallback (reverse of the legacy openrouter-primary layout).
    video_fb = audio_fb = None
    if os.environ.get("OPENROUTER_API_KEY"):
        or_cfg = media_cfg.get("openrouter_fallback", {}) or {}
        audio_models = _dedupe_models(
            _model_list(or_cfg.get("audio_model", "openai/whisper-1"))
            + _model_list(or_cfg.get("fallback_audio_models", []))
        )
        or_provider = OpenRouterMediaProvider(
            api_key=or_cfg.get("api_key"),
            base_url=or_cfg.get("base_url", "https://openrouter.ai/api/v1"),
            audio_models=audio_models,
            video_model=or_cfg.get("video_model", DEFAULT_VIDEO_MODEL),
            timeout=timeout,
            max_file_size_mb=max_mb,
        )
        video_fb = or_provider.analyze_video
        audio_fb = or_provider.transcribe_audio

    from providers.media.fallback import MediaFallbackProvider

    return MediaFallbackProvider(
        primary, video_fallback=video_fb, audio_fallback=audio_fb
    )


def build_media_provider(config: dict) -> MediaProvider | None:
    """Build an optional MediaProvider from the media section of config.yaml."""
    media_cfg = config.get("media", {})
    if not media_cfg.get("enabled", False):
        return None

    provider = media_cfg.get("provider", "openrouter")
    if provider == "none":
        return None
    if provider == "litellm":
        return _build_litellm_media_provider(media_cfg)
    if provider != "openrouter":
        raise ValueError(f"Unknown media provider: {provider}")

    primary_audio = media_cfg.get("audio_model", "openai/whisper-1")
    fallback_audio = _model_list(
        media_cfg.get(
            "fallback_audio_models",
            ["mistralai/voxtral-small-24b-2507", "google/gemini-2.5-flash-lite"],
        )
    )
    audio_models = media_cfg.get("audio_models")
    if audio_models is None:
        audio_models = [primary_audio, *fallback_audio]
    else:
        audio_models = _model_list(audio_models)

    from providers.media.openrouter_media import OpenRouterMediaProvider

    primary = OpenRouterMediaProvider(
        api_key=media_cfg.get("api_key"),
        base_url=media_cfg.get("base_url", "https://openrouter.ai/api/v1"),
        audio_models=_dedupe_models(list(audio_models)),
        video_model=media_cfg.get("video_model", DEFAULT_VIDEO_MODEL),
        timeout=media_cfg.get("timeout", 300.0),
        max_file_size_mb=media_cfg.get("max_file_size_mb", 50.0),
    )

    from providers.media.fallback import MediaFallbackProvider
    from providers.fallback import build_litellm_fallback
    from providers.fallback.litellm_fallback import audio_encoder, video_encoder

    video_fb = build_litellm_fallback(media_cfg.get("video", {}).get("fallback"),
                                      _VIDEO_PROMPT, video_encoder)
    audio_fb = build_litellm_fallback(media_cfg.get("audio", {}).get("fallback"),
                                      _AUDIO_PROMPT, audio_encoder)
    return MediaFallbackProvider(primary, video_fallback=video_fb, audio_fallback=audio_fb)
