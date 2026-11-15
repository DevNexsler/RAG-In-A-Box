"""Media providers: extract searchable text from audio/video files."""

import os
from collections.abc import Mapping
from pathlib import Path

from providers.media.base import MediaProvider

__all__ = ["MediaProvider", "build_media_provider", "DEFAULT_VIDEO_MODEL"]

DEFAULT_VIDEO_MODEL = "qwen/qwen3.5-397b-a17b"

_VIDEO_PROMPT = "Describe this video's visual content in detail for document search."


def _model_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _audio_model_route(
    value,
    *,
    default_endpoint: str,
    parameter_overrides: dict | None = None,
) -> dict:
    """Normalize legacy model strings and explicit capability mappings."""
    if isinstance(value, Mapping):
        model = value.get("model")
        endpoint = value.get("endpoint", default_endpoint)
        explicit_parameters = value.get("parameters") or {}
    else:
        model = value
        endpoint = default_endpoint
        explicit_parameters = {}
    endpoint = str(endpoint).strip().lstrip("/")
    parameters = (
        {"temperature": 0.0, "top_p": 1.0}
        if endpoint == "chat/completions"
        else {}
    )
    parameters.update(parameter_overrides or {})
    parameters.update(explicit_parameters)
    return {
        "model": str(model).strip(),
        "endpoint": endpoint,
        "parameters": parameters,
    }


def _openrouter_audio_routes(
    media_cfg: dict,
    *,
    default_fallbacks: tuple[str, ...] = (
        "mistralai/voxtral-small-24b-2507",
        "google/gemini-2.5-flash-lite",
    ),
) -> list[dict]:
    """Resolve ordered audio models into endpoint-and-payload capabilities."""
    configured = media_cfg.get("audio_models")
    if configured is not None:
        values = _model_list(configured)
        routes = [
            _audio_model_route(value, default_endpoint="chat/completions")
            for value in values
        ]
    else:
        primary = _audio_model_route(
            media_cfg.get("audio_model", "openai/whisper-1"),
            default_endpoint=media_cfg.get(
                "audio_model_endpoint",
                "chat/completions" if media_cfg.get("audio_provider") in {"litellm", "openai_compatible"}
                else "audio/transcriptions"
            ),
            parameter_overrides=media_cfg.get("audio_model_parameters"),
        )
        routes = [primary]
        routes.extend(
            _audio_model_route(
                value,
                default_endpoint=media_cfg.get(
                    "fallback_audio_model_endpoint", "chat/completions"
                ),
                parameter_overrides=media_cfg.get("fallback_audio_model_parameters"),
            )
            for value in _model_list(
                media_cfg.get(
                    "fallback_audio_models",
                    list(default_fallbacks),
                )
            )
        )

    seen: set[str] = set()
    unique: list[dict] = []
    for route in routes:
        if route["model"] and route["model"] not in seen:
            seen.add(route["model"])
            unique.append(route)
    return unique


class _LiteLLMMediaProvider:
    """MediaProvider that routes describe/transcribe through the LiteLLM proxy.

    Wraps the shared, proven LiteLLMFallback client (correct LITELLM_* key
    handling; reachable-but-blank -> "" rather than the string "None") for both
    modalities, enforcing the same max-file-size guard the OpenRouter provider
    applies before uploading."""

    def __init__(self, video_run, audio_run, *, max_file_size_mb: float = 800.0):
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
    single router for describe/transcribe, and OpenRouter stays reachable as the
    video fallback (via the same transient-vs-blank decision rule)."""
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
    max_mb = media_cfg.get("max_file_size_mb", 800.0)
    # Opt-in byte-size audio chunking (Mac mini long-audio routing): each part
    # must finish completely, so request the model's full output allowance.
    byte_chunks = media_cfg.get("audio_chunk_size_mb") is not None
    # "local" keeps every audio request on the LiteLLM route: no OpenRouter
    # long-audio route and no OpenRouter no-speech confirmation. Unset keeps
    # the mixed routing the large-provider options describe.
    route = media_cfg.get("audio_transcription_route", "mixed")
    if route not in {"mixed", "local"}:
        raise ValueError("media.audio_transcription_route must be mixed or local")
    local_only = route == "local"
    strict_audio = byte_chunks or local_only
    complete_audio = ({"require_complete": True,
                       "max_tokens": media_cfg.get("audio_max_output_tokens", 16384)}
                      if strict_audio else {})

    video = LiteLLMFallback(
        endpoint, media_cfg.get("video_model", "video"),
        _VIDEO_ANALYZE_PROMPT, video_encoder, api_key=api_key, timeout=timeout,
    )
    audio = LiteLLMFallback(
        endpoint, media_cfg.get("audio_model", "transcribe"),
        ("Transcribe all speech in this audio faithfully and completely for document search. "
         "Do not summarize or skip speech. If multiple speakers are clear, label them only "
         "as Speaker 1, Speaker 2; do not infer identity. Never invent or reconstruct "
         "dialogue; if there is no speech at all, return nothing. Return plain text only."
         if strict_audio else _AUDIO_TRANSCRIBE_PROMPT),
        audio_encoder, api_key=api_key, timeout=timeout, **complete_audio,
    )
    primary = _LiteLLMMediaProvider(video.run, audio.run, max_file_size_mb=max_mb)

    # OpenRouter as the video fallback (reverse of the legacy openrouter-primary
    # layout). Audio has none: its reachable-empty is a clean blank (#3835).
    video_fb = None
    # OpenRouter audio is used only by the opt-in chunked path below (large-part
    # routing and no-speech confirmation); it is never a generic audio fallback.
    or_audio = None
    if os.environ.get("OPENROUTER_API_KEY"):
        or_cfg = media_cfg.get("openrouter_fallback", {}) or {}
        audio_models = _openrouter_audio_routes(or_cfg, default_fallbacks=())
        or_provider = OpenRouterMediaProvider(
            api_key=or_cfg.get("api_key"),
            base_url=or_cfg.get("base_url", "https://openrouter.ai/api/v1"),
            audio_models=audio_models,
            video_model=or_cfg.get("video_model", DEFAULT_VIDEO_MODEL),
            timeout=timeout,
            max_file_size_mb=max_mb,
            **({"require_complete_audio": True,
                "audio_max_output_tokens": media_cfg.get("audio_max_output_tokens", 16384),
                "audio_reasoning": or_cfg.get("audio_reasoning")}
               if byte_chunks else {}),
        )
        video_fb = or_provider.analyze_video
        or_audio = or_provider.transcribe_audio

    from providers.media.fallback import MediaFallbackProvider

    wrapped = MediaFallbackProvider(primary, video_fallback=video_fb)
    large_provider = media_cfg.get("audio_large_provider")
    if large_provider not in {None, "openrouter"}:
        raise ValueError("media.audio_large_provider must be openrouter when configured")
    if local_only and large_provider:
        raise ValueError("media.audio_large_provider conflicts with audio_transcription_route: local")
    if large_provider and (not byte_chunks or or_audio is None):
        raise ValueError("Large-audio routing requires byte chunks and configured OpenRouter audio")
    if media_cfg.get("audio_chunk_seconds") is not None or byte_chunks:
        from providers.media.chunked_audio import ChunkedAudioProvider

        return ChunkedAudioProvider(
            wrapped,
            chunk_seconds=media_cfg.get("audio_chunk_seconds"),
            chunk_size_mb=media_cfg.get("audio_chunk_size_mb"),
            max_file_size_mb=max_mb,
            no_speech_fallback=None if local_only else or_audio,
            accept_unconfirmed_silence=local_only,
            large_audio_provider=or_audio if large_provider else None,
            max_local_audio_seconds=media_cfg.get("audio_local_max_seconds", 600),
        )
    return wrapped


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

    audio_provider = media_cfg.get("audio_provider", provider)
    if audio_provider not in {"openrouter", "litellm", "openai_compatible"}:
        raise ValueError(f"Unknown audio media provider: {audio_provider}")

    audio_models = _openrouter_audio_routes(media_cfg)

    from providers.media.openrouter_media import OpenRouterMediaProvider

    audio_api_key = media_cfg.get("audio_api_key")
    if not audio_api_key and audio_provider in {"litellm", "openai_compatible"}:
        audio_api_key = (
            os.environ.get("LITELLM_MASTER_KEY", "")
            or os.environ.get("LITELLM_API_KEY", "")
        )

    primary = OpenRouterMediaProvider(
        api_key=media_cfg.get("api_key"),
        base_url=media_cfg.get("base_url", "https://openrouter.ai/api/v1"),
        audio_api_key=audio_api_key,
        audio_base_url=media_cfg.get("audio_base_url"),
        audio_models=audio_models,
        video_model=media_cfg.get("video_model", DEFAULT_VIDEO_MODEL),
        timeout=media_cfg.get("timeout", 300.0),
        max_file_size_mb=media_cfg.get("max_file_size_mb", 800.0),
    )

    from providers.media.fallback import MediaFallbackProvider
    from providers.fallback import build_litellm_fallback
    from providers.fallback.litellm_fallback import video_encoder

    video_fb = build_litellm_fallback(media_cfg.get("video", {}).get("fallback"),
                                      _VIDEO_PROMPT, video_encoder)
    return MediaFallbackProvider(primary, video_fallback=video_fb)
