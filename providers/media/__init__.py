"""Media providers: extract searchable text from audio/video files."""

from providers.media.base import MediaProvider

__all__ = ["MediaProvider", "build_media_provider", "DEFAULT_VIDEO_MODEL"]

DEFAULT_VIDEO_MODEL = "qwen/qwen3.5-397b-a17b"


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


def build_media_provider(config: dict) -> MediaProvider | None:
    """Build an optional MediaProvider from the media section of config.yaml."""
    media_cfg = config.get("media", {})
    if not media_cfg.get("enabled", False):
        return None

    provider = media_cfg.get("provider", "openrouter")
    if provider == "none":
        return None
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

    return OpenRouterMediaProvider(
        api_key=media_cfg.get("api_key"),
        base_url=media_cfg.get("base_url", "https://openrouter.ai/api/v1"),
        audio_models=_dedupe_models(list(audio_models)),
        video_model=media_cfg.get("video_model", DEFAULT_VIDEO_MODEL),
        timeout=media_cfg.get("timeout", 300.0),
        max_file_size_mb=media_cfg.get("max_file_size_mb", 50.0),
    )
