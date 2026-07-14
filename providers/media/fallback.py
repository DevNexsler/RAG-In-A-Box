"""MediaProvider that adds a per-method LiteLLM fallback via the shared decision rule."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from core.fallback import resolve_with_fallback
from providers.media.base import MediaProvider

FallbackRun = Callable[[str], str]


class MediaFallbackProvider(MediaProvider):
    def __init__(self, primary: MediaProvider,
                 video_fallback: Optional[FallbackRun] = None,
                 audio_fallback: Optional[FallbackRun] = None):
        self._primary = primary
        self._video_fb = video_fallback
        self._audio_fb = audio_fallback

    def analyze_video(self, file_path: str | Path) -> str:
        p = str(file_path)
        return resolve_with_fallback(
            lambda: self._primary.analyze_video(p),
            (lambda: self._video_fb(p)) if self._video_fb else None)

    def transcribe_audio(self, file_path: str | Path) -> str:
        p = str(file_path)
        return resolve_with_fallback(
            lambda: self._primary.transcribe_audio(p),
            (lambda: self._audio_fb(p)) if self._audio_fb else None)
