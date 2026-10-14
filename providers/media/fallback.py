"""MediaProvider that adds a video LiteLLM fallback via the shared decision rule."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from core.fallback import resolve_with_fallback
from providers.media.base import MediaProvider

FallbackRun = Callable[[str], str]


class MediaFallbackProvider(MediaProvider):
    def __init__(self, primary: MediaProvider,
                 video_fallback: Optional[FallbackRun] = None):
        self._primary = primary
        self._video_fb = video_fallback

    def analyze_video(self, file_path: str | Path) -> str:
        p = str(file_path)
        return resolve_with_fallback(
            lambda: self._primary.analyze_video(p),
            (lambda: self._video_fb(p)) if self._video_fb else None)

    def transcribe_audio(self, file_path: str | Path) -> str:
        p = str(file_path)
        return resolve_with_fallback(
            lambda: self._primary.transcribe_audio(p),
            None,
            # The transcribe prompt maps "no speech" to an empty reply, so a
            # reachable-empty transcript is the model's answer, not a starved
            # provider. Re-asking a cloud model about silence or noise invents a
            # transcript (#3835). An unreachable primary still raises transient.
            empty_is_clean=True,
        )
