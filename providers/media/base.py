"""Media provider interface for audio/video extraction."""

from pathlib import Path
from typing import Protocol


class MediaProvider(Protocol):
    """Extract searchable text from local audio and video files."""

    def transcribe_audio(self, file_path: str | Path) -> str:
        """Return a transcript or searchable notes for an audio file."""
        ...

    def analyze_video(self, file_path: str | Path) -> str:
        """Return searchable notes for a video file."""
        ...
