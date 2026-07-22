"""Media provider interface for audio/video extraction."""

from pathlib import Path
from typing import Protocol


class MediaPolicyError(Exception):
    """The provider refuses this file for a reason derived from the FILE.

    Distinct from every other provider failure: an outage, a timeout or a bad
    response says nothing about the document and is worth retrying, but a file
    that breaks a configured policy (over media.max_file_size_mb, an
    unsupported container) fails identically on every future run. Extraction
    turns these into skip reasons rather than degradations, so the doc reaches
    a terminal, operator-visible state instead of being re-extracted forever
    (#0481: five 160-430MB videos re-attempted every 15 minutes to
    attempts=113 against a give-up cap of 5).

    `skip_reason` is the stable ledger token; subclasses narrow it."""

    skip_reason = "media_policy_violation"


class MediaFileTooLargeError(MediaPolicyError, ValueError):
    """Raised when a local media file exceeds configured upload size."""

    skip_reason = "media_file_too_large"


class MediaProvider(Protocol):
    """Extract searchable text from local audio and video files."""

    def transcribe_audio(self, file_path: str | Path) -> str:
        """Return a transcript or searchable notes for an audio file."""
        ...

    def analyze_video(self, file_path: str | Path) -> str:
        """Return searchable notes for a video file."""
        ...
