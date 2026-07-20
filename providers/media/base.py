"""Media provider interface for audio/video extraction."""

from pathlib import Path
from typing import Protocol


class MediaPolicyError(ValueError):
    """Deterministic policy rejection of a media file (e.g. configured size cap).

    Raised before any backend call: the file and the backend are both fine —
    policy says do not process this file. Extractors classify it as an
    intentional, permanent skip (skip ledger), never as an extraction failure
    (degraded ledger / provider-failure log lines). Subclasses set
    ``skip_reason`` to the ledger token naming their policy.
    """

    skip_reason = "media_policy_rejected"


class MediaProvider(Protocol):
    """Extract searchable text from local audio and video files."""

    def transcribe_audio(self, file_path: str | Path) -> str:
        """Return a transcript or searchable notes for an audio file."""
        ...

    def analyze_video(self, file_path: str | Path) -> str:
        """Return searchable notes for a video file."""
        ...
