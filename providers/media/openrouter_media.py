"""OpenRouter media extraction for local audio/video files."""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from pathlib import Path

import httpx

from core.resilience import TransientError
from providers.media.base import MediaPolicyError

logger = logging.getLogger(__name__)

_AUDIO_TRANSCRIBE_PROMPT = (
    "Transcribe this audio faithfully for document search. If multiple speakers "
    "are clear, label them only as Speaker 1, Speaker 2, and do not infer "
    "identity. Return plain text only."
)
_VIDEO_ANALYZE_PROMPT = (
    "You are reviewing a residential/property walkthrough video for maintenance, "
    "turnover, and owner-facing documentation. Use only visible/audible evidence. "
    "Do not invent defects. If uncertain, say unknown/unclear.\n\n"
    "Return plain text in this exact structure:\n\n"
    "1. Chronological walkthrough\n"
    "- Timestamp range: what is visible or happening, including rooms/areas, "
    "fixtures, appliances, cabinets, floors, walls, lighting, doors/windows.\n\n"
    "2. Condition evaluation\n"
    "- Cleanliness: clean / light cleaning / deep cleaning / unknown, with evidence.\n"
    "- Damage/defects: visible broken, missing, stained, cracked, leaking, loose, "
    "worn, unfinished, scuffed, damaged, or malfunctioning items. Say none clearly "
    "visible if none.\n"
    "- Quality/state: good/fair/poor/dated/new-looking for fixtures, cabinets, "
    "appliances, flooring, walls, storage.\n\n"
    "3. Action list\n"
    "- Must fix before turnover/rent/closeout.\n"
    "- Should clean.\n"
    "- Optional update/cosmetic.\n"
    "- Items needing human confirmation because view is unclear.\n\n"
    "4. Spoken audio and visible text\n"
    "- Transcript if audible; visible text if legible; otherwise none.\n\n"
    "5. Confidence and limitations\n"
    "- Note camera shake, lighting, blur, blocked view, or model uncertainty."
)

_AUDIO_FORMAT_BY_EXT = {
    "mp3": "mp3",
    "wav": "wav",
    "m4a": "m4a",
    "flac": "flac",
    "ogg": "ogg",
    "aac": "aac",
    "aiff": "aiff",
    "webm": "webm",
}

_VIDEO_MIME_BY_EXT = {
    "mp4": "video/mp4",
    "mov": "video/quicktime",
    "m4v": "video/mp4",
    "mkv": "video/x-matroska",
    "webm": "video/webm",
    "avi": "video/x-msvideo",
}


class MediaFileTooLargeError(MediaPolicyError):
    """Raised when a local media file exceeds configured upload size."""

    skip_reason = "media_file_too_large"


class OpenRouterMediaProvider:
    """Extract text from audio/video via OpenRouter chat completions."""

    def __init__(
        self,
        audio_models: list[str],
        video_model: str,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 300.0,
        max_file_size_mb: float = 50.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.audio_models = audio_models
        self.video_model = video_model
        self.timeout = timeout
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set. Set it in .env or pass api_key.")
        if not self.audio_models:
            raise ValueError("At least one audio model is required")
        if not self.video_model:
            raise ValueError("video_model is required")

    def transcribe_audio(self, file_path: str | Path) -> str:
        """Transcribe a local audio file, falling back across configured models."""
        file_path = Path(file_path)
        audio_b64 = self._read_base64(file_path)
        audio_format = self._audio_format(file_path)
        content = [
            {"type": "text", "text": _AUDIO_TRANSCRIBE_PROMPT},
            {
                "type": "input_audio",
                "input_audio": {
                    "data": audio_b64,
                    "format": audio_format,
                },
            },
        ]

        last_exc: Exception | None = None
        for model in self.audio_models:
            try:
                return self._chat(model=model, content=content)
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
                last_exc = exc
                logger.warning("OpenRouter audio model failed (%s): %s", model, exc)

        raise TransientError("All OpenRouter audio models failed") from last_exc

    def analyze_video(self, file_path: str | Path) -> str:
        """Analyze a local video file using a base64 data URL."""
        file_path = Path(file_path)
        video_b64 = self._read_base64(file_path)
        mime_type = self._video_mime_type(file_path)
        content = [
            {"type": "text", "text": _VIDEO_ANALYZE_PROMPT},
            {
                "type": "video_url",
                "video_url": {
                    "url": f"data:{mime_type};base64,{video_b64}",
                },
            },
        ]
        return self._chat(model=self.video_model, content=content)

    def _read_base64(self, file_path: Path) -> str:
        size = file_path.stat().st_size
        if self.max_file_size_bytes >= 0 and size > self.max_file_size_bytes:
            max_mb = self.max_file_size_bytes / 1024 / 1024
            raise MediaFileTooLargeError(
                f"{file_path.name} is {size} bytes, exceeds media.max_file_size_mb={max_mb:g}"
            )
        return base64.b64encode(file_path.read_bytes()).decode("ascii")

    def _audio_format(self, file_path: Path) -> str:
        ext = file_path.suffix.lower().lstrip(".")
        return _AUDIO_FORMAT_BY_EXT.get(ext, ext or "mp3")

    def _video_mime_type(self, file_path: Path) -> str:
        ext = file_path.suffix.lower().lstrip(".")
        return _VIDEO_MIME_BY_EXT.get(
            ext,
            mimetypes.guess_type(file_path.name)[0] or "video/mp4",
        )

    def _chat(self, model: str, content: list[dict]) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]
        content_value = message.get("content", "")
        if isinstance(content_value, list):
            return "\n".join(
                str(part.get("text", "") if isinstance(part, dict) else part)
                for part in content_value
            ).strip()
        return str(content_value).strip()
