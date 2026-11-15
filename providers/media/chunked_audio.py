"""Bound long audio requests without changing original files or media identity."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import tempfile
from threading import Timer
import wave

from core.resilience import TransientError
from providers.media.openrouter_media import AudioSpanUntranscribable, MediaFileTooLargeError


class ChunkedAudioProvider:
    """Split audio by a configured byte or duration ceiling, then join its text."""

    def __init__(self, provider, *, chunk_seconds: float | None = None,
                 chunk_size_mb: float | None = None,
                 max_file_size_mb: float, max_parts: int = 256,
                 no_speech_fallback=None, large_audio_provider=None,
                 max_local_audio_seconds: float = 600):
        self.provider = provider
        self.max_bytes = int(max_file_size_mb * 1024 * 1024)
        if (chunk_seconds is None) == (chunk_size_mb is None):
            raise ValueError("Configure exactly one audio chunk size or duration")
        self.chunk_seconds = None if chunk_seconds is None else float(chunk_seconds)
        self.chunk_bytes = None
        if chunk_size_mb is not None:
            size = float(chunk_size_mb)
            if not math.isfinite(size) or size * 1024 * 1024 < 46:
                raise ValueError("media.audio_chunk_size_mb must fit a PCM WAV header and frame")
            self.chunk_bytes = int(size * 1024 * 1024)
            if self.max_bytes >= 0 and self.chunk_bytes > self.max_bytes:
                raise ValueError("Audio chunk size exceeds media.max_file_size_mb")
        elif not math.isfinite(self.chunk_seconds) or self.chunk_seconds <= 0:
            raise ValueError("media.audio_chunk_seconds must be finite and positive")
        self.max_parts = max_parts
        # A part the provider cannot transcribe for a content reason (declined,
        # or a looping output that exhausts the token limit) is re-split into
        # pieces this long; a piece that still fails is marked, not retried.
        self.blocked_piece_seconds = 300.0
        self.no_speech_fallback = no_speech_fallback
        self.large_audio_provider = large_audio_provider
        self.max_local_audio_seconds = float(max_local_audio_seconds)
        if not math.isfinite(self.max_local_audio_seconds) or self.max_local_audio_seconds <= 0:
            raise ValueError("media.audio_local_max_seconds must be finite and positive")

    @staticmethod
    def _no_speech_marker(text):
        return text.strip().casefold().strip("[] .!") in {
            "no intelligible speech", "no speech detected", "no speech",
        }

    def analyze_video(self, file_path):
        return self.provider.analyze_video(file_path)

    @staticmethod
    def _run(command, *, timeout):
        try:
            return subprocess.run(command, check=True, capture_output=True,
                                  text=True, timeout=timeout).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            # Tool failure is retryable, never a successful empty transcript.
            raise TransientError("Audio segmentation tool failed") from exc

    def _duration(self, path):
        data = self._run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", str(path),
        ], timeout=30)
        try:
            duration = float(json.loads(data)["format"]["duration"])
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("invalid duration")
            return duration
        except (KeyError, TypeError, ValueError) as exc:
            raise TransientError("Audio duration could not be verified") from exc

    @staticmethod
    def _timestamp(seconds):
        seconds = int(round(seconds))
        return f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"

    def _split_by_size(self, path, directory):
        # WAV: 44 header bytes, mono signed-16-bit frames at 16 kHz. Splitting
        # decoded frames makes the byte bound exact, independent of source bitrate.
        payload_bytes = (self.chunk_bytes - 44) // 2 * 2
        command = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
                   "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
                   "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"]
        try:
            with tempfile.TemporaryFile() as errors, subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=errors) as process:
                timer = Timer(300, process.kill)
                timer.daemon = True
                timer.start()
                try:
                    count = 0
                    while data := process.stdout.read(payload_bytes):
                        if count >= self.max_parts or len(data) % 2:
                            raise TransientError("Audio segmentation exceeded its frame budget")
                        with wave.open(str(Path(directory) / f"part-{count:05d}.wav"), "wb") as part:
                            part.setnchannels(1)
                            part.setsampwidth(2)
                            part.setframerate(16000)
                            part.writeframes(data)
                        count += 1
                    if process.wait() != 0:
                        raise TransientError("Audio segmentation tool failed")
                finally:
                    timer.cancel()
                    if process.poll() is None:
                        process.kill()
                    process.wait()
        except OSError as exc:
            raise TransientError("Audio segmentation tool failed") from exc

    def _transcribe_part(self, path, seconds):
        large = self.large_audio_provider is not None and seconds > self.max_local_audio_seconds
        text = (self.large_audio_provider(path) if large
                else self.provider.transcribe_audio(path))
        if large and isinstance(text, str) and not text.strip():
            # The large route requires a complete response (finish_reason
            # stop) and its prompt asks for nothing when there is no speech,
            # so a complete empty answer is a verified silent segment -- long
            # recordings have hours of them.
            return "[No speech detected]"
        if not isinstance(text, str) or not text.strip():
            raise TransientError("Audio segment returned no verified transcript")
        if self._no_speech_marker(text) and not large:
            if self.no_speech_fallback is None:
                raise TransientError("No alternate provider to confirm no-speech response")
            text = self.no_speech_fallback(path)
            if not isinstance(text, str) or not text.strip():
                raise TransientError("No-speech confirmation returned no transcript")
        return text

    def _transcribe_blocked_part(self, path, seconds, offset, reason):
        """Re-split a part that failed for a content reason; mark only what still fails.

        The marker names the span and the provider's reason, so the gap is
        visible in the transcript instead of failing the whole recording on
        every index run (a 13.75 h Plaud recording failed 39 times on one
        jingle the provider's recitation filter declined).
        """
        pieces_dir = Path(tempfile.mkdtemp(prefix="blocked-", dir=Path(path).parent))
        self._run([
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
            "-f", "segment", "-segment_time", str(self.blocked_piece_seconds),
            "-reset_timestamps", "1", "-c", "copy",
            str(pieces_dir / "piece-%05d.wav"),
        ], timeout=300)
        pieces = sorted(pieces_dir.glob("piece-*.wav"))
        if not pieces:
            raise TransientError("Audio segmentation produced no pieces for a declined part")
        texts = []
        start = offset
        for piece in pieces:
            piece_seconds = self._duration(piece)
            span = f"{self._timestamp(start)}–{self._timestamp(start + piece_seconds)}"
            try:
                texts.append(self._transcribe_part(piece, piece_seconds).strip())
            except AudioSpanUntranscribable as exc:
                texts.append(f"[Audio {span} not transcribed: {exc.marker} ({exc.reason})]")
            start += piece_seconds
        if abs((start - offset) - seconds) > 1:
            raise TransientError("Declined-part pieces do not cover the part")
        return "\n".join(texts)

    def transcribe_audio(self, file_path):
        path = Path(file_path).resolve()
        duration = self._duration(path)
        seconds_per_part = (self.chunk_bytes - 44) // 2 / 16000 if self.chunk_bytes else self.chunk_seconds
        upload_limit = self.chunk_bytes if self.chunk_bytes is not None else self.max_bytes
        oversized = upload_limit >= 0 and path.stat().st_size > upload_limit
        if duration <= seconds_per_part and not oversized:
            if self.large_audio_provider is not None and duration > self.max_local_audio_seconds:
                return self._transcribe_part(path, duration)
            return self.provider.transcribe_audio(path)
        if math.ceil(duration / seconds_per_part) > self.max_parts:
            raise TransientError("Audio exceeds the bounded segmentation budget")

        with tempfile.TemporaryDirectory(prefix="doc-organizer-audio-") as tmp:
            if self.chunk_bytes is not None:
                self._split_by_size(path, tmp)
            else:
                self._run([
                    "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
                    "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
                    "-c:a", "pcm_s16le", "-f", "segment", "-segment_time",
                    str(self.chunk_seconds), "-reset_timestamps", "1",
                    str(Path(tmp) / "part-%05d.wav"),
                ], timeout=300)
            parts = sorted(Path(tmp).glob("part-*.wav"))
            if not parts or len(parts) > self.max_parts:
                raise TransientError("Audio segmentation produced an invalid part count")
            # The upload ceiling applies to requests, not the original recording.
            # Check every part before sending any of them.
            if upload_limit >= 0 and any(part.stat().st_size > upload_limit for part in parts):
                raise MediaFileTooLargeError("Audio segment exceeds media.max_file_size_mb")
            durations = [self._duration(part) for part in parts]
            if (any(d > seconds_per_part + 1 for d in durations)
                    or abs(sum(durations) - duration) > 1):
                raise TransientError("Audio segments do not cover the complete recording")
            results = [
                "Speaker labels are local to each segment and do not identify a person."
            ]
            offset = 0.0
            for i, (part, seconds) in enumerate(zip(parts, durations), 1):
                try:
                    text = self._transcribe_part(part, seconds)
                except AudioSpanUntranscribable as exc:
                    text = self._transcribe_blocked_part(part, seconds, offset, exc.reason)
                results.append(
                    f"[Audio segment {i}/{len(parts)} "
                    f"{self._timestamp(offset)}–{self._timestamp(offset + seconds)}]\n"
                    + text.strip()
                )
                offset += seconds
            # No partial transcript escapes if any segment or provider fails.
            return "\n\n".join(results)
