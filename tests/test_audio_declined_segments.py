"""A provider's content refusal on one span must not fail the whole recording.

Gemini's recitation filter declined one 30-minute part of a 13.75 h Plaud
recording (a long repeated jingle); the all-or-nothing chunk join then failed
the document on every index run. A declined part is re-split and only the
still-declined span is left out, with a visible marker.
"""

import shutil
from unittest.mock import Mock
import wave

import httpx
import pytest

from core.resilience import TransientError
from providers.media.chunked_audio import ChunkedAudioProvider
from providers.media.openrouter_media import AudioTranscriptBlocked, OpenRouterMediaProvider

pytestmark = pytest.mark.unit


def _response(finish, native=None, content=""):
    choice = {"finish_reason": finish, "message": {"content": content}}
    if native is not None:
        choice["native_finish_reason"] = native
    return httpx.Response(200, request=httpx.Request("POST", "http://audio/chat/completions"),
                          json={"choices": [choice]})


@pytest.mark.parametrize("finish,native", [("error", "RECITATION"), ("error", "SAFETY"),
                                           ("content_filter", None)])
def test_alternate_reports_a_content_refusal_distinctly(tmp_path, monkeypatch, finish, native):
    path = tmp_path / "part.wav"
    path.write_bytes(b"audio")
    monkeypatch.setattr(httpx, "post", Mock(return_value=_response(finish, native, "Firefox. Firefox.")))
    provider = OpenRouterMediaProvider(["audio"], "video", api_key="test",
                                       require_complete_audio=True)
    with pytest.raises(AudioTranscriptBlocked) as caught:
        provider.transcribe_audio(path)
    assert isinstance(caught.value, TransientError)  # callers without chunking keep retry semantics


def test_alternate_truncation_is_still_an_incomplete_response(tmp_path, monkeypatch):
    path = tmp_path / "part.wav"
    path.write_bytes(b"audio")
    monkeypatch.setattr(httpx, "post", Mock(return_value=_response("length", "MAX_TOKENS", "partial")))
    provider = OpenRouterMediaProvider(["audio"], "video", api_key="test",
                                       require_complete_audio=True)
    with pytest.raises(TransientError, match="finish completely") as caught:
        provider.transcribe_audio(path)
    assert not isinstance(caught.value, AudioTranscriptBlocked)


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                    reason="ffmpeg/ffprobe required")
def test_declined_part_is_resplit_and_only_the_declined_span_is_marked(tmp_path):
    source = tmp_path / "recording.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 16000 * 900)  # 15 minutes

    calls = []

    def large(path):
        calls.append(path.name)
        # Part 2 is declined whole; inside it, only its second piece is.
        if path.name == "part-00001.wav" or path.name == "piece-00001.wav":
            raise AudioTranscriptBlocked("RECITATION")
        return f"words from {path.name}"

    provider = ChunkedAudioProvider(Mock(), chunk_size_mb=10, max_file_size_mb=60,
                                    large_audio_provider=large, max_local_audio_seconds=10)
    provider.blocked_piece_seconds = 100
    text = provider.transcribe_audio(source)

    assert "words from part-00000.wav" in text
    assert "words from part-00002.wav" in text
    assert "words from piece-00000.wav" in text and "words from piece-00002.wav" in text
    assert text.count("not transcribed: provider declined (RECITATION)") == 1
    # The marker names the declined span inside the recording, not inside the part.
    part_seconds = (10 * 1024 * 1024 - 44) // 2 / 16000
    start = int(round(part_seconds + 100))
    assert f"[Audio 00:{start // 60:02}:{start % 60:02}–" in text
    assert calls[0] == "part-00000.wav" and calls[-1] == "part-00002.wav"


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                    reason="ffmpeg/ffprobe required")
def test_other_part_failures_still_fail_the_recording(tmp_path):
    source = tmp_path / "recording.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 16000 * 700)

    def large(path):
        if path.name == "part-00001.wav":
            raise TransientError("Audio transcript response did not finish completely")
        return "words"

    provider = ChunkedAudioProvider(Mock(), chunk_size_mb=10, max_file_size_mb=60,
                                    large_audio_provider=large, max_local_audio_seconds=10)
    with pytest.raises(TransientError, match="finish completely"):
        provider.transcribe_audio(source)
