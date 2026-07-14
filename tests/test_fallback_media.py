import pytest
from core.resilience import TransientError
from providers.media.base import MediaProvider
from providers.media.fallback import MediaFallbackProvider


class _Stub(MediaProvider):
    def __init__(self, video_fn=None, audio_fn=None):
        self._v = video_fn or (lambda p: "")
        self._a = audio_fn or (lambda p: "")
    def analyze_video(self, file_path): return self._v(file_path)
    def transcribe_audio(self, file_path): return self._a(file_path)


def _fb(text):
    def run(path): return text
    return run


def test_video_recovers_from_reachable_empty():
    w = MediaFallbackProvider(_Stub(video_fn=lambda p: ""), video_fallback=_fb("v"))
    assert w.analyze_video("/x.mp4") == "v"


def test_video_confirmed_blank_returns_empty():
    w = MediaFallbackProvider(_Stub(video_fn=lambda p: ""), video_fallback=_fb(""))
    assert w.analyze_video("/x.mp4") == ""


def test_video_primary_unreachable_raises_and_fallback_not_called():
    calls = {"n": 0}
    def fb(path): calls["n"] += 1; return "x"
    def boom(p): raise TransientError("down")
    w = MediaFallbackProvider(_Stub(video_fn=boom), video_fallback=fb)
    with pytest.raises(TransientError):
        w.analyze_video("/x.mp4")
    assert calls["n"] == 0


def test_video_dark_mode_empty_raises_transient():
    w = MediaFallbackProvider(_Stub(video_fn=lambda p: ""), video_fallback=None)
    with pytest.raises(TransientError):
        w.analyze_video("/x.mp4")


def test_audio_recovers_from_reachable_empty():
    w = MediaFallbackProvider(_Stub(audio_fn=lambda p: ""), audio_fallback=_fb("a"))
    assert w.transcribe_audio("/x.mp3") == "a"


def test_audio_dark_mode_empty_raises_transient():
    w = MediaFallbackProvider(_Stub(audio_fn=lambda p: ""), audio_fallback=None)
    with pytest.raises(TransientError):
        w.transcribe_audio("/x.mp3")


def test_audio_encoder_shape(tmp_path):
    from providers.fallback.litellm_fallback import audio_encoder
    f = tmp_path / "a.mp3"; f.write_bytes(b"ID3")
    parts = audio_encoder(f, "transcribe")
    assert parts[0] == {"type": "text", "text": "transcribe"}
    assert parts[1]["type"] == "input_audio"
    assert parts[1]["input_audio"]["format"] == "mp3"
    assert parts[1]["input_audio"]["data"]  # base64 present


def test_video_encoder_shape(tmp_path):
    from providers.fallback.litellm_fallback import video_encoder
    f = tmp_path / "v.mp4"; f.write_bytes(b"\x00\x00")
    parts = video_encoder(f, "analyze")
    assert parts[0] == {"type": "text", "text": "analyze"}
    assert parts[1]["type"] == "video_url"
    assert parts[1]["video_url"]["url"].startswith("data:video/mp4;base64,")
