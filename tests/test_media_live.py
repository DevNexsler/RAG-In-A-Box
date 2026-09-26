"""Live media tests: real OpenRouter transcription of the e2e fixtures.

Two small real-money calls (~5s clips). Auto-marked live via the `_live`
filename convention in tests/conftest.py.
"""

import os
from pathlib import Path

import pytest
import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from core.resilience import TransientError
from providers.media import build_media_provider

_has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))

pytestmark = pytest.mark.skipif(
    not _has_openrouter, reason="OPENROUTER_API_KEY not set"
)

FIXTURES = Path(__file__).parent / "fixtures" / "e2e"


def _provider():
    # Prefer config_test.yaml's media section (mirrors prod stack); it has
    # none today, so fall back to a minimal openrouter config — the provider
    # reads OPENROUTER_API_KEY from the env.
    config = {}
    cfg_path = Path("config_test.yaml")
    if cfg_path.exists():
        config = yaml.safe_load(cfg_path.read_text()) or {}
    if not (config.get("media") or {}).get("enabled"):
        config["media"] = {"enabled": True, "provider": "openrouter"}
    provider = build_media_provider(config)
    assert provider is not None, "config_test.yaml media section yielded no provider"
    return provider


def test_transcribe_audio_clip_live():
    text = _provider().transcribe_audio(FIXTURES / "clip.wav")
    assert isinstance(text, str)
    assert "the quick brown fox jumps over the lazy dog" in text.lower()


def test_transcribe_audio_tone_live():
    """Reachable nonspeech may be text or an explicitly unconfirmed blank."""
    try:
        text = _provider().transcribe_audio(FIXTURES / "tone.wav")
    except TransientError as exc:
        # Silence can legitimately yield no transcript. The production wrapper
        # deliberately refuses that blank without a confirming fallback.
        # Transport/auth/circuit failures must still fail this live check.
        if type(exc) is not TransientError or str(exc) != (
            "enrichment empty and no fallback configured (unconfirmed blank)"
        ):
            raise
        return
    assert isinstance(text, str)


def test_analyze_video_clip_live():
    text = _provider().analyze_video(FIXTURES / "clip.mp4")
    assert isinstance(text, str)
    assert text.strip(), "expected non-empty transcript/description for clip.mp4"
