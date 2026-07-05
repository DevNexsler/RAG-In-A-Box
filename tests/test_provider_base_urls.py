"""Configurable base_url for provider endpoints (staging gate).

Each provider must honor a config-supplied base_url override and fall back
to today's exact hardcoded URL when unset (zero behavior change).

Tests go through the factories (build_* functions) so the config plumbing is
exercised too, and monkeypatch httpx.post to capture the real URL each
provider constructs — no network calls.
"""

from __future__ import annotations

import httpx
import pytest

from core.storage import SearchHit
from providers.embed import build_embed_provider
from providers.llm import build_llm_provider
from providers.media import build_media_provider
from search_hybrid import build_reranker

OVERRIDE = "http://sim:9999"


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload
        self.headers: dict = {}
        self.text = ""

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


@pytest.fixture
def capture_post(monkeypatch):
    """Patch httpx.post to record the URL and return a canned response."""
    calls: dict = {"url": None, "payload": None}

    def _make(payload: dict):
        def _fake_post(url, **kwargs):
            calls["url"] = url
            return _FakeResponse(payload)

        monkeypatch.setattr(httpx, "post", _fake_post)
        return calls

    return _make


# ---------------------------------------------------------------------------
# OpenRouter embeddings — embeddings.base_url
# ---------------------------------------------------------------------------

def _embed_provider(base_url: str | None):
    emb_cfg = {"provider": "openrouter", "api_key": "test-key"}
    if base_url is not None:
        emb_cfg["base_url"] = base_url
    return build_embed_provider({"embeddings": emb_cfg})


def test_embed_base_url_override(capture_post):
    calls = capture_post({"data": [{"index": 0, "embedding": [0.1]}]})
    provider = _embed_provider(OVERRIDE)
    provider.embed_texts(["hello"])
    assert calls["url"].startswith(OVERRIDE)
    assert calls["url"] == f"{OVERRIDE}/embeddings"


def test_embed_base_url_default(capture_post):
    calls = capture_post({"data": [{"index": 0, "embedding": [0.1]}]})
    provider = _embed_provider(None)
    provider.embed_texts(["hello"])
    assert calls["url"] == "https://openrouter.ai/api/v1/embeddings"


# ---------------------------------------------------------------------------
# OpenRouter LLM — enrichment.base_url
# ---------------------------------------------------------------------------

def _llm_provider(base_url: str | None):
    enr_cfg = {"enabled": True, "provider": "openrouter", "api_key": "test-key"}
    if base_url is not None:
        enr_cfg["base_url"] = base_url
    provider = build_llm_provider({"enrichment": enr_cfg})
    assert provider is not None
    return provider


def test_llm_base_url_override(capture_post):
    calls = capture_post({"choices": [{"message": {"content": "{}"}}]})
    provider = _llm_provider(OVERRIDE)
    provider.generate("hi")
    assert calls["url"].startswith(OVERRIDE)
    assert calls["url"] == f"{OVERRIDE}/chat/completions"


def test_llm_base_url_default(capture_post):
    calls = capture_post({"choices": [{"message": {"content": "{}"}}]})
    provider = _llm_provider(None)
    provider.generate("hi")
    assert calls["url"] == "https://openrouter.ai/api/v1/chat/completions"


# ---------------------------------------------------------------------------
# DeepInfra reranker — search.reranker.base_url
# ---------------------------------------------------------------------------

def _reranker(base_url: str | None):
    rr_cfg = {"enabled": True, "provider": "deepinfra", "api_key": "test-key"}
    if base_url is not None:
        rr_cfg["base_url"] = base_url
    reranker = build_reranker({"search": {"reranker": rr_cfg}})
    assert reranker is not None
    return reranker


def _one_hit() -> SearchHit:
    return SearchHit(doc_id="d1", loc="0", snippet="s", text="body", score=0.5)


def test_reranker_base_url_override(capture_post):
    calls = capture_post({"scores": [1.0]})
    reranker = _reranker(OVERRIDE)
    reranker.rerank("query", [_one_hit()])
    assert calls["url"].startswith(OVERRIDE)
    assert calls["url"] == f"{OVERRIDE}/v1/inference/Qwen/Qwen3-Reranker-8B"


def test_reranker_base_url_default(capture_post):
    calls = capture_post({"scores": [1.0]})
    reranker = _reranker(None)
    reranker.rerank("query", [_one_hit()])
    assert calls["url"] == (
        "https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Reranker-8B"
    )


# ---------------------------------------------------------------------------
# OpenRouter media — media.base_url
# ---------------------------------------------------------------------------

def _media_provider(base_url: str | None):
    media_cfg = {
        "enabled": True,
        "provider": "openrouter",
        "api_key": "test-key",
        "audio_models": ["openai/whisper-1"],
        "video_model": "qwen/qwen3.5-397b-a17b",
    }
    if base_url is not None:
        media_cfg["base_url"] = base_url
    provider = build_media_provider({"media": media_cfg})
    assert provider is not None
    return provider


def test_media_base_url_override(capture_post, tmp_path):
    calls = capture_post({"choices": [{"message": {"content": "transcript"}}]})
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"fake-audio")
    provider = _media_provider(OVERRIDE)
    provider.transcribe_audio(audio)
    assert calls["url"].startswith(OVERRIDE)
    assert calls["url"] == f"{OVERRIDE}/chat/completions"


def test_media_base_url_default(capture_post, tmp_path):
    calls = capture_post({"choices": [{"message": {"content": "transcript"}}]})
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"fake-audio")
    provider = _media_provider(None)
    provider.transcribe_audio(audio)
    assert calls["url"] == "https://openrouter.ai/api/v1/chat/completions"
