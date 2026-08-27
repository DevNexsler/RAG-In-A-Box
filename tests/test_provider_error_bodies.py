"""Every migrated provider keeps the upstream reason on a permanent rejection.

httpx builds ``HTTPStatusError``'s message from the status line and URL alone, so
before #1657 a permanent 4xx reached ``indexer.log`` as nothing but ``Client error
'422 Unprocessable Entity' for url '...'``. A permanent error is never retried and
quarantines the document for good, so that line is the only account of the failure
anyone ever gets — the provider's actual reason sat unread in the response body.

#1657 added ``core.resilience.raise_for_status`` and adopted it at one call site;
#1662 migrated the rest. The table below IS that ticket's acceptance criterion:
one entry per migrated call site, each driving its provider through a permanent
4xx and handing back the exception the *caller* sees (some providers re-wrap it).
Two things are asserted of every entry:

  1. the upstream reason survives into that exception, and
  2. the enriched message still parses back to its true HTTP status — deep health
     reads the status out of this log line with an ordered regex tuple, and
     #0705's carve-out (400/413/422 is one bad document, not a provider outage)
     stops applying the moment that read goes wrong.

A provider added to the codebase without the helper is simply missing from this
table, which is the failure mode this file is meant to make visible.
"""

from __future__ import annotations

import httpx
import pytest

import mcp_server
from core.storage import SearchHit
from providers.embed.baseten_embed import BasetenEmbedProvider
from providers.embed.ollama_embed import OllamaEmbedProvider
from providers.fallback.litellm_fallback import LiteLLMFallback, image_encoder
from providers.llm.baseten_llm import BasetenGenerator
from providers.llm.litellm_llm import LiteLLMGenerator
from providers.llm.ollama_llm import OllamaGenerator
from providers.llm.openrouter_llm import OpenRouterGenerator
from providers.media.openrouter_media import OpenRouterMediaProvider
from providers.ocr.deepseek_ocr2_local import DeepSeekOCR2Local
from providers.ocr.ollama_vision import OllamaVisionOCR
from search_hybrid import DeepInfraReranker

PERMANENT_STATUS = 422

# A 422 whose body embeds a DIFFERENT status than the response carries. Provider
# envelopes really look like this — OpenRouter fronting Nebius, LiteLLM fronting
# anything — so it is the honest adversarial case for the status parser, not a
# contrived one.
UPSTREAM_BODY = (
    '{"error":{"message":"HTTP 500: Value error, The input sequence should have '
    'less than 131072 characters. Input length: 180439","code":500}}'
)
UPSTREAM_REASON = "less than 131072 characters"


def _rejection(url: str = "http://provider.invalid/v1/x", **_kw) -> httpx.Response:
    return httpx.Response(
        PERMANENT_STATUS, text=UPSTREAM_BODY, request=httpx.Request("POST", url)
    )


def _streamed_rejection(request: httpx.Request) -> httpx.Response:
    """The same rejection, but with the body still on the wire.

    `content=iter(...)` is what makes it genuinely streamed — httpx populates
    `_content` eagerly for a bytes/text response, so a `text=` response would
    quietly skip the exact condition a `client.stream(...)` call site is in.
    """
    return httpx.Response(
        PERMANENT_STATUS, content=iter([UPSTREAM_BODY.encode()]), request=request
    )


def _reachable(url: str = "http://provider.invalid/api/tags", **_kw) -> httpx.Response:
    return httpx.Response(200, json={"models": []}, request=httpx.Request("GET", url))


# --- one driver per migrated call site --------------------------------------


def _ollama_embed_probe(monkeypatch) -> BaseException:
    """providers/embed/ollama_embed.py — the reachability probe."""
    monkeypatch.setattr("httpx.get", _rejection)
    provider = OllamaEmbedProvider(base_url="http://provider.invalid")
    with pytest.raises(RuntimeError) as caught:
        provider._check_ollama()
    return caught.value


def _ollama_embed_call(monkeypatch) -> BaseException:
    """providers/embed/ollama_embed.py — the embeddings call."""
    monkeypatch.setattr("httpx.get", _reachable)
    monkeypatch.setattr("httpx.post", _rejection)
    provider = OllamaEmbedProvider(base_url="http://provider.invalid")
    with pytest.raises(httpx.HTTPStatusError) as caught:
        provider._call_embeddings(["hello"])
    return caught.value


def _ollama_llm_probe(monkeypatch) -> BaseException:
    """providers/llm/ollama_llm.py — the reachability probe."""
    monkeypatch.setattr("httpx.get", _rejection)
    provider = OllamaGenerator(base_url="http://provider.invalid")
    with pytest.raises(RuntimeError) as caught:
        provider._check_ollama()
    return caught.value


def _ollama_llm_generate(monkeypatch) -> BaseException:
    """providers/llm/ollama_llm.py — the chat call."""
    monkeypatch.setattr("httpx.get", _reachable)
    monkeypatch.setattr("httpx.post", _rejection)
    provider = OllamaGenerator(base_url="http://provider.invalid")
    provider._model_loaded = True  # skip the warmup round-trip
    with pytest.raises(httpx.HTTPStatusError) as caught:
        provider.generate("summarize this")
    return caught.value


def _ollama_vision(monkeypatch) -> BaseException:
    """providers/ocr/ollama_vision.py — a STREAMED response, whose body is not
    materialized until something reads it."""
    monkeypatch.setattr(
        "providers.ocr.ollama_vision.httpx.HTTPTransport",
        lambda **_kw: httpx.MockTransport(_streamed_rejection),
    )
    provider = OllamaVisionOCR(base_url="http://provider.invalid")
    with pytest.raises(httpx.HTTPStatusError) as caught:
        provider._request("aGk=", "describe this", 64)
    return caught.value


def _deepseek_ocr(monkeypatch) -> BaseException:
    """providers/ocr/deepseek_ocr2_local.py — inside call_with_retry."""
    monkeypatch.setattr("providers.ocr.deepseek_ocr2_local.httpx.post", _rejection)
    provider = DeepSeekOCR2Local(base_url="http://provider.invalid", attempts=1)
    with pytest.raises(httpx.HTTPStatusError) as caught:
        provider._send("/extract", _IMAGE)
    return caught.value


def _litellm_fallback(monkeypatch) -> BaseException:
    """providers/fallback/litellm_fallback.py — re-wrapped as TransientError so a
    misconfigured fallback self-heals (#0251); the reason rides along in the text."""
    monkeypatch.setattr(
        "providers.fallback.litellm_fallback.httpx.post",
        lambda url, **kw: _rejection(url),
    )
    fallback = LiteLLMFallback(
        "http://provider.invalid/v1", "m", "describe", image_encoder,
        api_key="k", attempts=1,
    )
    with pytest.raises(Exception) as caught:
        fallback.run(str(_IMAGE))
    return caught.value


def _openrouter_media(monkeypatch) -> BaseException:
    """providers/media/openrouter_media.py — the audio/video chat call."""
    monkeypatch.setattr(
        "providers.media.openrouter_media.httpx.post",
        lambda url, **kw: _rejection(url),
    )
    provider = OpenRouterMediaProvider(
        audio_models=["a"], video_model="v", api_key="k",
    )
    with pytest.raises(httpx.HTTPStatusError) as caught:
        provider._chat("a", [{"type": "text", "text": "hi"}])
    return caught.value


def _deepinfra_rerank(monkeypatch) -> BaseException:
    """search_hybrid.py — the reranker, re-wrapped as RuntimeError."""
    monkeypatch.setattr("httpx.post", _rejection)
    reranker = DeepInfraReranker(api_key="k")
    hit = SearchHit(doc_id="d", loc="1", snippet="s", text="t", score=1.0)
    with pytest.raises(RuntimeError) as caught:
        reranker.rerank("q", [hit])
    return caught.value


def _baseten_embed(monkeypatch) -> BaseException:
    """providers/embed/baseten_embed.py — replaces a hand-rolled log-and-re-raise."""
    monkeypatch.setattr("providers.embed.baseten_embed.httpx.post", _rejection)
    provider = BasetenEmbedProvider(model_id="abc", api_key="k")
    with pytest.raises(httpx.HTTPStatusError) as caught:
        provider._call_embeddings(["hello"])
    return caught.value


def _baseten_llm(monkeypatch) -> BaseException:
    """providers/llm/baseten_llm.py — replaces a hand-rolled log-and-re-raise."""
    monkeypatch.setattr("providers.llm.baseten_llm.httpx.post", _rejection)
    provider = BasetenGenerator(model_id="abc", api_key="k")
    with pytest.raises(httpx.HTTPStatusError) as caught:
        provider.generate("summarize this")
    return caught.value


def _litellm_llm(monkeypatch) -> BaseException:
    """providers/llm/litellm_llm.py — the enrichment route.

    This is the call site the 2026-08-27 reconciliation found still discarding
    the reason in production: `LLM enrichment failed for '<...>':
    HTTPStatusError: Client error '400 Bad Request' for url
    'http://host.docker.internal:4000/v1/chat/completions'` on 2026-08-26 06:28
    and 2026-08-27 02:02, with nothing naming which limit the document broke.
    """
    monkeypatch.setattr("providers.llm.litellm_llm.httpx.post", _rejection)
    monkeypatch.setattr("providers.llm.litellm_llm.time.sleep", lambda _s: None)
    generator = LiteLLMGenerator(
        model="qwen-bulk", base_url="http://provider.invalid/v1", api_key="k",
    )
    with pytest.raises(httpx.HTTPStatusError) as caught:
        generator.generate("summarize this")
    return caught.value


def _openrouter_llm(monkeypatch) -> BaseException:
    """providers/llm/openrouter_llm.py — the same shape, same omission."""
    monkeypatch.setattr("providers.llm.openrouter_llm.httpx.post", _rejection)
    monkeypatch.setattr("providers.llm.openrouter_llm.time.sleep", lambda _s: None)
    generator = OpenRouterGenerator(model="openai/gpt-4.1-mini", api_key="k")
    with pytest.raises(httpx.HTTPStatusError) as caught:
        generator.generate("summarize this")
    return caught.value


MIGRATED_CALL_SITES = {
    "ollama_embed:probe": _ollama_embed_probe,
    "ollama_embed:embeddings": _ollama_embed_call,
    "ollama_llm:probe": _ollama_llm_probe,
    "ollama_llm:chat": _ollama_llm_generate,
    "ollama_vision:stream": _ollama_vision,
    "deepseek_ocr2_local:send": _deepseek_ocr,
    "litellm_fallback:run": _litellm_fallback,
    "openrouter_media:chat": _openrouter_media,
    "deepinfra_reranker:rerank": _deepinfra_rerank,
    "baseten_embed:embeddings": _baseten_embed,
    "baseten_llm:generate": _baseten_llm,
    "litellm_llm:chat": _litellm_llm,
    "openrouter_llm:chat": _openrouter_llm,
}

_IMAGE = None  # set by the fixture below; the file-taking providers need a real path


@pytest.fixture(autouse=True)
def _image_file(tmp_path):
    global _IMAGE
    _IMAGE = tmp_path / "page.png"
    _IMAGE.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    yield
    _IMAGE = None


@pytest.mark.parametrize("call_site", sorted(MIGRATED_CALL_SITES))
def test_permanent_rejection_carries_the_upstream_reason(call_site, monkeypatch):
    """The body is the only actionable fact about a permanently skipped document."""
    exc = MIGRATED_CALL_SITES[call_site](monkeypatch)

    assert UPSTREAM_REASON in str(exc), (
        f"{call_site} discarded the upstream body: {exc}"
    )


@pytest.mark.parametrize("call_site", sorted(MIGRATED_CALL_SITES))
def test_enriched_message_still_parses_to_its_true_http_status(call_site, monkeypatch):
    """The enriched line lands in indexer.log, which deep health parses with an
    ordered regex tuple — one pattern of which reads `"code": NNN` out of JSON.
    httpx's own `Client error 'NNN'` text has to stay in the message and keep
    winning that race, or a rejected document reads as a provider outage."""
    exc = MIGRATED_CALL_SITES[call_site](monkeypatch)
    line = (
        "2026-08-26 12:00:00,000 ERROR prefect.flow_runs: Skipping "
        f"comm_messages::zoho_mail/x after retries exhausted: {exc}"
    )

    assert mcp_server._http_status_from_log_line(line) == PERMANENT_STATUS
    assert mcp_server._provider_failure_kind(line) is None


def test_the_table_covers_every_migrated_call_site():
    """Guard the table itself: #1662 migrated nine bare call sites across seven
    provider modules, plus the two baseten blocks that hand-rolled the same idea.
    The 2026-08-27 reconciliation added the two chat-completions sites #1662 left
    behind — litellm_llm, which serves enrichment and was observably discarding
    400 bodies in production, and openrouter_llm, which is the same code shape."""
    assert len(MIGRATED_CALL_SITES) == 13
