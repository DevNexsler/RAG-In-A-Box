import httpx, pytest
from core.resilience import is_transient
from providers.fallback.litellm_fallback import LiteLLMFallback, image_encoder


def _resp(content):
    return {"choices": [{"message": {"content": content}}]}


def test_returns_text(monkeypatch, tmp_path):
    img = tmp_path / "a.png"; img.write_bytes(b"\x89PNG\r\n")
    def fake_post(url, **kw):
        return httpx.Response(200, json=_resp("a description"), request=httpx.Request("POST", url))
    monkeypatch.setattr("providers.fallback.litellm_fallback.httpx.post", fake_post)
    fb = LiteLLMFallback("http://lite/v1", "m", "describe", image_encoder, api_key="k")
    assert fb.run(str(img)) == "a description"


def test_reachable_empty_returns_empty(monkeypatch, tmp_path):
    img = tmp_path / "a.png"; img.write_bytes(b"x")
    def fake_post(url, **kw):
        return httpx.Response(200, json=_resp("   "), request=httpx.Request("POST", url))
    monkeypatch.setattr("providers.fallback.litellm_fallback.httpx.post", fake_post)
    fb = LiteLLMFallback("http://lite/v1", "m", "p", image_encoder, api_key="k")
    assert fb.run(str(img)) == ""


def test_unreachable_raises_transient(monkeypatch, tmp_path):
    img = tmp_path / "a.png"; img.write_bytes(b"x")
    def fake_post(url, **kw):
        raise httpx.ConnectError("refused", request=httpx.Request("POST", url))
    monkeypatch.setattr("providers.fallback.litellm_fallback.httpx.post", fake_post)
    fb = LiteLLMFallback("http://lite/v1", "m", "p", image_encoder, api_key="k",
                         attempts=1)
    with pytest.raises(Exception) as e:
        fb.run(str(img))
    assert is_transient(e.value)


def test_misconfigured_4xx_becomes_transient(monkeypatch, tmp_path):
    # A 401/404 (bad key / wrong model) is non-transient by default; the fallback
    # client MUST coerce it to transient so a config error retries, never caps (#0251).
    img = tmp_path / "a.png"; img.write_bytes(b"x")
    def fake_post(url, **kw):
        return httpx.Response(401, json={"error": "bad key"}, request=httpx.Request("POST", url))
    monkeypatch.setattr("providers.fallback.litellm_fallback.httpx.post", fake_post)
    fb = LiteLLMFallback("http://lite/v1", "m", "p", image_encoder, api_key="k", attempts=1)
    with pytest.raises(Exception) as e:
        fb.run(str(img))
    assert is_transient(e.value)


def test_malformed_body_becomes_transient(monkeypatch, tmp_path):
    img = tmp_path / "a.png"; img.write_bytes(b"x")
    def fake_post(url, **kw):
        return httpx.Response(200, json={"unexpected": "shape"}, request=httpx.Request("POST", url))
    monkeypatch.setattr("providers.fallback.litellm_fallback.httpx.post", fake_post)
    fb = LiteLLMFallback("http://lite/v1", "m", "p", image_encoder, api_key="k", attempts=1)
    with pytest.raises(Exception) as e:
        fb.run(str(img))
    assert is_transient(e.value)
