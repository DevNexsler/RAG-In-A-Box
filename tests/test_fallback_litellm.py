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


# A LiteLLM 400 whose body is the omni engine's answer for an undecodable mp4
# (#4387: documents::003AL, a file with no moov atom, answered this 46 times).
_UNDECODABLE_VIDEO_BODY = {"error": {"message": (
    "litellm.BadRequestError: Openai_likeException - Could not open video "
    "stream. Received Model Group=video"), "code": "400"}}


@pytest.mark.parametrize("status", [400, 413, 422])
def test_input_rejection_stays_permanent(monkeypatch, tmp_path, status):
    # 400/413/422 judge the request's own payload (#0705): the same bytes get
    # the same answer, so retrying them every run can never heal the doc.
    img = tmp_path / "a.png"; img.write_bytes(b"x")
    def fake_post(url, **kw):
        return httpx.Response(status, json=_UNDECODABLE_VIDEO_BODY,
                              request=httpx.Request("POST", url))
    monkeypatch.setattr("providers.fallback.litellm_fallback.httpx.post", fake_post)
    fb = LiteLLMFallback("http://lite/v1", "m", "p", image_encoder, api_key="k", attempts=1)
    with pytest.raises(httpx.HTTPStatusError) as e:
        fb.run(str(img))
    assert e.value.response.status_code == status
    assert not is_transient(e.value)


def test_undecodable_video_charges_the_degraded_ledger(monkeypatch, tmp_path):
    # Production shape (media.provider: litellm, LiteLLMFallback as the video
    # primary): a corrupt video must degrade as doc-specific so the ledger caps
    # it for this file version, not as an outage that retries forever (#4387).
    from extractors import begin_degradation_capture, collect_degradations, extract_video
    from providers.media import build_media_provider

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    calls = []
    def fake_post(url, **kw):
        calls.append(url)
        return httpx.Response(400, json=_UNDECODABLE_VIDEO_BODY,
                              request=httpx.Request("POST", url))
    monkeypatch.setattr("providers.fallback.litellm_fallback.httpx.post", fake_post)
    provider = build_media_provider({"media": {
        "enabled": True, "provider": "litellm", "endpoint": "http://lite/v1",
        "video_model": "video", "api_key": "k"}})
    clip = tmp_path / "clip.mp4"; clip.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)

    begin_degradation_capture()
    result = extract_video(clip, media_provider=provider)

    assert result.full_text == ""
    assert len(calls) == 1  # a rejection is not retried within the run either
    degs = collect_degradations()
    assert [d.reason for d in degs] == ["video_extract_failed"]
    assert degs[0].transient is False
