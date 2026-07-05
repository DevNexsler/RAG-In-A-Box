"""The file_status reranker health check must honor search.reranker.base_url.

Regression guard for the staging stack: with a base_url override in config
(e.g. the provider-sim), the health check must NOT ping production DeepInfra
with a real key on every status call. With no override, behavior is unchanged.
"""
import httpx

import mcp_server


class _FakeStore:
    def list_doc_ids(self):
        return []

    def count_chunks(self):
        return 0

    def fts_available(self):
        return True

    def _metadata_subfields(self):
        return set()


def _run_status_capture_url(monkeypatch, tmp_path, reranker_cfg):
    config = {
        "index_root": str(tmp_path),
        "embeddings": {"provider": "openrouter"},
        "search": {"reranker": reranker_cfg},
    }
    captured = {}

    class _Resp:
        status_code = 200

    def fake_post(url, **kwargs):
        captured["url"] = url
        return _Resp()

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(mcp_server, "_get_deps", lambda: (_FakeStore(), None, config))
    monkeypatch.setattr(mcp_server, "_get_deep_health", lambda **kwargs: {})

    result = mcp_server._file_status_impl()
    return captured, result


def test_health_check_honors_base_url_override(monkeypatch, tmp_path):
    captured, result = _run_status_capture_url(
        monkeypatch, tmp_path,
        {"enabled": True, "provider": "deepinfra",
         "model": "Qwen/Qwen3-Reranker-8B",
         "base_url": "http://provider-sim:9999"},
    )
    assert captured["url"] == (
        "http://provider-sim:9999/v1/inference/Qwen/Qwen3-Reranker-8B"
    )
    assert result["health"]["reranker_responsive"] is True


def test_health_check_strips_trailing_slash_on_base_url(monkeypatch, tmp_path):
    captured, _ = _run_status_capture_url(
        monkeypatch, tmp_path,
        {"enabled": True, "model": "m", "base_url": "http://provider-sim:9999/"},
    )
    assert captured["url"] == "http://provider-sim:9999/v1/inference/m"


def test_health_check_defaults_to_deepinfra_when_unset(monkeypatch, tmp_path):
    captured, _ = _run_status_capture_url(
        monkeypatch, tmp_path,
        {"enabled": True, "model": "Qwen/Qwen3-Reranker-8B"},
    )
    assert captured["url"] == (
        "https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Reranker-8B"
    )
