import json
import os
from urllib.error import HTTPError
from unittest.mock import Mock, patch

from hooks.dispatcher import dispatch_event, matching_hooks
from hooks.http import HookSendResult


def _event():
    return {
        "event": "document.indexed",
        "doc_id": "documents::000hF",
        "metadata": {"enr_summary": "safe"},
    }


def test_dispatch_event_skips_when_disabled():
    sender = Mock()

    warnings = dispatch_event({"enabled": False, "hooks": [{"name": "h", "type": "http"}]}, _event(), sender=sender)

    assert warnings == []
    sender.assert_not_called()


def test_dispatch_event_skips_unmatched_events():
    sender = Mock()
    config = {"enabled": True, "hooks": [{"name": "h", "type": "http", "events": ["other.event"]}]}

    warnings = dispatch_event(config, _event(), sender=sender)

    assert warnings == []
    sender.assert_not_called()


def test_dispatch_event_posts_json_to_http_hook():
    sender = Mock(return_value=HookSendResult(True, "accepted", False))
    config = {"enabled": True, "hooks": [{"name": "h", "type": "http", "url": "http://hook"}]}

    warnings = dispatch_event(config, _event(), sender=sender)

    assert warnings == []
    sender.assert_called_once()
    hook, event = sender.call_args.args
    assert hook["url"] == "http://hook"
    assert event["event"] == "document.indexed"


def test_dispatch_event_returns_warning_when_sender_raises():
    def sender(_hook, _event):
        raise RuntimeError("sender broke")

    config = {"enabled": True, "hooks": [{"name": "h", "type": "http", "url": "http://hook"}]}

    warnings = dispatch_event(config, _event(), sender=sender)

    assert warnings == ["hook h failed: transport_error"]


def test_dispatch_event_continues_after_one_hook_fails():
    sender = Mock(side_effect=[
        HookSendResult(False, "transport_error", True),
        HookSendResult(True, "accepted", False),
    ])
    config = {
        "enabled": True,
        "hooks": [
            {"name": "first", "type": "http", "url": "http://first"},
            {"name": "second", "type": "http", "url": "http://second"},
        ],
    }

    warnings = dispatch_event(config, _event(), sender=sender)

    assert warnings == ["hook first failed: transport_error"]
    assert sender.call_count == 2


def test_send_http_event_sends_json_and_secret_header():
    from hooks.http import send_http_event

    request_holder = {}
    response = Mock()
    response.status = 204
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=None)

    def fake_urlopen(request, timeout):
        request_holder["request"] = request
        request_holder["timeout"] = timeout
        return response

    hook = {
        "name": "h",
        "url": "http://hook",
        "timeout_seconds": 7,
        "secret_env": "HOOK_SECRET",
    }

    with patch.dict(os.environ, {"HOOK_SECRET": "secret-value"}):
        with patch("urllib.request.urlopen", fake_urlopen):
            result = send_http_event(hook, _event())

    request = request_holder["request"]
    assert result == HookSendResult(True, "accepted", False, http_status=204)
    assert request.full_url == "http://hook"
    assert request.get_method() == "POST"
    assert request.headers["Content-type"] == "application/json"
    assert request.headers["X-rag-hook-secret"] == "secret-value"
    assert request_holder["timeout"] == 7
    assert json.loads(request.data.decode("utf-8"))["doc_id"] == "documents::000hF"


def test_send_http_event_returns_warning_on_failure():
    from hooks.http import send_http_event

    with patch("urllib.request.urlopen", side_effect=OSError("boom")):
        result = send_http_event({"name": "h", "url": "http://hook"}, _event())

    assert result == HookSendResult(False, "transport_error", True, error="HTTP request failed")


def test_send_http_event_returns_warning_when_secret_missing():
    from hooks.http import send_http_event

    with patch.dict(os.environ, {}, clear=True):
        result = send_http_event({"name": "h", "url": "http://hook", "secret_env": "MISSING"}, _event())

    assert result == HookSendResult(False, "configuration_error", False, error="missing hook secret")


def test_send_http_event_returns_warning_for_invalid_timeout():
    from hooks.http import send_http_event

    result = send_http_event({"name": "h", "url": "http://hook", "timeout_seconds": "bad"}, _event())

    assert result == HookSendResult(False, "configuration_error", False, error="invalid timeout")


def test_send_http_event_returns_warning_for_malformed_url():
    from hooks.http import send_http_event

    result = send_http_event({"name": "h", "url": "://bad"}, _event())

    assert result == HookSendResult(False, "transport_error", True, error="HTTP request failed")


def test_send_http_event_resolves_env_var_url():
    """url: ${VAR} is expanded from the environment before POSTing."""
    from hooks.http import send_http_event

    request_holder = {}
    response = Mock()
    response.status = 200
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=None)

    def fake_urlopen(request, timeout):
        request_holder["url"] = request.full_url
        return response

    hook = {"name": "cds-callback", "url": "${CDS_HOOK_URL}"}
    with patch.dict(os.environ, {"CDS_HOOK_URL": "http://host.docker.internal:8095/hooks/doc-indexed"}):
        with patch("urllib.request.urlopen", fake_urlopen):
            result = send_http_event(hook, _event())

    assert result == HookSendResult(True, "accepted", False, http_status=200)
    assert request_holder["url"] == "http://host.docker.internal:8095/hooks/doc-indexed"


def test_send_http_event_env_var_url_unset_is_silent_noop():
    """An env-driven url whose var is unset is skipped silently — no warning,
    no POST — so an optional callback can't spam a warning per indexed doc."""
    from hooks.http import send_http_event

    called = {"n": 0}

    def fake_urlopen(request, timeout):  # pragma: no cover - must not run
        called["n"] += 1
        raise AssertionError("should not POST when env url is unset")

    with patch.dict(os.environ, {}, clear=True):
        with patch("urllib.request.urlopen", fake_urlopen):
            result = send_http_event({"name": "cds-callback", "url": "${CDS_HOOK_URL}"}, _event())

    assert result == HookSendResult(True, "disabled", False)
    assert called["n"] == 0


def test_send_http_event_literal_empty_url_still_warns():
    """A genuinely missing (non-env) url is still a warning, not a silent skip."""
    from hooks.http import send_http_event

    result = send_http_event({"name": "h", "url": ""}, _event())

    assert result == HookSendResult(False, "configuration_error", False, error="missing url")


def test_send_http_event_rejects_cds_no_match_even_with_http_200(monkeypatch):
    from hooks.http import send_http_event

    class Response:
        status = 200

        def read(self):
            return b'{"status":"no_match","detail":"private document text"}'

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("hooks.http.urllib.request.urlopen", lambda *args, **kwargs: Response())
    result = send_http_event(
        {"name": "cds", "url": "http://hook", "accepted_statuses": ["updated", "duplicate"]},
        {"event": "document.indexed", "event_id": "evt-1"},
    )

    assert result.accepted is False
    assert result.outcome == "no_match"
    assert result.retryable is False
    assert result.error is None


def test_matching_hooks_returns_only_valid_matching_http_hooks():
    config = {
        "enabled": True,
        "hooks": [
            {"name": "match", "type": "http", "url": "http://match", "events": ["document.indexed"]},
            {"name": "other-event", "type": "http", "url": "http://other", "events": ["other.event"]},
            {"name": "other-type", "type": "queue", "url": "http://queue"},
            "not-a-hook",
        ],
    }

    assert matching_hooks(config, "document.indexed") == [
        {"name": "match", "type": "http", "url": "http://match", "events": ["document.indexed"]}
    ]


def test_send_http_event_classifies_http_errors_by_status(monkeypatch):
    from hooks.http import send_http_event

    def raise_http_error(_request, *, timeout):
        raise HTTPError("http://hook", 404, "not found", hdrs=None, fp=None)

    monkeypatch.setattr("hooks.http.urllib.request.urlopen", raise_http_error)

    result = send_http_event({"name": "h", "url": "http://hook"}, _event())

    assert result == HookSendResult(False, "http_error", False, http_status=404, error="HTTP request failed")


def test_send_http_event_retries_server_http_error(monkeypatch):
    from hooks.http import send_http_event

    def raise_http_error(_request, *, timeout):
        raise HTTPError("http://hook", 503, "unavailable", hdrs=None, fp=None)

    monkeypatch.setattr("hooks.http.urllib.request.urlopen", raise_http_error)

    result = send_http_event({"name": "h", "url": "http://hook"}, _event())

    assert result == HookSendResult(False, "http_error", True, http_status=503, error="HTTP request failed")


def test_send_http_event_rejects_terminal_status_even_if_configured_accepted(monkeypatch):
    from hooks.http import send_http_event

    class Response:
        status = 200

        def read(self):
            return b'{"status":"no_match"}'

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("hooks.http.urllib.request.urlopen", lambda *args, **kwargs: Response())

    result = send_http_event(
        {"name": "cds", "url": "http://hook", "accepted_statuses": ["no_match"]},
        {"event": "document.indexed", "event_id": "evt-1"},
    )

    assert result == HookSendResult(False, "no_match", False, http_status=200)
