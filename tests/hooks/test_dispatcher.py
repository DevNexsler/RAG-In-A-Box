import json
import os
from unittest.mock import Mock, patch

from hooks.dispatcher import dispatch_event


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
    sender = Mock(return_value=None)
    config = {"enabled": True, "hooks": [{"name": "h", "type": "http", "url": "http://hook"}]}

    warnings = dispatch_event(config, _event(), sender=sender)

    assert warnings == []
    sender.assert_called_once()
    hook, event = sender.call_args.args
    assert hook["url"] == "http://hook"
    assert event["event"] == "document.indexed"


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
            warning = send_http_event(hook, _event())

    request = request_holder["request"]
    assert warning is None
    assert request.full_url == "http://hook"
    assert request.get_method() == "POST"
    assert request.headers["Content-type"] == "application/json"
    assert request.headers["X-rag-hook-secret"] == "secret-value"
    assert request_holder["timeout"] == 7
    assert json.loads(request.data.decode("utf-8"))["doc_id"] == "documents::000hF"


def test_send_http_event_returns_warning_on_failure():
    from hooks.http import send_http_event

    with patch("urllib.request.urlopen", side_effect=OSError("boom")):
        warning = send_http_event({"name": "h", "url": "http://hook"}, _event())

    assert "hook h failed" in warning
    assert "boom" in warning


def test_send_http_event_returns_warning_when_secret_missing():
    from hooks.http import send_http_event

    with patch.dict(os.environ, {}, clear=True):
        warning = send_http_event({"name": "h", "url": "http://hook", "secret_env": "MISSING"}, _event())

    assert warning == "hook h disabled: secret env MISSING is not set"


def test_send_http_event_returns_warning_for_invalid_timeout():
    from hooks.http import send_http_event

    warning = send_http_event({"name": "h", "url": "http://hook", "timeout_seconds": "bad"}, _event())

    assert warning == "hook h disabled: invalid timeout_seconds 'bad'"


def test_send_http_event_returns_warning_for_malformed_url():
    from hooks.http import send_http_event

    warning = send_http_event({"name": "h", "url": "://bad"}, _event())

    assert warning is not None
    assert warning.startswith("hook h failed:")
    assert "unknown url type" in warning
