"""Tests for PrefectServer context manager.

All tests mock subprocess/httpx so no real Prefect server is needed.
"""

import os
import signal
import subprocess
from unittest.mock import MagicMock, patch

import httpx
import pytest

from prefect_server import API_URL, HEALTH_URL, PrefectServer


# -- Helpers --

def _healthy_response():
    """Fake 200 response for health check."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    return resp


def _make_mock_popen():
    """Fake Popen that stays 'running' (poll returns None)."""
    proc = MagicMock()
    proc.poll.return_value = None  # still running
    proc.pid = 12345
    proc.returncode = None
    return proc


# -- Tests: reuse existing server --

class TestReuseExistingServer:
    """When Prefect server is already healthy, reuse it without starting one."""

    @patch("prefect_server.httpx.get", return_value=_healthy_response())
    def test_sets_api_url(self, mock_get):
        old = os.environ.pop("PREFECT_API_URL", None)
        try:
            with PrefectServer():
                assert os.environ["PREFECT_API_URL"] == API_URL
        finally:
            if old is not None:
                os.environ["PREFECT_API_URL"] = old
            else:
                os.environ.pop("PREFECT_API_URL", None)

    @patch("prefect_server.httpx.get", return_value=_healthy_response())
    def test_restores_original_api_url(self, mock_get):
        os.environ["PREFECT_API_URL"] = "http://original:9999/api"
        try:
            with PrefectServer():
                assert os.environ["PREFECT_API_URL"] == API_URL
            assert os.environ["PREFECT_API_URL"] == "http://original:9999/api"
        finally:
            os.environ.pop("PREFECT_API_URL", None)

    @patch("prefect_server.httpx.get", return_value=_healthy_response())
    def test_removes_api_url_if_was_unset(self, mock_get):
        os.environ.pop("PREFECT_API_URL", None)
        with PrefectServer():
            pass
        assert "PREFECT_API_URL" not in os.environ

    @patch("prefect_server.subprocess.Popen")
    @patch("prefect_server.httpx.get", return_value=_healthy_response())
    def test_does_not_start_subprocess(self, mock_get, mock_popen):
        with PrefectServer():
            pass
        mock_popen.assert_not_called()


# -- Tests: start new server --

class TestStartServer:
    """When no server is running, start one and stop it on exit."""

    def _get_side_effect(self, fail_count=2):
        """First `fail_count` calls raise ConnectError, then return healthy."""
        calls = {"n": 0}
        def side_effect(url, **kwargs):
            calls["n"] += 1
            if calls["n"] <= fail_count:
                raise httpx.ConnectError("refused")
            return _healthy_response()
        return side_effect

    @patch("prefect_server.signal.signal")
    @patch("prefect_server.time.sleep")
    @patch("prefect_server.subprocess.Popen")
    @patch("prefect_server.httpx.get")
    def test_starts_and_stops_subprocess(self, mock_get, mock_popen, mock_sleep, mock_signal):
        mock_get.side_effect = self._get_side_effect(fail_count=2)
        proc = _make_mock_popen()
        mock_popen.return_value = proc

        os.environ.pop("PREFECT_API_URL", None)
        try:
            with PrefectServer():
                assert os.environ["PREFECT_API_URL"] == API_URL
                mock_popen.assert_called_once()
                # Verify --no-services flag
                cmd = mock_popen.call_args[0][0]
                assert "--no-services" in cmd
            # After exit, terminate was called
            proc.terminate.assert_called_once()
        finally:
            os.environ.pop("PREFECT_API_URL", None)

    @patch("prefect_server.signal.signal")
    @patch("prefect_server.time.sleep")
    @patch("prefect_server.subprocess.Popen")
    @patch("prefect_server.httpx.get")
    def test_polls_until_healthy(self, mock_get, mock_popen, mock_sleep, mock_signal):
        mock_get.side_effect = self._get_side_effect(fail_count=3)
        mock_popen.return_value = _make_mock_popen()

        os.environ.pop("PREFECT_API_URL", None)
        try:
            with PrefectServer():
                pass
            # 1 initial check (not healthy) + 3 poll calls until healthy = 4 total
            assert mock_get.call_count == 4
            # sleep called between failed polls (not after the successful one)
            assert mock_sleep.call_count == 2
        finally:
            os.environ.pop("PREFECT_API_URL", None)


# -- Tests: error cases --

class TestErrorCases:
    """Timeout and early-exit errors."""

    @patch("prefect_server.STARTUP_TIMEOUT", 0.1)
    @patch("prefect_server.time.sleep")
    @patch("prefect_server.subprocess.Popen")
    @patch("prefect_server.httpx.get", side_effect=httpx.ConnectError("refused"))
    @patch("prefect_server.signal.signal")
    def test_timeout_raises(self, mock_signal, mock_get, mock_popen, mock_sleep):
        proc = _make_mock_popen()
        mock_popen.return_value = proc

        os.environ.pop("PREFECT_API_URL", None)
        with pytest.raises(TimeoutError, match="did not become healthy"):
            with PrefectServer():
                pass

    @patch("prefect_server.signal.signal")
    @patch("prefect_server.time.sleep")
    @patch("prefect_server.subprocess.Popen")
    @patch("prefect_server.httpx.get", side_effect=httpx.ConnectError("refused"))
    def test_subprocess_early_exit_raises(self, mock_get, mock_popen, mock_sleep, mock_signal):
        proc = _make_mock_popen()
        proc.poll.return_value = 1  # exited immediately
        proc.returncode = 1
        mock_popen.return_value = proc

        os.environ.pop("PREFECT_API_URL", None)
        with pytest.raises(RuntimeError, match="exited with code 1"):
            with PrefectServer():
                pass


# -- Tests: cleanup safety --

class TestCleanupSafety:
    """atexit and SIGTERM handler registration."""

    @patch("prefect_server.atexit.register")
    @patch("prefect_server.signal.signal")
    @patch("prefect_server.time.sleep")
    @patch("prefect_server.subprocess.Popen")
    @patch("prefect_server.httpx.get")
    def test_registers_atexit(self, mock_get, mock_popen, mock_sleep, mock_signal, mock_atexit):
        # First call: not healthy (triggers start), second call: healthy
        calls = {"n": 0}
        def side_effect(url, **kwargs):
            calls["n"] += 1
            if calls["n"] <= 1:
                raise httpx.ConnectError("refused")
            return _healthy_response()
        mock_get.side_effect = side_effect
        mock_popen.return_value = _make_mock_popen()

        os.environ.pop("PREFECT_API_URL", None)
        try:
            with PrefectServer():
                mock_atexit.assert_called_once()
        finally:
            os.environ.pop("PREFECT_API_URL", None)

    @patch("prefect_server.atexit.register")
    @patch("prefect_server.signal.signal")
    @patch("prefect_server.time.sleep")
    @patch("prefect_server.subprocess.Popen")
    @patch("prefect_server.httpx.get")
    def test_registers_sigterm_handler(self, mock_get, mock_popen, mock_sleep, mock_signal, mock_atexit):
        calls = {"n": 0}
        def side_effect(url, **kwargs):
            calls["n"] += 1
            if calls["n"] <= 1:
                raise httpx.ConnectError("refused")
            return _healthy_response()
        mock_get.side_effect = side_effect
        mock_popen.return_value = _make_mock_popen()

        os.environ.pop("PREFECT_API_URL", None)
        try:
            with PrefectServer():
                # Check signal.SIGTERM was registered with a callable handler
                sigterm_calls = [c for c in mock_signal.call_args_list if c[0][0] == signal.SIGTERM]
                assert len(sigterm_calls) == 1
                handler = sigterm_calls[0][0][1]
                assert callable(handler)
        finally:
            os.environ.pop("PREFECT_API_URL", None)

    @patch("prefect_server.httpx.get", return_value=_healthy_response())
    def test_stop_is_noop_when_reusing(self, mock_get):
        """When we didn't start the server, _stop does nothing."""
        ps = PrefectServer()
        ps.__enter__()
        assert not ps._we_started
        assert ps._process is None
        ps.__exit__(None, None, None)
        # No error — clean no-op


# -- Tests: log file instead of DEVNULL (Fix 5) --

class TestLogFile:
    """Subprocess output should go to a log file, not DEVNULL."""

    @patch("prefect_server.signal.signal")
    @patch("prefect_server.time.sleep")
    @patch("prefect_server.subprocess.Popen")
    @patch("prefect_server.httpx.get")
    def test_popen_uses_log_file_not_devnull(self, mock_get, mock_popen, mock_sleep, mock_signal):
        calls = {"n": 0}
        def side_effect(url, **kwargs):
            calls["n"] += 1
            if calls["n"] <= 1:
                raise httpx.ConnectError("refused")
            return _healthy_response()
        mock_get.side_effect = side_effect
        mock_popen.return_value = _make_mock_popen()

        os.environ.pop("PREFECT_API_URL", None)
        try:
            with PrefectServer() as ps:
                popen_kwargs = mock_popen.call_args[1]
                assert popen_kwargs.get("stdout") is not subprocess.DEVNULL
                assert popen_kwargs.get("stderr") is not subprocess.DEVNULL
                # Should be a file handle, not None
                assert popen_kwargs.get("stdout") is not None
        finally:
            os.environ.pop("PREFECT_API_URL", None)
