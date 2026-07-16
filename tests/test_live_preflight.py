"""Unit tests for scripts/live_preflight.py — all probes monkeypatched.

No real network, no real docker, no real postgres: each check function is
exercised against simulated success/failure conditions, and main() is checked
for failure aggregation (report everything, not just the first problem).
"""

import re
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.live_preflight as lp


# ---------------------------------------------------------------------------
# Contract: the script must end with sys.exit(main()) so a nonzero return
# from main() actually becomes a nonzero exit code (gate.py checks rc == 0).
# ---------------------------------------------------------------------------

def test_module_ends_with_sys_exit_main():
    source = Path(lp.__file__).read_text()
    assert re.search(r"sys\.exit\(main\(\)\)\s*$", source), (
        "live_preflight.py must end with sys.exit(main()) — otherwise a "
        "failing main() would exit 0 and approve the real-money live tier"
    )


# ---------------------------------------------------------------------------
# check_api_keys
# ---------------------------------------------------------------------------

def test_api_keys_present(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "di-x")
    ok, reason = lp.check_api_keys()
    assert ok


@pytest.mark.parametrize("missing", ["OPENROUTER_API_KEY", "DEEPINFRA_API_KEY"])
def test_api_keys_missing(monkeypatch, missing):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "di-x")
    monkeypatch.delenv(missing)
    ok, reason = lp.check_api_keys()
    assert not ok
    assert missing in reason


def test_api_keys_empty_string_fails(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "di-x")
    ok, reason = lp.check_api_keys()
    assert not ok
    assert "OPENROUTER_API_KEY" in reason


# ---------------------------------------------------------------------------
# check_config_test
# ---------------------------------------------------------------------------

def test_config_test_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config_test.yaml").write_text("vault_root: ./v\n")
    ok, reason = lp.check_config_test()
    assert ok


def test_config_test_missing_hints_copy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ok, reason = lp.check_config_test()
    assert not ok
    assert "config_test.yaml" in reason
    assert "copy" in reason.lower()  # actionable hint: copy from main checkout


# ---------------------------------------------------------------------------
# check_mac_ocr
# ---------------------------------------------------------------------------

def _write_config(path, body):
    path.write_text(body)
    return path


def test_mac_ocr_reachable(tmp_path, monkeypatch):
    cfg = _write_config(tmp_path / "config.yaml",
                        'ocr:\n  base_url: "http://mac:8790"\n')
    monkeypatch.setattr(lp, "config_candidates", lambda: [cfg])
    calls = {}

    def fake_get(url, timeout=None, **kw):
        calls["url"] = url
        return SimpleNamespace(status_code=404)  # any response = reachable

    monkeypatch.setattr(lp.httpx, "get", fake_get)
    ok, reason = lp.check_mac_ocr()
    assert ok
    assert calls["url"] == "http://mac:8790"


def test_mac_ocr_accepts_legacy_endpoint_key(tmp_path, monkeypatch):
    cfg = _write_config(
        tmp_path / "config.yaml",
        'ocr:\n  endpoint: "http://mac:4000/v1"\n',
    )
    monkeypatch.setattr(lp, "config_candidates", lambda: [cfg])
    calls = {}

    def fake_get(url, timeout=None, **kw):
        calls["url"] = url
        return SimpleNamespace(status_code=404)

    monkeypatch.setattr(lp.httpx, "get", fake_get)
    ok, _ = lp.check_mac_ocr()

    assert ok
    assert calls["url"] == "http://mac:4000/v1"


def test_mac_ocr_unreachable(tmp_path, monkeypatch):
    cfg = _write_config(tmp_path / "config.yaml",
                        'ocr:\n  base_url: "http://mac:8790"\n')
    monkeypatch.setattr(lp, "config_candidates", lambda: [cfg])

    def fake_get(url, timeout=None, **kw):
        raise lp.httpx.ConnectError("refused")

    monkeypatch.setattr(lp.httpx, "get", fake_get)
    ok, reason = lp.check_mac_ocr()
    assert not ok
    assert "mac:8790" in reason


def test_mac_ocr_no_base_url_in_config(tmp_path, monkeypatch):
    cfg = _write_config(tmp_path / "config.yaml", "ocr:\n  enabled: true\n")
    monkeypatch.setattr(lp, "config_candidates", lambda: [cfg])
    ok, reason = lp.check_mac_ocr()
    assert not ok
    assert "base_url" in reason


def test_mac_ocr_no_config_file(tmp_path, monkeypatch):
    monkeypatch.setattr(lp, "config_candidates",
                        lambda: [tmp_path / "nope.yaml"])
    ok, reason = lp.check_mac_ocr()
    assert not ok


# ---------------------------------------------------------------------------
# check_prod_indexer_idle
# ---------------------------------------------------------------------------

def _fake_run(stdout="", returncode=0, raise_exc=None):
    def run(cmd, **kw):
        if raise_exc is not None:
            raise raise_exc
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")
    return run


def test_indexer_heartbeat_fresh_fails(monkeypatch):
    hb = str(time.time() - 10)  # 10s old — actively writing
    monkeypatch.setattr(lp.subprocess, "run", _fake_run(stdout=hb))
    ok, reason = lp.check_prod_indexer_idle()
    assert not ok
    assert "active" in reason.lower()


def test_indexer_heartbeat_stale_passes(monkeypatch):
    hb = str(time.time() - 3600)  # an hour quiet
    monkeypatch.setattr(lp.subprocess, "run", _fake_run(stdout=hb))
    ok, reason = lp.check_prod_indexer_idle()
    assert ok


def test_indexer_heartbeat_boundary_uses_named_threshold(monkeypatch):
    # just over the threshold: idle
    hb = str(time.time() - (lp.HEARTBEAT_ACTIVE_THRESHOLD_S + 5))
    monkeypatch.setattr(lp.subprocess, "run", _fake_run(stdout=hb))
    ok, _ = lp.check_prod_indexer_idle()
    assert ok


def test_indexer_heartbeat_just_under_threshold_fails(monkeypatch):
    # just under the threshold: still active — pins the comparison direction
    hb = str(time.time() - (lp.HEARTBEAT_ACTIVE_THRESHOLD_S - 5))
    monkeypatch.setattr(lp.subprocess, "run", _fake_run(stdout=hb))
    ok, reason = lp.check_prod_indexer_idle()
    assert not ok
    assert "active" in reason.lower()


def test_indexer_container_not_running_passes(monkeypatch):
    # docker exec nonzero rc: container down or heartbeat file absent
    monkeypatch.setattr(lp.subprocess, "run",
                        _fake_run(stdout="", returncode=1))
    ok, reason = lp.check_prod_indexer_idle()
    assert ok


def test_indexer_docker_binary_missing_passes_with_warning(monkeypatch):
    monkeypatch.setattr(lp.subprocess, "run",
                        _fake_run(raise_exc=FileNotFoundError("docker")))
    ok, reason = lp.check_prod_indexer_idle()
    assert ok
    assert "docker" in reason.lower()
    assert "warn" in reason.lower()


def test_indexer_docker_timeout_passes(monkeypatch):
    exc = subprocess.TimeoutExpired(cmd=["docker"], timeout=10)
    monkeypatch.setattr(lp.subprocess, "run", _fake_run(raise_exc=exc))
    ok, reason = lp.check_prod_indexer_idle()
    assert ok


def test_indexer_unparseable_heartbeat_passes_with_note(monkeypatch):
    monkeypatch.setattr(lp.subprocess, "run", _fake_run(stdout="garbage\n"))
    ok, reason = lp.check_prod_indexer_idle()
    assert ok
    assert "unreadable" in reason.lower() or "unparse" in reason.lower()


# ---------------------------------------------------------------------------
# check_comm_postgres
# ---------------------------------------------------------------------------

def test_comm_postgres_missing_dsn_fails(monkeypatch):
    monkeypatch.delenv("COMM_DATA_STORE_DSN", raising=False)
    ok, reason = lp.check_comm_postgres()
    assert not ok
    assert "COMM_DATA_STORE_DSN" in reason


def test_comm_postgres_connect_error_fails(monkeypatch):
    monkeypatch.setenv("COMM_DATA_STORE_DSN", "postgresql://u@h/db")

    def boom(*a, **kw):
        raise lp.psycopg.OperationalError("connection refused")

    monkeypatch.setattr(lp.psycopg, "connect", boom)
    ok, reason = lp.check_comm_postgres()
    assert not ok


def test_comm_postgres_select_one_ok(monkeypatch):
    monkeypatch.setenv("COMM_DATA_STORE_DSN", "postgresql://u@h/db")

    class FakeCursor:
        def execute(self, sql):
            assert sql == "SELECT 1"
        def fetchone(self):
            return (1,)

    class FakeConn:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def cursor(self):
            return FakeCursor()

    monkeypatch.setattr(lp.psycopg, "connect", lambda *a, **kw: FakeConn())
    ok, reason = lp.check_comm_postgres()
    assert ok


# ---------------------------------------------------------------------------
# main(): aggregation — every failing check is reported, rc is 1
# ---------------------------------------------------------------------------

def _stub_checks(monkeypatch, results):
    checks = [(name, (lambda r=r: r)) for name, r in results.items()]
    monkeypatch.setattr(lp, "CHECKS", checks)
    monkeypatch.setattr(lp, "_load_env", lambda: None)


def test_main_all_pass(monkeypatch, capsys):
    _stub_checks(monkeypatch, {
        "a": (True, "fine"),
        "b": (True, "fine"),
    })
    assert lp.main() == 0
    out = capsys.readouterr().out
    assert "ok a: fine" in out
    assert "ok b: fine" in out


def test_main_reports_all_failures(monkeypatch, capsys):
    _stub_checks(monkeypatch, {
        "a": (False, "broken one"),
        "b": (True, "fine"),
        "c": (False, "broken two"),
    })
    assert lp.main() == 1
    out = capsys.readouterr().out
    # does NOT stop at the first failure: both problems reported at once
    assert "FAIL a: broken one" in out
    assert "FAIL c: broken two" in out
    assert "ok b: fine" in out


def test_main_check_exception_is_a_failure(monkeypatch, capsys):
    def boom():
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(lp, "CHECKS", [("a", boom), ("b", lambda: (True, "fine"))])
    monkeypatch.setattr(lp, "_load_env", lambda: None)
    assert lp.main() == 1
    out = capsys.readouterr().out
    assert "FAIL a:" in out and "probe exploded" in out
    assert "ok b: fine" in out


def test_real_checks_are_registered():
    names = [name for name, _ in lp.CHECKS]
    assert names == ["api_keys", "config_test", "mac_ocr",
                     "prod_indexer_idle", "comm_postgres"]
