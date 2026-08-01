"""Unit tests for scripts/live_preflight.py — all probes monkeypatched.

No real network, no real docker, no real postgres: each check function is
exercised against simulated success/failure conditions, and main() is checked
for failure aggregation (report everything, not just the first problem).
"""

import re
import subprocess
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
# check_litellm_ocr
# ---------------------------------------------------------------------------

def _write_litellm_config(path, **overrides):
    ocr = {
        "provider": "litellm",
        "endpoint": "http://litellm:4000/v1",
        "extract_model": "ocr",
        "describe_model": "vision",
    }
    ocr.update(overrides)
    path.write_text(lp.yaml.safe_dump({"ocr": ocr}))
    return path


def _http_response(status_code=200, payload=None, content=None):
    kwargs = {"request": lp.httpx.Request("GET", "http://litellm/v1/models")}
    if content is not None:
        kwargs["content"] = content
    else:
        kwargs["json"] = payload
    return lp.httpx.Response(status_code, **kwargs)


def test_config_candidates_prefer_cwd_before_main(monkeypatch):
    monkeypatch.setattr(lp, "main_checkout_root", lambda: Path("/main"))
    assert lp.config_candidates() == [Path("config.yaml"),
                                      Path("/main/config.yaml")]


@pytest.mark.parametrize(
    ("endpoint", "expected_url"),
    [
        ("http://litellm:4000", "http://litellm:4000/models"),
        ("http://litellm:4000/v1", "http://litellm:4000/v1/models"),
        ("http://litellm:4000/v1/", "http://litellm:4000/v1/models"),
    ],
)
def test_litellm_ocr_validates_both_aliases_with_auth(
        tmp_path, monkeypatch, endpoint, expected_url):
    cfg = _write_litellm_config(tmp_path / "config.yaml", endpoint=endpoint)
    monkeypatch.setattr(lp, "config_candidates", lambda: [cfg])
    monkeypatch.setenv("LITELLM_API_KEY", "test-secret-key")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "fallback-secret-key")
    calls = {}

    def fake_get(url, *, headers, timeout):
        calls.update(url=url, headers=headers, timeout=timeout)
        return _http_response(
            payload={"data": [{"id": "ocr"}, {"id": "vision"}]},
        )

    monkeypatch.setattr(lp.httpx, "get", fake_get)
    ok, reason = lp.check_litellm_ocr()

    assert ok
    assert calls["url"] == expected_url
    assert calls["headers"]["Authorization"] == "Bearer test-secret-key"
    assert calls["timeout"] == lp.PROBE_TIMEOUT_S
    assert "ocr" in reason and "vision" in reason
    assert "test-secret-key" not in reason
    assert "fallback-secret-key" not in reason


def test_litellm_ocr_accepts_master_key_when_api_key_missing(
        tmp_path, monkeypatch):
    cfg = _write_litellm_config(tmp_path / "config.yaml")
    monkeypatch.setattr(lp, "config_candidates", lambda: [cfg])
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.setenv("LITELLM_MASTER_KEY", "master-secret-key")
    calls = {}

    def fake_get(url, *, headers, timeout):
        calls["authorization"] = headers["Authorization"]
        return _http_response(
            payload={"data": [{"id": "ocr"}, {"id": "vision"}]},
        )

    monkeypatch.setattr(lp.httpx, "get", fake_get)
    ok, reason = lp.check_litellm_ocr()

    assert ok
    assert calls["authorization"] == "Bearer master-secret-key"
    assert "master-secret-key" not in reason


def test_litellm_ocr_requires_provider_litellm(tmp_path, monkeypatch):
    cfg = _write_litellm_config(tmp_path / "config.yaml", provider="deepseek_ocr2")
    monkeypatch.setattr(lp, "config_candidates", lambda: [cfg])
    monkeypatch.setenv("LITELLM_API_KEY", "test-secret-key")
    ok, reason = lp.check_litellm_ocr()
    assert not ok
    assert "provider" in reason and "litellm" in reason


@pytest.mark.parametrize("field", ["endpoint", "extract_model", "describe_model"])
def test_litellm_ocr_reports_missing_config_field(tmp_path, monkeypatch, field):
    cfg = _write_litellm_config(tmp_path / "config.yaml", **{field: ""})
    monkeypatch.setattr(lp, "config_candidates", lambda: [cfg])
    monkeypatch.setenv("LITELLM_API_KEY", "test-secret-key")
    ok, reason = lp.check_litellm_ocr()
    assert not ok
    assert field in reason


def test_litellm_ocr_requires_env_key_and_ignores_yaml_key(tmp_path, monkeypatch):
    cfg = _write_litellm_config(
        tmp_path / "config.yaml",
        api_key="must-not-be-used-or-printed",
    )
    monkeypatch.setattr(lp, "config_candidates", lambda: [cfg])
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    monkeypatch.setattr(
        lp.httpx,
        "get",
        lambda *args, **kwargs: pytest.fail("request must not run without env key"),
    )

    ok, reason = lp.check_litellm_ocr()

    assert not ok
    assert "LITELLM_API_KEY" in reason
    assert "LITELLM_MASTER_KEY" in reason
    assert "must-not-be-used-or-printed" not in reason


def test_litellm_ocr_reports_missing_config(tmp_path, monkeypatch):
    monkeypatch.setattr(lp, "config_candidates",
                        lambda: [tmp_path / "missing.yaml"])
    monkeypatch.setenv("LITELLM_API_KEY", "test-secret-key")
    ok, reason = lp.check_litellm_ocr()
    assert not ok
    assert "config.yaml" in reason


def test_litellm_ocr_reports_unreachable_endpoint(tmp_path, monkeypatch):
    cfg = _write_litellm_config(tmp_path / "config.yaml")
    monkeypatch.setattr(lp, "config_candidates", lambda: [cfg])
    monkeypatch.setenv("LITELLM_API_KEY", "test-secret-key")

    def refuse(*args, **kwargs):
        raise lp.httpx.ConnectError("refused")

    monkeypatch.setattr(lp.httpx, "get", refuse)
    ok, reason = lp.check_litellm_ocr()
    assert not ok
    assert "unreachable" in reason.lower()
    assert "test-secret-key" not in reason


def test_litellm_ocr_invalid_url_preserves_tuple_contract(tmp_path, monkeypatch):
    cfg = _write_litellm_config(tmp_path / "config.yaml", endpoint=":not-a-url")
    monkeypatch.setattr(lp, "config_candidates", lambda: [cfg])
    monkeypatch.setenv("LITELLM_API_KEY", "test-secret-key")

    def invalid(*args, **kwargs):
        raise lp.httpx.InvalidURL("invalid URL")

    monkeypatch.setattr(lp.httpx, "get", invalid)
    result = lp.check_litellm_ocr()

    assert isinstance(result, tuple) and len(result) == 2
    ok, reason = result
    assert not ok
    assert "invalid" in reason.lower()
    assert "test-secret-key" not in reason


def test_litellm_ocr_does_not_swallow_unexpected_error(tmp_path, monkeypatch):
    cfg = _write_litellm_config(tmp_path / "config.yaml")
    monkeypatch.setattr(lp, "config_candidates", lambda: [cfg])
    monkeypatch.setenv("LITELLM_API_KEY", "test-secret-key")

    def crash(*args, **kwargs):
        raise RuntimeError("unexpected defect")

    monkeypatch.setattr(lp.httpx, "get", crash)
    with pytest.raises(RuntimeError, match="unexpected defect"):
        lp.check_litellm_ocr()


def test_litellm_ocr_reports_unauthorized_without_key_leak(tmp_path, monkeypatch):
    cfg = _write_litellm_config(tmp_path / "config.yaml")
    monkeypatch.setattr(lp, "config_candidates", lambda: [cfg])
    monkeypatch.setenv("LITELLM_API_KEY", "test-secret-key")
    monkeypatch.setattr(
        lp.httpx,
        "get",
        lambda *args, **kwargs: _http_response(401, {"detail": "unauthorized"}),
    )
    ok, reason = lp.check_litellm_ocr()
    assert not ok
    assert "401" in reason
    assert "test-secret-key" not in reason


def test_litellm_ocr_reports_malformed_json(tmp_path, monkeypatch):
    cfg = _write_litellm_config(tmp_path / "config.yaml")
    monkeypatch.setattr(lp, "config_candidates", lambda: [cfg])
    monkeypatch.setenv("LITELLM_API_KEY", "test-secret-key")
    monkeypatch.setattr(
        lp.httpx,
        "get",
        lambda *args, **kwargs: _http_response(200, content=b"not-json"),
    )
    ok, reason = lp.check_litellm_ocr()
    assert not ok
    assert "invalid" in reason.lower() or "malformed" in reason.lower()


@pytest.mark.parametrize("payload", [{}, {"data": "not-a-list"},
                                      {"data": [{"name": "ocr"}]}])
def test_litellm_ocr_reports_malformed_model_schema(tmp_path, monkeypatch, payload):
    cfg = _write_litellm_config(tmp_path / "config.yaml")
    monkeypatch.setattr(lp, "config_candidates", lambda: [cfg])
    monkeypatch.setenv("LITELLM_API_KEY", "test-secret-key")
    monkeypatch.setattr(
        lp.httpx,
        "get",
        lambda *args, **kwargs: _http_response(payload=payload),
    )
    ok, reason = lp.check_litellm_ocr()
    assert not ok
    assert "schema" in reason.lower()


@pytest.mark.parametrize(
    ("models", "missing"),
    [(["vision"], "ocr"), (["ocr"], "vision")],
)
def test_litellm_ocr_reports_each_missing_alias(
        tmp_path, monkeypatch, models, missing):
    cfg = _write_litellm_config(tmp_path / "config.yaml")
    monkeypatch.setattr(lp, "config_candidates", lambda: [cfg])
    monkeypatch.setenv("LITELLM_API_KEY", "test-secret-key")
    monkeypatch.setattr(
        lp.httpx,
        "get",
        lambda *args, **kwargs: _http_response(
            payload={"data": [{"id": model} for model in models]},
        ),
    )
    ok, reason = lp.check_litellm_ocr()
    assert not ok
    assert missing in reason


# ---------------------------------------------------------------------------
# check_prod_indexer_idle
# ---------------------------------------------------------------------------

def _fake_run(stdout="", stderr="", returncode=0, raise_exc=None, calls=None):
    def run(cmd, **kw):
        if calls is not None:
            calls.append((cmd, kw))
        if raise_exc is not None:
            raise raise_exc
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)
    return run


def _run_indexer_pid_probe(tmp_path, *, pid=None, state=None, cmdline=None):
    pid_path = tmp_path / "indexer.pid"
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    if pid is not None:
        pid_path.write_text(str(pid))
        if state is not None:
            process_dir = proc_root / str(pid)
            process_dir.mkdir()
            (process_dir / "status").write_text(f"State:\t{state}\n")
            if cmdline is not None:
                (process_dir / "cmdline").write_bytes(cmdline)
    return subprocess.run(
        ["sh", "-c", lp.build_indexer_pid_probe(pid_path, proc_root)],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("probe_state", "expected"),
    [
        ({}, "idle"),
        ({"pid": 4242}, "idle"),
        ({"pid": 4242, "state": "Z (zombie)"}, "idle"),
    ],
)
def test_indexer_pid_probe_accepts_only_verified_idle_states(
        tmp_path, probe_state, expected):
    proc = _run_indexer_pid_probe(tmp_path, **probe_state)

    assert proc.returncode == 0
    assert proc.stdout.strip() == expected


def test_indexer_pid_probe_identifies_matching_live_process(tmp_path):
    proc = _run_indexer_pid_probe(
        tmp_path,
        pid=4242,
        state="S (sleeping)",
        cmdline=b"python\0-c\0from flow_index_vault import index_vault_flow\0",
    )

    assert proc.returncode == 0
    assert proc.stdout.strip() == "active:4242"


@pytest.mark.parametrize(
    ("probe_state", "expected_error"),
    [
        ({"pid": "not-a-pid"}, "invalid indexer pid file"),
        (
            {"pid": 4242, "state": "S (sleeping)", "cmdline": b"python\0worker.py\0"},
            "pid does not identify indexer process",
        ),
        ({"pid": 4242, "state": "S (sleeping)"}, "cmdline"),
    ],
)
def test_indexer_pid_probe_fails_closed_on_ambiguous_state(
        tmp_path, probe_state, expected_error):
    proc = _run_indexer_pid_probe(tmp_path, **probe_state)

    assert proc.returncode != 0
    assert expected_error in proc.stderr.lower()


def test_indexer_active_pid_fails_even_without_heartbeat(monkeypatch):
    calls = []
    monkeypatch.setattr(
        lp.subprocess,
        "run",
        _fake_run(stdout="active:4242\n", calls=calls),
    )

    ok, reason = lp.check_prod_indexer_idle()

    assert not ok
    assert "active" in reason.lower()
    assert "4242" in reason
    command = calls[0][0]
    assert command[:3] == ["docker", "exec", lp.PROD_CONTAINER]
    probe = command[-1]
    assert lp.INDEXER_PID_PATH in probe
    assert 'status="$proc_root/$pid/status"' in probe
    assert '"$proc_root/$pid/cmdline"' in probe
    assert "flow_index_vault" in probe


def test_indexer_no_pid_is_idle(monkeypatch):
    monkeypatch.setattr(lp.subprocess, "run", _fake_run(stdout="idle\n"))

    ok, reason = lp.check_prod_indexer_idle()
    assert ok
    assert "idle" in reason.lower()


def test_indexer_probe_nonzero_fails_closed(monkeypatch):
    monkeypatch.setattr(
        lp.subprocess,
        "run",
        _fake_run(stderr="container is not running", returncode=1),
    )

    ok, reason = lp.check_prod_indexer_idle()

    assert not ok
    assert "container is not running" in reason


def test_indexer_docker_binary_missing_fails_closed(monkeypatch):
    monkeypatch.setattr(lp.subprocess, "run",
                        _fake_run(raise_exc=FileNotFoundError("docker")))

    ok, reason = lp.check_prod_indexer_idle()

    assert not ok
    assert "docker" in reason.lower()


def test_indexer_docker_timeout_fails_closed(monkeypatch):
    exc = subprocess.TimeoutExpired(cmd=["docker"], timeout=10)
    monkeypatch.setattr(lp.subprocess, "run", _fake_run(raise_exc=exc))

    ok, reason = lp.check_prod_indexer_idle()

    assert not ok
    assert "timed out" in reason.lower()


def test_indexer_unexpected_probe_output_fails_closed(monkeypatch):
    monkeypatch.setattr(lp.subprocess, "run", _fake_run(stdout="garbage\n"))

    ok, reason = lp.check_prod_indexer_idle()

    assert not ok
    assert "unexpected" in reason.lower()


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
    assert names == ["api_keys", "config_test", "litellm_ocr",
                     "prod_indexer_idle", "comm_postgres"]
    assert "mac_ocr" not in names
