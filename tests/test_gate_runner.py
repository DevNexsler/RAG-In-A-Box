# NOTE: `scripts` has no __init__.py — this import works via conftest's sys.path
# insert + namespace packages. Do not "fix" by adding __init__.py.
import json
import os
import subprocess
import sys
from pathlib import Path

import scripts.gate as gate
from scripts.gate import TIERS, next_tier_allowed, preflight_passed

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_gate(args, tmp_path, env=None):
    full_env = {**os.environ, **(env or {})}
    return subprocess.run(
        [sys.executable, "scripts/gate.py", *args, "--run-dir", str(tmp_path / "run")],
        capture_output=True, text=True, cwd=REPO_ROOT, env=full_env,
    )


def test_tier_order():
    assert [t.name for t in TIERS] == ["static", "unit", "integration", "staging-e2e", "live"]


def test_fail_fast():
    results = {"static": True, "unit": False}
    assert next_tier_allowed("integration", results) is False


def test_live_requires_all_prior():
    results = {"static": True, "unit": True, "integration": True, "staging-e2e": True}
    assert next_tier_allowed("live", results) is True


def test_preflight_rc_handling():
    # Regression: True == 1 in Python, so an `rc in (0, None, True)` check
    # would approve the real-money live tier on a FAILING preflight (rc=1).
    assert preflight_passed(0) is True
    assert preflight_passed(1) is False
    assert preflight_passed(2) is False
    assert preflight_passed(None) is False


def test_only_staging_e2e_fails_without_compose_file(tmp_path):
    # docker-compose.staging.yml exists as of Task 7, so point the runner at a
    # nonexistent compose file via GATE_COMPOSE_FILE — the missing-compose path
    # must still fail fast, and must never invoke docker.
    proc = _run_gate(
        ["--only", "staging-e2e"], tmp_path,
        env={"GATE_COMPOSE_FILE": "docker-compose.does-not-exist.yml"},
    )
    assert proc.returncode != 0
    assert "docker-compose.does-not-exist.yml not found" in proc.stdout


def _run_gate_from(cwd, args, tmp_path):
    # Run gate.py from an arbitrary cwd: gate resolves scripts/live_preflight.py
    # relative to cwd, so a bare tmp dir simulates "preflight missing" and a
    # planted fake simulates any preflight exit code — hermetically. Never run
    # `--only live` with cwd=REPO_ROOT here: the REAL preflight exists there
    # now (Task 10), and passing preflight would start the real-money live tier.
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "gate.py"),
         *args, "--run-dir", str(tmp_path / "run")],
        capture_output=True, text=True, cwd=cwd, env={**os.environ},
    )


def test_only_live_fails_when_preflight_missing(tmp_path):
    proc = _run_gate_from(tmp_path, ["--only", "live"], tmp_path)
    assert proc.returncode != 0
    assert "live preflight not implemented" in proc.stdout
    assert "-m pytest -m live" not in proc.stdout  # live tier never launched


def test_only_live_blocked_when_preflight_fails(tmp_path):
    fake = tmp_path / "scripts" / "live_preflight.py"
    fake.parent.mkdir()
    fake.write_text("import sys\nsys.exit(1)\n")
    proc = _run_gate_from(tmp_path, ["--only", "live"], tmp_path)
    assert proc.returncode != 0
    assert "FAIL: live" in proc.stdout
    assert "-m pytest -m live" not in proc.stdout  # failing preflight must block the spend


# --- result.json ---------------------------------------------------------------

ALL_TIERS = ["static", "unit", "integration", "staging-e2e", "live"]
# e2e-real is opt-in (--with-real-e2e); it stays not_run in a default gate run.


def _read_result(run_dir):
    return json.loads((run_dir / "result.json").read_text())


def test_result_json_all_pass(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no scripts/gate_report.py here — hermetic
    monkeypatch.setattr(gate, "dispatch", lambda tier, run_dir: True)
    run_dir = tmp_path / "run"
    assert gate.main(["--run-dir", str(run_dir)]) == 0
    data = _read_result(run_dir)
    assert data["overall"] == "pass"
    assert data["tiers"] == {**{name: "pass" for name in ALL_TIERS}, "e2e-real": "not_run"}


def test_result_json_marks_skipped_after_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gate, "dispatch",
                        lambda tier, run_dir: tier.name != "unit")
    run_dir = tmp_path / "run"
    assert gate.main(["--run-dir", str(run_dir)]) == 1
    data = _read_result(run_dir)
    assert data["overall"] == "fail"
    assert data["tiers"] == {
        "static": "pass", "unit": "fail", "integration": "skipped",
        "staging-e2e": "skipped", "live": "skipped", "e2e-real": "not_run",
    }


def test_result_json_only_mode_marks_unselected_not_run(tmp_path):
    # --only staging-e2e with a nonexistent compose file: the tier fails
    # hermetically (never invokes docker) and everything else is not_run.
    proc = _run_gate(
        ["--only", "staging-e2e"], tmp_path,
        env={"GATE_COMPOSE_FILE": "docker-compose.does-not-exist.yml"},
    )
    assert proc.returncode != 0
    data = _read_result(tmp_path / "run")
    assert data["overall"] == "fail"
    assert data["tiers"] == {
        "static": "not_run", "unit": "not_run", "integration": "not_run",
        "staging-e2e": "fail", "live": "not_run", "e2e-real": "not_run",
    }


def test_result_json_written_before_report_generator_runs(tmp_path, monkeypatch):
    # gate_report.py must be able to prefer result.json — so the runner must
    # write it BEFORE invoking the report. The fake report script snapshots
    # what it can see at invocation time.
    fake = tmp_path / "scripts" / "gate_report.py"
    fake.parent.mkdir()
    fake.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "run_dir = Path(sys.argv[1])\n"
        "path = run_dir / 'result.json'\n"
        "seen = json.loads(path.read_text())['overall'] if path.exists() else 'MISSING'\n"
        "(run_dir / 'seen-by-report.txt').write_text(seen)\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gate, "dispatch", lambda tier, run_dir: True)
    run_dir = tmp_path / "run"
    assert gate.main(["--run-dir", str(run_dir)]) == 0
    assert (run_dir / "seen-by-report.txt").read_text() == "pass"


def test_with_real_e2e_appends_e2e_real_stage(tmp_path, monkeypatch):
    # --with-real-e2e adds the opt-in real-API stage; it runs last and, when
    # every prior tier passes, is exercised (here dispatch is stubbed to pass).
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gate, "dispatch", lambda tier, run_dir: True)
    run_dir = tmp_path / "run"
    assert gate.main(["--with-real-e2e", "--run-dir", str(run_dir)]) == 0
    data = _read_result(run_dir)
    assert data["tiers"]["e2e-real"] == "pass"


def test_e2e_real_preflight_requires_real_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert gate.e2e_real_preflight() is False       # unset → blocked
    monkeypatch.setenv("OPENROUTER_API_KEY", "sim")
    assert gate.e2e_real_preflight() is False        # sim placeholder → blocked
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real")
    assert gate.e2e_real_preflight() is True          # real key → allowed


def test_e2e_real_not_in_default_tiers():
    # The default gate must never include the money-spending stage.
    assert "e2e-real" not in [t.name for t in gate.TIERS]
    assert gate.E2E_REAL_TIER.name == "e2e-real"


def test_staging_tiers_fail_closed_against_host_provider_selectors():
    staging = next(tier for tier in gate.TIERS if tier.name == "staging-e2e")

    assert dict(staging.compose_env) == {
        "STAGING_CONFIG": "./config.staging.yaml",
        "E2E_REAL": "0",
    }
    assert dict(staging.pytest_env) == {
        "STAGING_CONFIG": "./config.staging.yaml",
        "E2E_REAL": "0",
    }
    assert dict(gate.E2E_REAL_TIER.compose_env) == {
        "STAGING_CONFIG": "./config.staging.realmedia.yaml",
        "E2E_REAL": "1",
    }
    assert dict(gate.E2E_REAL_TIER.pytest_env) == {
        "STAGING_CONFIG": "./config.staging.realmedia.yaml",
        "E2E_REAL": "1",
    }


def test_compose_tiers_propagate_explicit_provider_env(
    tmp_path,
    monkeypatch,
):
    compose_file = tmp_path / "compose.yml"
    compose_file.write_text("services: {}\n")
    monkeypatch.setattr(gate, "COMPOSE_FILE", compose_file)
    monkeypatch.setenv("STAGING_CONFIG", "./hostile-realmedia.yaml")
    monkeypatch.setenv("E2E_REAL", "1")

    calls = []

    def capture_run(cmd, **kwargs):
        calls.append((list(cmd), dict(kwargs.get("env") or {})))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(gate.subprocess, "run", capture_run)
    staging = next(tier for tier in gate.TIERS if tier.name == "staging-e2e")

    assert gate.run_compose_tier(staging, tmp_path / "staging") is True
    staging_calls = list(calls)
    calls.clear()
    assert gate.run_compose_tier(gate.E2E_REAL_TIER, tmp_path / "real") is True

    for cmd, env in staging_calls:
        if cmd[:2] == ["docker", "compose"] or "pytest" in cmd:
            assert env["STAGING_CONFIG"] == "./config.staging.yaml"
            assert env["E2E_REAL"] == "0"

    for cmd, env in calls:
        if cmd[:2] == ["docker", "compose"] or "pytest" in cmd:
            assert env["STAGING_CONFIG"] == "./config.staging.realmedia.yaml"
            assert env["E2E_REAL"] == "1"
