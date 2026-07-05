# NOTE: `scripts` has no __init__.py — this import works via conftest's sys.path
# insert + namespace packages. Do not "fix" by adding __init__.py.
import os
import subprocess
import sys
from pathlib import Path

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
