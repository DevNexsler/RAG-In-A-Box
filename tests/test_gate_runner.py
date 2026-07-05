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


def test_only_live_fails_without_preflight(tmp_path):
    # scripts/live_preflight.py does not exist until Task 10 lands.
    proc = _run_gate(["--only", "live"], tmp_path)
    assert proc.returncode != 0
    assert "Task 10 pending" in proc.stdout
