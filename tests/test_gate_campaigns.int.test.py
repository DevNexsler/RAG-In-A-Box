"""Gate CLI must run safety campaigns before containers or paid providers.

Only external commands are replaced. The real CLI chooses tiers, runs commands,
stops on failure, and writes its verdict.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def gate_cli(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    commands = tmp_path / "bin"
    commands.mkdir()
    stub = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "name = Path(sys.argv[0]).stem\n"
        "with Path('commands.jsonl').open('a') as log:\n"
        "    log.write(json.dumps([name, *sys.argv[1:]]) + '\\n')\n"
        "failed = Path('fail-command').read_text() if Path('fail-command').exists() else ''\n"
        "raise SystemExit(1 if name == failed else 0)\n"
    )
    (tmp_path / "pytest.py").write_text(stub)
    for name in ("ruff", "docker"):
        path = commands / name
        path.write_text(f"#!{sys.executable}\n" + stub)
        path.chmod(0o755)
    for name in ("mutation_gate", "pipeline_soak", "live_preflight",
                 "check_tool_coverage", "attachment_path_audit"):
        (scripts / f"{name}.py").write_text(stub)
    env = {**os.environ, "PATH": f"{commands}{os.pathsep}{os.environ['PATH']}"}
    # Fixed local bindings avoid needing a fake Docker port-discovery response.
    env.pop("COMPOSE_PROJECT_NAME", None)
    env.pop("GATE_COMPOSE_FILE", None)
    (tmp_path / "docker-compose.staging.yml").write_text("services: {}\n")

    def run(*args, fail=None):
        if fail:
            (tmp_path / "fail-command").write_text(fail)
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/gate.py"), *args,
             "--run-dir", str(tmp_path / "run")],
            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
        )
        calls = [json.loads(line) for line in (tmp_path / "commands.jsonl").read_text().splitlines()]
        result = json.loads((tmp_path / "run/result.json").read_text())
        return completed, calls, result

    return run


def test_release_runs_mutations_and_soak_before_staging_and_live(gate_cli):
    completed, calls, result = gate_cli()
    assert completed.returncode == 0, completed.stdout + completed.stderr
    names = [call[0] for call in calls]
    assert "mutation_gate" in names, "release omitted mutation campaign"
    assert "pipeline_soak" in names, "release omitted bounded soak"
    assert names.index("mutation_gate") < names.index("pipeline_soak") < names.index("docker")
    assert names.index("docker") < names.index("live_preflight")
    assert result["tiers"]["mutation"] == result["tiers"]["soak"] == "pass"
    for command, artifact in (("mutation_gate", "mutation.json"), ("pipeline_soak", "soak.json")):
        call = next(call for call in calls if call[0] == command)
        assert Path(call[call.index("--output") + 1]).name == artifact


@pytest.mark.parametrize("tier,command", [("mutation", "mutation_gate"), ("soak", "pipeline_soak")])
def test_failed_campaign_blocks_containers_and_paid_providers(gate_cli, tier, command):
    completed, calls, result = gate_cli(fail=command)
    assert completed.returncode == 1
    assert result["overall"] == "fail"
    assert result["tiers"][tier] == "fail"
    assert result["tiers"]["staging-e2e"] == result["tiers"]["live"] == "skipped"
    names = [call[0] for call in calls]
    assert "docker" not in names
    assert "live_preflight" not in names
    if tier == "mutation":
        assert "pipeline_soak" not in names
        assert result["tiers"]["soak"] == "skipped"


def test_fast_gate_leaves_campaigns_unrun(gate_cli):
    completed, calls, result = gate_cli("--fast")
    assert completed.returncode == 0
    assert [call[0] for call in calls] == ["ruff", "pytest", "pytest", "pytest"]
    assert result["tiers"]["integration"] == "pass"
    assert result["tiers"]["mutation"] == result["tiers"]["soak"] == "not_run"
    assert result["tiers"]["staging-e2e"] == result["tiers"]["live"] == "not_run"


@pytest.mark.parametrize("tier,command", [("mutation", "mutation_gate"), ("soak", "pipeline_soak")])
def test_campaign_can_run_alone_with_own_verdict(gate_cli, tier, command):
    completed, calls, result = gate_cli("--only", tier)
    assert completed.returncode == 0
    assert [call[0] for call in calls] == [command]
    assert result["tiers"][tier] == "pass"
    assert all(state == "not_run" for name, state in result["tiers"].items() if name != tier)
