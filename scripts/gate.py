#!/usr/bin/env python3
"""Gate runner: ordered tiers, fail-fast, artifacts per run.

Usage: python scripts/gate.py [--fast] [--only TIER] [--run-dir DIR]
"""
import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class Tier:
    name: str
    cmd: list[str]     # command template; {run_dir} substituted
    needs_compose: bool = False


RUN_ROOT = Path(".evals/gate-runs")
# GATE_COMPOSE_FILE override exists so tests can exercise the missing-compose
# path without touching the real stack (and without ever invoking docker).
COMPOSE_FILE = Path(os.environ.get("GATE_COMPOSE_FILE", "docker-compose.staging.yml"))

# static is two commands; run_tier runs them in sequence, both must pass
STATIC_SECOND_CMD = [sys.executable, "-m", "pytest", "--collect-only", "-q"]

TIERS = [
    Tier("static", ["ruff", "check", "."]),
    Tier("unit", [sys.executable, "-m", "pytest", "-m", "unit", "-q",
                  "--junitxml={run_dir}/unit.xml"]),
    Tier("integration", [sys.executable, "-m", "pytest", "-m", "integration", "-q",
                         "--junitxml={run_dir}/integration.xml"]),
    Tier("staging-e2e", [sys.executable, "-m", "pytest", "tests/e2e", "-m", "e2e", "-q",
                         "--junitxml={run_dir}/e2e.xml"], needs_compose=True),
    Tier("live", [sys.executable, "-m", "pytest", "-m", "live", "-q",
                  "--junitxml={run_dir}/live.xml"]),
]


def next_tier_allowed(name, results):
    for t in TIERS:
        if t.name == name:
            return True
        if not results.get(t.name, False):
            return False
    return False


def _run(cmd, run_dir, tier_name):
    cmd = [part.format(run_dir=run_dir) for part in cmd]
    print(f"  $ {' '.join(cmd)}", flush=True)
    try:
        return subprocess.run(cmd).returncode == 0
    except FileNotFoundError:
        print(f"FAIL {tier_name}: command not found: {cmd[0]}", flush=True)
        return False


def run_tier(tier, run_dir):
    ok = _run(tier.cmd, run_dir, tier.name)
    if tier.name == "static":
        ok = ok and _run(STATIC_SECOND_CMD, run_dir, tier.name)
    return ok


def collect_staging_traces(run_dir):
    # Copy the traces volume (tracing.directory in config.staging.yaml) out of
    # the app container; tolerate failure with a warning.
    cmd = ["docker", "compose", "-f", str(COMPOSE_FILE),
           "cp", "doc-organizer-staging:/data/traces", str(run_dir / "traces")]
    try:
        rc = subprocess.run(cmd).returncode
    except FileNotFoundError:
        rc = 1
    if rc != 0:
        print("WARN: could not collect staging traces", flush=True)


def check_tool_coverage(run_dir):
    # Two-sided tool-coverage enforcement (Task 9). Needs the live MCP endpoint
    # for list_tools, so it must run INSIDE the compose window, after
    # collect_staging_traces has copied the span artifacts into run_dir.
    cmd = [sys.executable, "scripts/check_tool_coverage.py", "--run-dir", str(run_dir)]
    print(f"  $ {' '.join(cmd)}", flush=True)
    try:
        return subprocess.run(cmd).returncode == 0
    except FileNotFoundError:
        print("FAIL staging-e2e: tool-coverage check could not run", flush=True)
        return False


def run_compose_tier(tier, run_dir):
    if not COMPOSE_FILE.exists():
        print(f"FAIL {tier.name}: {COMPOSE_FILE} not found", flush=True)
        return False
    up = ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d", "--build", "--wait"]
    down = ["docker", "compose", "-f", str(COMPOSE_FILE), "down", "-v"]
    ok = False
    try:
        # up runs INSIDE the try: a partially-started stack must still get `down -v`
        subprocess.run(up, check=True)
        ok = run_tier(tier, run_dir)
        collect_staging_traces(run_dir)
        if ok:
            ok = check_tool_coverage(run_dir)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"FAIL {tier.name}: compose up failed: {exc}", flush=True)
    finally:
        try:
            subprocess.run(down, check=False)
        except FileNotFoundError:
            pass  # docker itself missing; the except above already reported it
    return ok


def preflight_passed(rc):
    # Strict: ONLY exit code 0 approves the real-money live tier.
    # (Never use `rc in (0, None, True)` — True == 1 in Python, so a failing
    # preflight exit code of 1 would silently pass.)
    return rc == 0


def live_preflight_ok():
    if not Path("scripts/live_preflight.py").exists():
        print("FAIL live: live preflight not implemented (Task 10 pending)", flush=True)
        return False
    rc = subprocess.run([sys.executable, "scripts/live_preflight.py"]).returncode
    return preflight_passed(rc)


def dispatch(tier, run_dir):
    if tier.name == "live" and not live_preflight_ok():
        return False
    if tier.needs_compose:
        return run_compose_tier(tier, run_dir)
    return run_tier(tier, run_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="stop after integration")
    parser.add_argument("--only", metavar="TIER", choices=[t.name for t in TIERS],
                        help="run a single tier, skipping prerequisites")
    parser.add_argument("--run-dir", help="artifact directory (default: .evals/gate-runs/<timestamp>)")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir) if args.run_dir else RUN_ROOT / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.only:
        selected = [t for t in TIERS if t.name == args.only]
    elif args.fast:
        selected = TIERS[:[t.name for t in TIERS].index("integration") + 1]
    else:
        selected = TIERS

    results = {}
    all_ok = True
    # Machine-readable per-tier states for result.json: every tier (including
    # static, which emits no junit artifact) starts as not_run; selected tiers
    # become pass/fail when run, or skipped when a prior tier failed.
    states = {t.name: "not_run" for t in TIERS}
    for i, tier in enumerate(selected):
        if not args.only and not next_tier_allowed(tier.name, results):
            print(f"STOP: {tier.name} skipped (prior tier failed)", flush=True)
            for rest in selected[i:]:
                states[rest.name] = "skipped"
            break
        print(f"=== {tier.name} ===", flush=True)
        ok = dispatch(tier, run_dir)
        results[tier.name] = ok
        states[tier.name] = "pass" if ok else "fail"
        print(f"{'PASS' if ok else 'FAIL'}: {tier.name}", flush=True)
        if not ok:
            all_ok = False

    # Written BEFORE gate_report.py runs so the report can prefer the runner's
    # own verdict over artifact inference.
    (run_dir / "result.json").write_text(json.dumps(
        {"tiers": states, "overall": "pass" if all_ok else "fail"}, indent=2) + "\n")

    if Path("scripts/gate_report.py").exists():
        subprocess.run([sys.executable, "scripts/gate_report.py", str(run_dir)], check=False)

    print(f"gate: {'PASS' if all_ok else 'FAIL'} (artifacts: {run_dir})", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
