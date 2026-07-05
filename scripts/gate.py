#!/usr/bin/env python3
"""Gate runner: ordered tiers, fail-fast, artifacts per run.

Usage: python scripts/gate.py [--fast] [--only TIER] [--run-dir DIR]
"""
import argparse
import dataclasses
import subprocess
import sys
import time
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class Tier:
    name: str
    cmd: list          # command template; {run_dir} substituted
    needs_compose: bool = False


RUN_ROOT = Path(".evals/gate-runs")
COMPOSE_FILE = Path("docker-compose.staging.yml")

# static is two commands; run_tier runs them in sequence, both must pass
STATIC_SECOND_CMD = ["python", "-m", "pytest", "--collect-only", "-q"]

TIERS = [
    Tier("static", ["ruff", "check", "."]),
    Tier("unit", ["python", "-m", "pytest", "-m", "unit", "-q",
                  "--junitxml={run_dir}/unit.xml"]),
    Tier("integration", ["python", "-m", "pytest", "-m", "integration", "-q",
                         "--junitxml={run_dir}/integration.xml"]),
    Tier("staging-e2e", ["python", "-m", "pytest", "tests/e2e", "-m", "e2e", "-q",
                         "--junitxml={run_dir}/e2e.xml"], needs_compose=True),
    Tier("live", ["python", "-m", "pytest", "-m", "live", "-q",
                  "--junitxml={run_dir}/live.xml"]),
]


def next_tier_allowed(name, results):
    for t in TIERS:
        if t.name == name:
            return True
        if not results.get(t.name, False):
            return False
    return False


def _run(cmd, run_dir):
    cmd = [part.format(run_dir=run_dir) for part in cmd]
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd).returncode == 0


def run_tier(tier, run_dir):
    ok = _run(tier.cmd, run_dir)
    if tier.name == "static":
        ok = ok and _run(STATIC_SECOND_CMD, run_dir)
    return ok


def collect_staging_traces(run_dir):
    # Stub until the staging stack settles: copy the traces volume out of the
    # app container; tolerate failure with a warning.
    cmd = ["docker", "compose", "-f", str(COMPOSE_FILE),
           "cp", "app:/traces", str(run_dir / "traces")]
    if subprocess.run(cmd).returncode != 0:
        print("WARN: could not collect staging traces", flush=True)


def run_compose_tier(tier, run_dir):
    if not COMPOSE_FILE.exists():
        print(f"FAIL {tier.name}: docker-compose.staging.yml not found (Task 7 pending)", flush=True)
        return False
    up = ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d", "--build", "--wait"]
    down = ["docker", "compose", "-f", str(COMPOSE_FILE), "down", "-v"]
    subprocess.run(up, check=True)
    try:
        ok = run_tier(tier, run_dir)
        collect_staging_traces(run_dir)
    finally:
        subprocess.run(down, check=False)
    return ok


def live_preflight_ok():
    if not Path("scripts/live_preflight.py").exists():
        print("FAIL live: live preflight not implemented (Task 10 pending)", flush=True)
        return False
    from scripts.live_preflight import main as preflight  # noqa: PLC0415
    rc = preflight()
    return rc in (0, None, True)


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
    for tier in selected:
        if not args.only and not next_tier_allowed(tier.name, results):
            print(f"STOP: {tier.name} skipped (prior tier failed)", flush=True)
            break
        print(f"=== {tier.name} ===", flush=True)
        ok = dispatch(tier, run_dir)
        results[tier.name] = ok
        print(f"{'PASS' if ok else 'FAIL'}: {tier.name}", flush=True)
        if not ok:
            all_ok = False

    if Path("scripts/gate_report.py").exists():
        subprocess.run([sys.executable, "scripts/gate_report.py", str(run_dir)], check=False)

    print(f"gate: {'PASS' if all_ok else 'FAIL'} (artifacts: {run_dir})", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
