#!/usr/bin/env python3
"""Restart doc-organizer when its server process outgrows the memory it can share.

The server's resident set grows between restarts (measured 2026-09-07: about
0.4 GB per hour under organic load, most of it after each index sweep) and
shares a 10 GiB cgroup with the ~3.7 GB sweep subprocess. Past ~7 GB the next
sweep can reach the cgroup ceiling and the kernel kills the fattest task — the
server — mid-run. A planned restart at a quiet moment costs a 15-second health
blip and nothing else: the index lives on disk.

Rules, in order: never restart while an index run holds the table
(/data/index/indexer.pid), never restart a container that is not healthy
(something else is wrong; a restart would hide it), restart only above the
threshold. One JSON line per run to stdout; systemd appends it to the log.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_THRESHOLD_MB = int(os.environ.get("DOC_ORGANIZER_RSS_RESTART_MB", "7000"))
_REPO = Path(__file__).resolve().parents[1]


def decide(rss_mb: int | None, threshold_mb: int, sweep_running: bool, healthy: bool) -> tuple[str, str]:
    """(action, reason): action is "restart" or "skip"."""
    if rss_mb is None:
        return "skip", "server rss unreadable"
    if rss_mb <= threshold_mb:
        return "skip", f"rss {rss_mb} MB <= {threshold_mb} MB"
    if sweep_running:
        return "skip", f"rss {rss_mb} MB over threshold but an index run holds the table"
    if not healthy:
        return "skip", f"rss {rss_mb} MB over threshold but container is not healthy"
    return "restart", f"rss {rss_mb} MB > {threshold_mb} MB, no index run, healthy"


def parse_rss_mb(status_text: str) -> int | None:
    for line in status_text.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) // 1024
    return None


def _docker(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=60)


def server_rss_mb(container: str) -> int | None:
    """RSS of `python server.py` inside the container (not the init shim, not the indexer)."""
    script = (
        "for p in /proc/[0-9]*; do "
        "c=$(tr '\\0' ' ' < $p/cmdline 2>/dev/null); "
        "case \"$c\" in 'python server.py '*) cat $p/status; exit 0;; esac; done; exit 1"
    )
    out = _docker("exec", container, "sh", "-c", script)
    return parse_rss_mb(out.stdout) if out.returncode == 0 else None


def sweep_running(container: str) -> bool:
    return _docker("exec", container, "test", "-f", "/data/index/indexer.pid").returncode == 0


def container_healthy(container: str) -> bool:
    out = _docker("inspect", "--format", "{{.State.Health.Status}}", container)
    return out.returncode == 0 and out.stdout.strip() == "healthy"


def restart(container: str, compose_file: Path, project_dir: Path, health_url: str) -> bool:
    subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "--project-directory", str(project_dir), "restart", container],
        check=True, capture_output=True, text=True, timeout=300,
    )
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            if urllib.request.urlopen(health_url, timeout=5).status == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def main(argv: list[str] | None = None, *, probes=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--container", default="doc-organizer")
    parser.add_argument("--threshold-mb", type=int, default=DEFAULT_THRESHOLD_MB)
    parser.add_argument("--compose-file", type=Path, default=_REPO / "docker-compose.yml")
    parser.add_argument("--project-dir", type=Path, default=_REPO)
    parser.add_argument("--health-url", default="http://127.0.0.1:7788/health")
    parser.add_argument("--dry-run", action="store_true", help="decide and report; never restart")
    args = parser.parse_args(argv)
    p = probes or {"rss": server_rss_mb, "sweep": sweep_running, "healthy": container_healthy, "restart": restart}

    rss = p["rss"](args.container)
    action, reason = decide(rss, args.threshold_mb, p["sweep"](args.container), p["healthy"](args.container))
    report = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "container": args.container, "rss_mb": rss,
              "threshold_mb": args.threshold_mb, "action": action, "reason": reason, "dry_run": args.dry_run}
    if action == "restart" and not args.dry_run:
        report["healthy_after"] = p["restart"](args.container, args.compose_file, args.project_dir, args.health_url)
    print(json.dumps(report), flush=True)
    return 0 if action == "skip" or args.dry_run or report.get("healthy_after") else 1


if __name__ == "__main__":
    sys.exit(main())
