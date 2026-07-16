#!/usr/bin/env python3
"""Live-tier preflight: verify money/infra prerequisites before `pytest -m live`.

Invoked by scripts/gate.py as a subprocess before the live tier; ONLY exit
code 0 approves the spend. Runs every check (no early exit) so one run reports
every problem at once, printing one line per check:

    ok <name>: <reason>   /   FAIL <name>: <reason>

Checks: OpenRouter/DeepInfra API keys, config_test.yaml present in CWD,
Mac Mini OCR endpoint reachable, prod indexer idle (don't contend for the
Mac), comm-store Postgres reachable.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import psycopg
import yaml

PROBE_TIMEOUT_S = 5
# An indexing run touches the heartbeat continuously; if it is fresher than
# this, the prod indexer is actively writing and would contend with the live
# tier for the shared Mac Mini (OCR/vision) — so we refuse to start.
HEARTBEAT_ACTIVE_THRESHOLD_S = 120
PROD_CONTAINER = "doc-organizer"
HEARTBEAT_PATH = "/data/index/indexer.heartbeat"


def main_checkout_root() -> Path:
    """The main checkout (worktrees live under <main>/.worktrees/<name>)."""
    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        if parent.name == ".worktrees":
            return parent.parent
    return cwd


def config_candidates() -> list[Path]:
    """Real config.yaml: prefer the main checkout's, fall back to CWD's."""
    return [main_checkout_root() / "config.yaml", Path("config.yaml")]


def _load_env() -> None:
    # Same pattern as the live tests: dotenv if available, silently otherwise.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def check_api_keys() -> tuple[bool, str]:
    missing = [key for key in ("OPENROUTER_API_KEY", "DEEPINFRA_API_KEY")
               if not os.environ.get(key)]
    if missing:
        return False, f"missing or empty: {', '.join(missing)} (set in .env)"
    return True, "OPENROUTER_API_KEY and DEEPINFRA_API_KEY set"


def check_config_test() -> tuple[bool, str]:
    path = Path("config_test.yaml")
    if not path.exists():
        return False, (
            "config_test.yaml not found in CWD (gitignored, not in worktrees) "
            f"— copy it from {main_checkout_root() / 'config_test.yaml'}"
        )
    return True, "config_test.yaml present"


def check_mac_ocr() -> tuple[bool, str]:
    config = None
    used = None
    for candidate in config_candidates():
        if candidate.exists():
            config = yaml.safe_load(candidate.read_text()) or {}
            used = candidate
            break
    if config is None:
        return False, "no config.yaml found (main checkout or CWD)"
    ocr = config.get("ocr") or {}
    describe = ocr.get("describe") if isinstance(ocr.get("describe"), dict) else {}
    extract = ocr.get("extract") if isinstance(ocr.get("extract"), dict) else {}
    base_url = (
        ocr.get("base_url")
        or ocr.get("endpoint")
        or describe.get("base_url")
        or describe.get("endpoint")
        or extract.get("base_url")
        or extract.get("endpoint")
    )
    if not base_url:
        return False, f"ocr endpoint (base_url/endpoint) missing in {used}"
    try:
        # Reachability only: any HTTP response (404 included) means the Mac
        # Mini OCR service host is up.
        httpx.get(base_url, timeout=PROBE_TIMEOUT_S)
    except httpx.HTTPError as exc:
        return False, f"Mac OCR unreachable at {base_url}: {exc}"
    return True, f"Mac OCR reachable at {base_url}"


def check_prod_indexer_idle() -> tuple[bool, str]:
    cmd = ["docker", "exec", PROD_CONTAINER, "cat", HEARTBEAT_PATH]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=PROBE_TIMEOUT_S * 2)
    except FileNotFoundError:
        return True, ("WARNING: docker binary missing — cannot check prod "
                      "indexer heartbeat; assuming idle")
    except subprocess.TimeoutExpired:
        return True, ("WARNING: docker exec timed out — cannot verify prod "
                      "indexer heartbeat; assuming idle")
    if proc.returncode != 0:
        # Container not running, or heartbeat file absent: nothing is
        # contending for the Mac. Surface stderr so operators can tell
        # container-not-running vs daemon-unreachable vs file-absent apart.
        detail = proc.stderr.strip()[:120]
        return True, ("prod container not running or no heartbeat file"
                      + (f" ({detail})" if detail else ""))
    try:
        age = time.time() - float(proc.stdout.strip())
    except ValueError:
        return True, "heartbeat unreadable; assuming idle"
    if age < HEARTBEAT_ACTIVE_THRESHOLD_S:
        return False, (f"prod indexer active (heartbeat {age:.0f}s old, "
                       f"threshold {HEARTBEAT_ACTIVE_THRESHOLD_S}s); rerun later")
    return True, f"prod indexer idle (heartbeat {age:.0f}s old)"


def check_comm_postgres() -> tuple[bool, str]:
    dsn = os.environ.get("COMM_DATA_STORE_DSN")
    if not dsn:
        return False, "COMM_DATA_STORE_DSN not set (required by live sor tests)"
    try:
        with psycopg.connect(dsn, connect_timeout=PROBE_TIMEOUT_S) as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.fetchone()
    except psycopg.Error as exc:
        return False, f"comm-store Postgres unreachable: {exc}"
    return True, "comm-store Postgres answered SELECT 1"


CHECKS = [
    ("api_keys", check_api_keys),
    ("config_test", check_config_test),
    ("mac_ocr", check_mac_ocr),
    ("prod_indexer_idle", check_prod_indexer_idle),
    ("comm_postgres", check_comm_postgres),
]


def main() -> int:
    _load_env()
    failed = 0
    for name, check in CHECKS:
        try:
            ok, reason = check()
        except Exception as exc:  # a crashed probe must never approve spend
            ok, reason = False, f"check crashed: {exc}"
        print(f"{'ok' if ok else 'FAIL'} {name}: {reason}", flush=True)
        if not ok:
            failed += 1
    if failed:
        print(f"live preflight: FAIL ({failed} check(s) failed)", flush=True)
    else:
        print("live preflight: PASS", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
