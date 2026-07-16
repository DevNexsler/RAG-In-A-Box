#!/usr/bin/env python3
"""Live-tier preflight: verify money/infra prerequisites before `pytest -m live`.

Invoked by scripts/gate.py as a subprocess before the live tier; ONLY exit
code 0 approves the spend. Runs every check (no early exit) so one run reports
every problem at once, printing one line per check:

    ok <name>: <reason>   /   FAIL <name>: <reason>

Checks: OpenRouter/DeepInfra API keys, config_test.yaml present in CWD,
LiteLLM OCR/vision aliases available, prod indexer idle, comm-store Postgres
reachable.
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
# tier for shared OCR/vision providers — so we refuse to start.
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
    """Real config.yaml: prefer CWD's, fall back to the main checkout's."""
    return [Path("config.yaml"), main_checkout_root() / "config.yaml"]


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


def check_litellm_ocr() -> tuple[bool, str]:
    config = None
    used = None
    for candidate in config_candidates():
        if candidate.exists():
            config = yaml.safe_load(candidate.read_text()) or {}
            used = candidate
            break
    if config is None:
        return False, "no config.yaml found (CWD or main checkout)"
    if not isinstance(config, dict):
        return False, f"config root in {used} must be a mapping"

    ocr = config.get("ocr") or {}
    if not isinstance(ocr, dict):
        return False, f"ocr config in {used} must be a mapping"
    if ocr.get("provider") != "litellm":
        return False, f"ocr.provider must be litellm in {used}"

    required = ("endpoint", "extract_model", "describe_model")
    missing = [field for field in required if not ocr.get(field)]
    if missing:
        return False, f"ocr config missing in {used}: {', '.join(missing)}"
    if any(not isinstance(ocr[field], str) for field in required):
        return False, f"ocr config fields must be strings in {used}"

    api_key = (os.environ.get("LITELLM_API_KEY")
               or os.environ.get("LITELLM_MASTER_KEY"))
    if not api_key:
        return False, ("LITELLM_API_KEY or LITELLM_MASTER_KEY missing or empty "
                       "(set in .env)")

    endpoint = ocr["endpoint"].rstrip("/")
    models_url = f"{endpoint}/models"
    try:
        response = httpx.get(
            models_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=PROBE_TIMEOUT_S,
        )
        response.raise_for_status()
    except httpx.InvalidURL:
        return False, f"LiteLLM endpoint invalid at {models_url}"
    except httpx.HTTPStatusError as exc:
        return False, (f"LiteLLM model list returned HTTP "
                       f"{exc.response.status_code} at {models_url}")
    except httpx.HTTPError as exc:
        return False, (f"LiteLLM unreachable at {models_url}: "
                       f"{type(exc).__name__}")
    try:
        payload = response.json()
    except ValueError:
        return False, "LiteLLM model list returned invalid JSON"

    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return False, "LiteLLM model list returned invalid schema"
    model_rows = payload["data"]
    if any(not isinstance(row, dict) or not isinstance(row.get("id"), str)
           for row in model_rows):
        return False, "LiteLLM model list returned invalid schema"

    model_ids = {row["id"] for row in model_rows}
    aliases = [ocr["extract_model"], ocr["describe_model"]]
    missing_aliases = [alias for alias in aliases if alias not in model_ids]
    if missing_aliases:
        return False, ("LiteLLM model list missing configured alias(es): "
                       + ", ".join(missing_aliases))
    return True, "LiteLLM aliases available: " + ", ".join(aliases)


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
        # contending for shared LiteLLM/local inference hardware. Surface
        # stderr so operators can tell
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
    ("litellm_ocr", check_litellm_ocr),
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
