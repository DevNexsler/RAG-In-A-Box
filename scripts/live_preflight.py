#!/usr/bin/env python3
"""Live-tier preflight: verify money/infra prerequisites before `pytest -m live`.

Invoked by scripts/gate.py as a subprocess before the live tier; ONLY exit
code 0 approves the spend. Runs every check (no early exit) so one run reports
every problem at once, printing one line per check:

    ok <name>: <reason>   /   FAIL <name>: <reason>

Checks: OpenRouter/DeepInfra API keys, config_test.yaml present in CWD,
every configured root and database is scratch (not production storage),
LiteLLM OCR/vision aliases available, prod indexer idle, comm-store Postgres
reachable.

`--compose FILE` runs only the scratch-storage check, against the compose
stack an E2E tier is about to start (scripts/gate.py calls it before `up`).
"""
import argparse
import dataclasses
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

import httpx
import psycopg
import yaml
from psycopg.conninfo import conninfo_to_dict

# scripts/ is not a package: make the repo root importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.config import (  # noqa: E402
    filesystem_source_roots,
    load_config,
    resolve_env_reference,
)

PROBE_TIMEOUT_S = 5
PROD_CONTAINER = "doc-organizer"
INDEXER_PID_PATH = "/data/index/indexer.pid"


def build_indexer_pid_probe(
        pid_path: str | Path = INDEXER_PID_PATH,
        proc_root: str | Path = "/proc",
) -> str:
    """Build fail-closed shell probe for an indexer PID inside its container."""
    return f"""
pid_file={shlex.quote(str(pid_path))}
proc_root={shlex.quote(str(proc_root))}
if [ ! -e "$pid_file" ]; then
    printf 'idle\\n'
    exit 0
fi
pid=$(cat "$pid_file") || exit 10
case "$pid" in
    ''|*[!0-9]*|0) printf 'invalid indexer pid file\\n' >&2; exit 11 ;;
esac
status="$proc_root/$pid/status"
if [ ! -e "$status" ]; then
    printf 'idle\\n'
    exit 0
fi
state=$(awk '$1 == "State:" {{ print $2; exit }}' "$status") || exit 12
if [ -z "$state" ]; then
    printf 'cannot read indexer process state\\n' >&2
    exit 13
fi
if [ "$state" = Z ]; then
    printf 'idle\\n'
    exit 0
fi
cmdline=$(tr '\\000' ' ' < "$proc_root/$pid/cmdline") || exit 14
case "$cmdline" in
    *index_vault_flow*|*flow_index_vault*) printf 'active:%s\\n' "$pid" ;;
    *) printf 'pid does not identify indexer process\\n' >&2; exit 15 ;;
esac
""".strip()


INDEXER_PID_PROBE = build_indexer_pid_probe()


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


@dataclasses.dataclass(frozen=True)
class ProductionStorage:
    """Where the production deployment keeps its data on this host."""
    paths: frozenset[Path]      # host paths (resolved)
    volumes: frozenset[str]     # docker volume names
    databases: frozenset[str]   # postgres database names

    def path_conflict(self, path: Path) -> Path | None:
        """The production path that ``path`` is, contains, or lives under.

        Containment counts both ways: a scratch root above production storage
        still scans (and dedupe archives out of) production documents.
        """
        resolved = path.expanduser().resolve()  # dedupe expands ~ in archive_root
        for prod in sorted(self.paths):
            if resolved == prod or prod in resolved.parents or resolved in prod.parents:
                return prod
        return None

    def database_conflict(self, dsn: str) -> str | None:
        database = dsn_database(dsn)
        return database if database in self.databases else None


def dsn_database(dsn: str) -> str | None:
    """Database a libpq connection string names (libpq defaults it to the
    user); None for a value that is not a connection string."""
    try:
        info = conninfo_to_dict(dsn)
    except psycopg.ProgrammingError:
        return None
    return info.get("dbname") or info.get("user")


def config_paths(config: Mapping) -> list[tuple[str, str]]:
    """(key, path) for every filesystem location a config reads or writes."""
    paths = [("index_root", config.get("index_root"))]
    paths += [(f"sources[{name}].root", str(root))
              for name, root in filesystem_source_roots(config).items()]
    paths.append(("dedupe.archive_root", (config.get("dedupe") or {}).get("archive_root")))
    return [(key, str(path)) for key, path in paths if path]


def config_dsns(config: Mapping, env: Mapping[str, str]) -> list[tuple[str, str]]:
    """(key, resolved dsn) for every postgres source of a config."""
    return [(f"sources[{src.get('name')}].dsn", resolve_env_reference(str(src["dsn"]), env))
            for src in config.get("sources") or []
            if isinstance(src, dict) and src.get("type") == "postgres" and src.get("dsn")]


def _container_to_host(path: str, workdir: str, mounts: list[dict]) -> Path:
    """Host path behind a container path: the deepest covering mount wins."""
    container_path = PurePosixPath(workdir) / path
    best = None
    for mount in mounts:
        dest = PurePosixPath(mount["Destination"])
        if container_path == dest or dest in container_path.parents:
            if best is None or len(dest.parts) > len(PurePosixPath(best["Destination"]).parts):
                best = mount
    if best is None:
        return Path(container_path)
    return Path(best["Source"]) / container_path.relative_to(best["Destination"])


def production_storage() -> ProductionStorage:
    """Production storage as the production container itself describes it.

    The docker-compose.yml deployment (with any override) is what the running
    container's mounts and environment are; its config.yaml is the file it
    loads from its working directory. Every mount source, every path that
    config names, and every database its postgres sources resolve to (in the
    container's environment) is production.
    """
    template = "{{json .Mounts}}\n{{json .Config.WorkingDir}}\n{{json .Config.Env}}"
    proc = subprocess.run(["docker", "inspect", "--format", template, PROD_CONTAINER],
                          capture_output=True, text=True, timeout=PROBE_TIMEOUT_S * 2)
    if proc.returncode != 0:
        raise RuntimeError(f"cannot inspect production container {PROD_CONTAINER} "
                           f"({proc.stderr.strip()[:120]})")
    mounts_json, workdir_json, env_json = proc.stdout.splitlines()[:3]
    # A tmpfs mount has no host Source; Path("") would resolve to our CWD.
    mounts = [m for m in json.loads(mounts_json) or [] if m.get("Source")]
    workdir = json.loads(workdir_json) or "/"
    env = dict(item.split("=", 1) for item in json.loads(env_json) or [] if "=" in item)
    config_file = _container_to_host("config.yaml", workdir, mounts)
    config = yaml.safe_load(config_file.read_text()) or {}
    paths = {Path(mount["Source"]).resolve() for mount in mounts}
    paths |= {_container_to_host(path, workdir, mounts).resolve()
              for _, path in config_paths(config)}
    return ProductionStorage(
        paths=frozenset(paths),
        volumes=frozenset(m["Name"] for m in mounts if m.get("Type") == "volume"),
        databases=frozenset(filter(None, (dsn_database(dsn)
                                          for _, dsn in config_dsns(config, env)))),
    )


def _storage_verdict(problems: list[str], checked: str) -> tuple[bool, str]:
    if problems:
        return False, ("production storage — point every root and DSN at scratch: "
                       + "; ".join(problems))
    return True, f"no production storage in {checked}"


def check_scratch_storage() -> tuple[bool, str]:
    """Every config the live tier loads resolves to scratch storage.

    Live tests load CWD config.yaml and config_test.yaml in-process, so each is
    resolved with load_config itself (INDEX_ROOT/DOCUMENTS_ROOT overrides
    included) and the DSNs with this process's environment.
    """
    try:
        production = production_storage()
    except RuntimeError as exc:
        return False, str(exc)
    config_yaml = next((c for c in config_candidates() if c.exists()), Path("config.yaml"))
    problems = []
    for path in (config_yaml, Path("config_test.yaml")):
        try:
            config = load_config(path)
        except (FileNotFoundError, ValueError) as exc:
            problems.append(f"{path}: cannot resolve ({exc})")
            continue
        for key, root in config_paths(config):
            hit = production.path_conflict(Path(root))
            if hit:
                problems.append(f"{path} {key}={root} overlaps {hit}")
        for key, dsn in config_dsns(config, os.environ):
            database = production.database_conflict(dsn)
            if database:
                problems.append(f"{path} {key} names production database {database!r}")
    return _storage_verdict(problems, f"{config_yaml}, config_test.yaml")


def check_compose_storage(compose_file: str) -> tuple[bool, str]:
    """The compose stack an E2E tier starts uses no production storage.

    Inside a container a configured root persists only through a mount, and a
    `${VAR}` DSN resolves only from the service environment, so checking every
    service's mounts and environment covers every root its config can name.
    """
    try:
        production = production_storage()
    except RuntimeError as exc:
        return False, str(exc)
    proc = subprocess.run(["docker", "compose", "-f", compose_file, "config", "--format", "json"],
                          capture_output=True, text=True, timeout=PROBE_TIMEOUT_S * 6)
    if proc.returncode != 0:
        return False, f"cannot render {compose_file} ({proc.stderr.strip()[:120]})"
    rendered = json.loads(proc.stdout)
    volumes = rendered.get("volumes") or {}
    problems = []
    for service, spec in (rendered.get("services") or {}).items():
        for mount in spec.get("volumes") or []:
            source, hit = mount.get("source"), None
            if mount.get("type") == "bind" and source:
                hit = production.path_conflict(Path(source))
            elif mount.get("type") == "volume" and source:
                name = (volumes.get(source) or {}).get("name", source)
                hit = name if name in production.volumes else None
            if hit:
                problems.append(f"{service} mount {mount.get('target')} uses {hit}")
        for key, value in (spec.get("environment") or {}).items():
            database = production.database_conflict(str(value or ""))
            if database:
                problems.append(f"{service} env {key} names production database {database!r}")
    return _storage_verdict(problems, compose_file)


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
    cmd = ["docker", "exec", PROD_CONTAINER, "sh", "-c", INDEXER_PID_PROBE]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=PROBE_TIMEOUT_S * 2)
    except FileNotFoundError:
        return False, "docker binary missing — cannot verify prod indexer PID"
    except subprocess.TimeoutExpired:
        return False, "docker exec timed out — cannot verify prod indexer PID"
    if proc.returncode != 0:
        detail = proc.stderr.strip()[:120]
        return False, ("cannot verify prod indexer PID"
                       + (f" ({detail})" if detail else ""))
    result = proc.stdout.strip()
    if result == "idle":
        return True, "prod indexer idle (no live indexer PID)"
    if result.startswith("active:"):
        raw_pid = result.removeprefix("active:")
        if raw_pid.isdigit() and int(raw_pid) > 0:
            return False, f"prod indexer active (pid {raw_pid}); rerun later"
    return False, f"unexpected prod indexer probe output: {result[:120]!r}"


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
    ("scratch_storage", check_scratch_storage),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compose", metavar="FILE",
                        help="check only the storage of this compose stack (E2E tiers)")
    args = parser.parse_args()
    if args.compose:
        CHECKS = [("scratch_storage", lambda: check_compose_storage(args.compose))]
    sys.exit(main())
