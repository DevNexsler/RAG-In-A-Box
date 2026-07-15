"""Deployment config contract tests."""

from pathlib import Path

import yaml


def test_docker_compose_uses_env_vars_for_database_dsns():
    """Compose must not inline database credentials."""
    compose_path = Path("docker-compose.yml")
    raw = compose_path.read_text()

    assert "postgresql://admin:" not in raw

    compose = yaml.safe_load(raw)
    env = compose["services"]["doc-organizer"]["environment"]

    assert "SOR_DSN=${SOR_DSN}" in env
    for item in env:
        if isinstance(item, str) and item.startswith(("SOR_DSN=", "COMM_DATA_STORE_DSN=")):
            assert "${" in item, f"{item.split('=', 1)[0]} must come from environment"


def test_docker_compose_raises_doc_organizer_nofile_limit():
    """Indexer concurrency needs a higher FD limit than Docker's default 1024."""
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())
    nofile = compose["services"]["doc-organizer"]["ulimits"]["nofile"]

    assert nofile["soft"] >= 65535
    assert nofile["hard"] >= nofile["soft"]


def test_docker_compose_bounds_lance_retention_and_monitors_disk():
    """Compose must bound Lance MVCC garbage (#0232): a short version-prune
    window, a small daily restore-point count (every retained tag pins that
    day's data files against reclaim), and a disk high-water threshold so
    /health surfaces pressure before ENOSPC."""
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())
    env = compose["services"]["doc-organizer"]["environment"]

    assert "LANCE_VERSION_RETENTION_MINUTES=${LANCE_VERSION_RETENTION_MINUTES:-30}" in env
    assert "LANCE_DAILY_RESTORE_POINTS=${LANCE_DAILY_RESTORE_POINTS:-7}" in env
    assert "DISK_USAGE_MAX_PERCENT=${DISK_USAGE_MAX_PERCENT:-90}" in env


def test_compose_disables_prefect_ephemeral_server():
    """Production must disable Prefect's ephemeral-server auto-start (#0325).

    The indexer runs as a background subprocess launched by file_index_update.
    With ephemeral mode enabled and no PREFECT_API_URL, every index flow
    auto-starts a throwaway Prefect temporary server that orphans when earlyoom
    kills the indexer before teardown (49+ orphans, ~19 GiB baseline). The
    entrypoint runs one persistent PrefectServer and exports PREFECT_API_URL, so
    disabling ephemeral turns a missing URL into a loud failure instead of a
    silent orphan leak."""
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())
    env = compose["services"]["doc-organizer"]["environment"]

    assert "PREFECT_SERVER_ALLOW_EPHEMERAL_MODE=false" in env


def test_compose_uses_init_and_direct_exec_entrypoint():
    """Container PID 1 must reap children and forward shutdown directly.

    A shell PID 1 cannot reliably reap an index process group's descendants or
    deliver Docker's stop signal to ``server.py``.  Compose must install the
    built-in init and start Python directly, without ``sh -c``.
    """
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())
    service = compose["services"]["doc-organizer"]

    assert service.get("init") is True
    assert service.get("command") == ["python", "server.py"]
    assert service.get("stop_grace_period") == "30s"


def test_compose_enforces_configurable_resource_envelope():
    """Indexer regressions must stay inside the Doc Organizer container.

    Defaults are sized from the production-shaped #0325 profile while keeping
    environment overrides for hosts with a deliberately different capacity.
    """
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())
    service = compose["services"]["doc-organizer"]

    assert service["mem_limit"] == "${DOC_ORGANIZER_MEMORY_LIMIT:-8g}"
    assert service["mem_reservation"] == "${DOC_ORGANIZER_MEMORY_RESERVATION:-4g}"
    assert service["pids_limit"] == "${DOC_ORGANIZER_PIDS_LIMIT:-512}"


def test_example_config_keeps_memory_observability_opt_in():
    """Per-doc sampling must be explicit; default indexing pays no probe cost."""
    config = yaml.safe_load(Path("config.yaml.example").read_text())

    assert config["memory_observability"]["enabled"] is False


def test_dockerfile_declares_health_check_on_health_endpoint():
    """The image must ship a HEALTHCHECK so the container's health is visible
    (docker ps / .State.Health) instead of relying only on an external probe
    (#0127). It must target the /health endpoint, and — the slim image has no
    curl/wget — must not depend on curl/wget."""
    dockerfile = Path("Dockerfile").read_text()

    assert "HEALTHCHECK" in dockerfile, "Dockerfile must declare a HEALTHCHECK"
    healthcheck_line = next(
        line for line in dockerfile.splitlines() if line.startswith("HEALTHCHECK")
    )
    # start-period gives the server time to boot before failures count.
    assert "--start-period" in healthcheck_line

    # The probe body follows the HEALTHCHECK line (CMD ...).
    hc_index = dockerfile.index("HEALTHCHECK")
    hc_block = dockerfile[hc_index:]
    assert "/health" in hc_block
    assert "curl" not in hc_block and "wget" not in hc_block
