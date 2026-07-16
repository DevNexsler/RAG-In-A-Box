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


def test_docker_compose_forwards_litellm_credentials_to_doc_organizer():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())
    env = compose["services"]["doc-organizer"]["environment"]

    assert "LITELLM_API_KEY=${LITELLM_API_KEY}" in env
    assert "LITELLM_MASTER_KEY=${LITELLM_MASTER_KEY}" in env


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
