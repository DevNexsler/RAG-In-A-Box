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
