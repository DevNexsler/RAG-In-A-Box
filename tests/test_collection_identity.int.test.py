"""Real pytest CLI must never silently collect a different module."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("integration_first", [True, False])
@pytest.mark.parametrize("colliding", [True, False])
def test_collection_preserves_module_identity(tmp_path, integration_first, colliding):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "__init__.py").touch()
    (tests / "conftest.py").write_text((ROOT / "tests/conftest.py").read_text())
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\naddopts="--import-mode=importlib"\n'
    )
    unit = tests / "test_example.py"
    integration = tests / (
        "test_example.int.test.py" if colliding else "test_example_worker.int.test.py"
    )
    unit.write_text("def test_unit_visible(): pass\n")
    integration.write_text("def test_integration_visible(): pass\n")
    paths = [integration, unit] if integration_first else [unit, integration]
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTEST_ADDOPTS": ""}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *map(str, paths)],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30,
    )
    output = result.stdout + result.stderr
    if colliding and integration_first:
        assert result.returncode == 4, output
        assert "module identity collision" in output
        assert "test_example.py" in output
    else:
        assert result.returncode == 0, output
        assert "2 tests collected" in output
        assert "test_unit_visible" in output
        assert "test_integration_visible" in output
