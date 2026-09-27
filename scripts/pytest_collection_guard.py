"""Reject importlib namespace collisions instead of silently dropping tests."""
from pathlib import Path

import pytest


_MODULES = pytest.StashKey[list[pytest.Module]]()


def pytest_collectstart(collector):
    if isinstance(collector, pytest.Module):
        collector.session.stash.setdefault(_MODULES, []).append(collector)


def pytest_collection_finish(session):
    for collector in session.stash.get(_MODULES, []):
        # Failed/skipped imports already have their own collection report. Do
        # not retry them, or turn normal import errors into misleading failures.
        module = collector.__dict__.get("_obj")
        if module is None:
            continue
        imported_path = getattr(module, "__file__", None)
        if imported_path is None or Path(imported_path).resolve() != collector.path.resolve():
            raise pytest.UsageError(
                f"module identity collision: {collector.path} resolved to "
                f"{imported_path!r}; give dotted integration filenames a unique stem"
            )
