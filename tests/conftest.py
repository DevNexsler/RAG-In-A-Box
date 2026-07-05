from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_collection_modifyitems(config, items):
    for item in items:
        fname = item.fspath.basename
        if item.get_closest_marker("live") or "_live" in fname:
            item.add_marker(pytest.mark.live)
        elif item.get_closest_marker("e2e") or ".e2e.test" in fname or "/tests/e2e/" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif item.get_closest_marker("integration") or ".int.test" in fname:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)
