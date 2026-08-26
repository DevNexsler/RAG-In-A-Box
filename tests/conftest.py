from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _isolate_process_global_resilience_state():
    """Two pieces of state outlive a single call and are process-global by design:
    the thread-local degradation capture (production brackets every document with
    begin/collect) and the per-endpoint circuit breaker (an outage is remembered
    across documents). Reset both per test so one test's simulated outage cannot
    leak into the next one's assertions."""
    from core.resilience import CIRCUITS
    from extractors import begin_degradation_capture

    begin_degradation_capture()
    CIRCUITS.reset()
    yield
    CIRCUITS.reset()


def pytest_collection_modifyitems(config, items):
    for item in items:
        fname = item.fspath.basename
        # An explicit marker always wins. After that the location decides before
        # the name does: a file under tests/e2e/ drives the staging stack, so it
        # is e2e even when its name happens to contain "_live". The name rule
        # then matches the "_live.py" suffix rather than a bare substring, so
        # only files that really are named for the live tier land in it.
        if item.get_closest_marker("live"):
            item.add_marker(pytest.mark.live)
        elif item.get_closest_marker("e2e") or ".e2e.test" in fname or "/tests/e2e/" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif fname.endswith("_live.py"):
            item.add_marker(pytest.mark.live)
        elif item.get_closest_marker("integration") or ".int.test" in fname:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)
