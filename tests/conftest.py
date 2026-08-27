from pathlib import Path
import subprocess
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


class RealIndexerLaunchAttempted(BaseException):
    """A unit test tried to launch the real detached indexer.

    Derived from ``BaseException`` so it travels through the production
    ``except Exception`` handler in ``IndexRunSupervisor.start`` instead of
    being recorded as an ordinary launch failure — the test that forgot to
    substitute a launcher must fail, not quietly observe a failed launch.
    """


def _refuse_real_indexer_launch(*_args, **_kwargs):
    raise RealIndexerLaunchAttempted(
        "unit test reached the real detached indexer launcher; pass "
        "popen_factory= to IndexRunSupervisor or patch subprocess.Popen"
    )


@pytest.fixture(autouse=True)
def _no_real_detached_index_runs(request, monkeypatch):
    """Keep the unit tier from spawning a real indexer that outlives it (#1663).

    ``IndexRunSupervisor.start`` launches a ``start_new_session`` child that is
    reparented to init and runs ``index_vault_flow`` against the real config, so
    it issues live provider calls and keeps spending after the pytest session
    dies — nothing in the suite reaps it. Tests that exercise that seam already
    substitute a launcher (``popen_factory=`` or a patched ``subprocess.Popen``);
    this turns *forgetting* to do so from a silent leak into a loud failure
    across the whole tier, rather than one call site at a time. Only the unit tier is
    guarded: the integration tier drives the real launcher on purpose, with
    bounded commands it reaps itself.
    """
    if request.node.get_closest_marker("unit") is None:
        yield
        return

    from index_run_supervisor import IndexRunSupervisor

    real_popen = subprocess.Popen
    original_init = IndexRunSupervisor.__init__

    def guarded_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Resolved to the genuine Popen means neither a popen_factory nor a
        # patch was supplied — this supervisor would spawn for real.
        if self._popen is real_popen:
            self._popen = _refuse_real_indexer_launch

    monkeypatch.setattr(IndexRunSupervisor, "__init__", guarded_init)
    yield


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
