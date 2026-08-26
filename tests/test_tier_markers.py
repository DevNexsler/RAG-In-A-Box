"""The tier system: markers are auto-derived from filename conventions."""
import subprocess, sys
from pathlib import Path

from tests import conftest as tier_conftest


def _collect(marker):
    # cwd is pinned to the repo root so these tests pass from any directory.
    out = subprocess.run(
        [
            sys.executable, "-m", "pytest", "--collect-only", "-q",
            "-m", marker, "tests/",
        ],
        capture_output=True, text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    # Markers derive from FILENAME conventions, so assert on the file-path part
    # of each node id only — test *function* names may legitimately contain
    # "_live" etc. (e.g. test_gate_runner.py::test_live_requires_all_prior).
    return "\n".join(
        line.split("::")[0] for line in out.stdout.splitlines() if "::" in line
    )


class _StubPath:
    """`item.fspath` stand-in: the classifier reads `.basename` and `str()`."""

    def __init__(self, relpath):
        self._relpath = relpath
        self.basename = relpath.rsplit("/", 1)[-1]

    def __str__(self):
        return f"/repo/{self._relpath}"


class _StubItem:
    """`pytest.Item` stand-in carrying only what the classifier reads."""

    def __init__(self, relpath, explicit):
        self.fspath = _StubPath(relpath)
        self._explicit = set(explicit)
        self.tiers = []

    def get_closest_marker(self, name):
        return name if name in self._explicit else None

    def add_marker(self, marker):
        self.tiers.append(marker.name)


def _classify(relpath, explicit=()):
    """Tier(s) the real classifier assigns to a hypothetical test file. Lets us
    cover paths/names that do not exist in the tree without adding dead tests."""
    item = _StubItem(relpath, explicit)
    tier_conftest.pytest_collection_modifyitems(None, [item])
    return item.tiers


def test_integration_tier_matches_filenames():
    out = _collect("integration")
    assert ".int.test" in out
    assert "_live" not in out          # live files never in integration tier
    # (covers the renamed test_multi_source_flow_live.py too, via the _live check)


def test_unit_tier_excludes_special_files():
    out = _collect("unit")
    assert "test_config.py" in out     # positive check: collection is not empty
    assert ".int.test" not in out
    assert ".e2e.test" not in out
    # Suffix, not substring: tests/test_live_preflight.py is a hermetic unit
    # test whose name merely contains "_live".
    assert not any(line.endswith("_live.py") for line in out.splitlines())


def test_live_tier_collects_live_files():
    out = _collect("live")
    assert "_live" in out


def test_e2e_directory_wins_over_a_live_filename_substring():
    """#1654: tests/e2e/test_health_liveness.py contains "_live", so a substring
    match tiered it live. That dragged tests/e2e/conftest.py's session-scoped
    staging-stack probe into the live tier, which runs after the stack is torn
    down — and the probe pytest.exit()s the whole session."""
    assert "tests/e2e/test_health_liveness.py" in _collect("e2e")
    assert "tests/e2e/test_health_liveness.py" not in _collect("live")


def test_live_tier_is_keyed_on_the_filename_suffix():
    assert _classify("tests/test_media_live.py") == ["live"]
    assert _classify("tests/test_live_preflight.py") == ["unit"]
    assert _classify("tests/e2e/test_health_liveness.py") == ["e2e"]


def test_explicit_live_marker_wins_over_the_e2e_directory():
    assert _classify("tests/e2e/test_provider_cost.py", explicit=["live"]) == ["live"]
