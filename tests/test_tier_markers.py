"""The tier system: markers are auto-derived from filename conventions."""
import subprocess, sys
from pathlib import Path


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
    assert "_live" not in out


def test_live_tier_collects_live_files():
    out = _collect("live")
    assert "_live" in out
