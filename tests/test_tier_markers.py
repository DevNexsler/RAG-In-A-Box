"""The tier system: markers are auto-derived from filename conventions."""
import subprocess, sys


def _collect(marker):
    # --ignore self: this file's own test IDs contain "_live" and would
    # pollute the substring assertions below.
    out = subprocess.run(
        [
            sys.executable, "-m", "pytest", "--collect-only", "-q",
            "-m", marker, "--ignore", "tests/test_tier_markers.py", "tests/",
        ],
        capture_output=True, text=True,
    )
    return out.stdout


def test_integration_tier_matches_filenames():
    out = _collect("integration")
    assert ".int.test" in out
    assert "_live" not in out          # live files never in integration tier
    # (covers the renamed test_multi_source_flow_live.py too, via the _live check)


def test_unit_tier_excludes_special_files():
    out = _collect("unit")
    assert ".int.test" not in out
    assert ".e2e.test" not in out
    assert "_live" not in out


def test_live_tier_collects_live_files():
    out = _collect("live")
    assert "_live" in out
