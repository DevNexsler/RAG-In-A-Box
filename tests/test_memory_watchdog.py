"""scripts/memory_watchdog.py: restart only over threshold, never mid-sweep, never unhealthy."""

import json

import pytest

import scripts.memory_watchdog as wd


@pytest.mark.parametrize(
    "rss, sweep, healthy, expected",
    [
        (None, False, True, "skip"),      # unreadable -> do nothing
        (6999, False, True, "skip"),      # under threshold
        (7000, False, True, "skip"),      # at threshold
        (7001, True, True, "skip"),       # over, but an index run holds the table
        (7001, False, False, "skip"),     # over, but unhealthy: a restart would hide it
        (7001, False, True, "restart"),
    ],
)
def test_decide(rss, sweep, healthy, expected):
    action, reason = wd.decide(rss, 7000, sweep, healthy)
    assert action == expected, reason


def test_parse_rss_mb_reads_proc_status():
    assert wd.parse_rss_mb("Name:\tpython\nVmRSS:\t 7150592 kB\nThreads:\t161\n") == 6983
    assert wd.parse_rss_mb("Name:\tpython\n") is None


def _probes(rss, sweep=False, healthy=True, restart_ok=True, calls=None):
    calls = calls if calls is not None else []
    return {
        "rss": lambda c: rss,
        "sweep": lambda c: sweep,
        "healthy": lambda c: healthy,
        "restart": lambda *a: (calls.append(a), restart_ok)[1],
    }


def test_main_restarts_over_threshold_and_reports(capsys):
    calls = []
    code = wd.main(["--threshold-mb", "7000"], probes=_probes(7500, calls=calls))
    report = json.loads(capsys.readouterr().out)
    assert code == 0 and report["action"] == "restart" and report["healthy_after"] is True
    assert len(calls) == 1


def test_main_dry_run_never_restarts(capsys):
    calls = []
    code = wd.main(["--threshold-mb", "7000", "--dry-run"], probes=_probes(9000, calls=calls))
    report = json.loads(capsys.readouterr().out)
    assert code == 0 and report["action"] == "restart" and report["dry_run"] is True
    assert calls == [] and "healthy_after" not in report


def test_main_skips_during_a_sweep(capsys):
    calls = []
    code = wd.main(["--threshold-mb", "7000"], probes=_probes(9000, sweep=True, calls=calls))
    report = json.loads(capsys.readouterr().out)
    assert code == 0 and report["action"] == "skip" and calls == []


def test_main_fails_loudly_when_health_does_not_return(capsys):
    code = wd.main(["--threshold-mb", "7000"], probes=_probes(9000, restart_ok=False))
    report = json.loads(capsys.readouterr().out)
    assert code == 1 and report["healthy_after"] is False
