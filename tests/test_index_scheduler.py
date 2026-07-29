"""In-process index scheduler: periodic queue-drain + periodic full sweep.

The self-heal machinery (durable queue drain, degraded re-queue during the full
sweep) already exists but nothing triggered it on a schedule — the drain ran
only when a later request or sweep happened to take the write lock, so a
force-index could sit for 10+ minutes and thin docs never re-enriched. This
scheduler is the missing trigger. The tick logic is pure and clock-injected so
the interval behavior is tested without threads or real time.
"""

from core.index_scheduler import IndexScheduler


def _scheduler(**overrides):
    calls = {"drain": 0, "sweep": 0}

    def drain_fn():
        calls["drain"] += 1
        return {"status": "drained", "drained": calls["drain"]}

    def sweep_fn():
        calls["sweep"] += 1
        return {"pid": 1000 + calls["sweep"]}

    kwargs = dict(
        drain_interval_s=60,
        sweep_interval_s=3600,
        drain_fn=drain_fn,
        sweep_fn=sweep_fn,
        sweep_running_fn=lambda: False,
    )
    kwargs.update(overrides)
    return IndexScheduler(**kwargs), calls


def test_drain_runs_on_first_tick_but_seeded_sweep_does_not():
    sched, calls = _scheduler()
    sched.seed(now=1000.0)  # seeding suppresses an immediate heavy sweep on boot
    actions = sched.tick(now=1000.0)
    kinds = [a[0] for a in actions]
    assert "drain" in kinds
    assert "sweep" not in kinds
    assert calls == {"drain": 1, "sweep": 0}


def test_drain_respects_its_interval():
    sched, calls = _scheduler()
    sched.seed(now=1000.0)
    sched.tick(now=1000.0)            # drains
    sched.tick(now=1000.0 + 59)      # too soon
    assert calls["drain"] == 1
    sched.tick(now=1000.0 + 61)      # window elapsed
    assert calls["drain"] == 2


def test_sweep_fires_after_its_interval():
    sched, calls = _scheduler()
    sched.seed(now=0.0)
    sched.tick(now=0.0)
    assert calls["sweep"] == 0
    sched.tick(now=3600.0 + 1)
    assert calls["sweep"] == 1


def test_sweep_skipped_when_one_is_already_running():
    running = {"v": True}
    sched, calls = _scheduler(sweep_running_fn=lambda: running["v"])
    sched.seed(now=0.0)
    actions = sched.tick(now=3600.0 + 1)
    assert ("sweep_skipped_running", None) in actions
    assert calls["sweep"] == 0
    # And the interval still advanced, so it does not busy-retry every tick.
    running["v"] = False
    assert calls["sweep"] == 0


def test_disabled_intervals_never_fire():
    sched, calls = _scheduler(drain_interval_s=0, sweep_interval_s=0)
    sched.seed(now=0.0)
    for t in range(0, 100000, 5000):
        sched.tick(now=float(t))
    assert calls == {"drain": 0, "sweep": 0}


def test_drain_error_is_captured_not_raised():
    def boom():
        raise RuntimeError("lock exploded")

    sched, calls = _scheduler(drain_fn=boom)
    sched.seed(now=0.0)
    actions = sched.tick(now=0.0)
    assert any(k == "drain_error" for k, _ in actions)
    # A later tick still tries again — one failure does not wedge the loop.
    actions2 = sched.tick(now=61.0)
    assert any(k == "drain_error" for k, _ in actions2)


def test_sweep_error_is_captured_not_raised():
    def boom():
        raise RuntimeError("spawn failed")

    sched, _ = _scheduler(sweep_fn=boom)
    sched.seed(now=0.0)
    actions = sched.tick(now=3600.0 + 1)
    assert any(k == "sweep_error" for k, _ in actions)


def test_unseeded_scheduler_runs_both_immediately():
    # Without seed(), both are due on the first tick (None last-run = due).
    sched, calls = _scheduler()
    sched.tick(now=5.0)
    assert calls == {"drain": 1, "sweep": 1}


# --- factory gating (config-driven) ---


def test_build_scheduler_is_disabled_by_default():
    from mcp_server import build_index_scheduler

    assert build_index_scheduler({"index_root": "/tmp/x"}) is None
    assert build_index_scheduler(
        {"index_root": "/tmp/x", "scheduler": {"enabled": False}}
    ) is None


def test_build_scheduler_enabled_uses_config_intervals():
    from mcp_server import build_index_scheduler

    sched = build_index_scheduler({
        "index_root": "/tmp/x",
        "scheduler": {"enabled": True, "drain_interval_s": 30, "sweep_interval_s": 900},
    })
    assert sched is not None
    assert sched.drain_interval_s == 30
    assert sched.sweep_interval_s == 900
