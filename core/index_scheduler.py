"""In-process scheduler that triggers the index self-heal on a timer.

The durable queue drain and the degraded-doc re-queue (during the full sweep)
already exist; before this, nothing ran them on a schedule. The drain fired only
when a fresh request or sweep happened to take the write lock, so a targeted
force-index could sit for many minutes, and thin/degraded docs never re-enriched
until someone manually kicked a sweep.

Three jobs, three very different cadences:
  * queue drain  — short interval (~60s): keep targeted requests moving.
  * full sweep   — long interval (~hourly): rescan + re-enrich degraded docs.
  * compaction   — long interval (~hourly), at most one rewrite per calendar
    day: the daily Lance data compaction, which forks a full-memory worker and
    is therefore only safe between runs (#1254).

The scheduling decision lives in the pure, clock-injected ``tick`` so interval
behavior is testable without threads or real time. ``run_forever`` is the thin
threaded wrapper the server starts.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


class IndexScheduler:
    def __init__(
        self,
        *,
        drain_interval_s: float,
        sweep_interval_s: float,
        drain_fn: Callable[[], Any],
        sweep_fn: Callable[[], Any],
        sweep_running_fn: Callable[[], bool],
        run_was_interrupted_fn: Callable[[], bool] | None = None,
        compact_interval_s: float = 0.0,
        compact_fn: Callable[[], Any] | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        self.drain_interval_s = drain_interval_s
        self.sweep_interval_s = sweep_interval_s
        self.compact_interval_s = compact_interval_s if compact_fn else 0.0
        self._drain_fn = drain_fn
        self._sweep_fn = sweep_fn
        self._sweep_running_fn = sweep_running_fn
        self._run_was_interrupted_fn = run_was_interrupted_fn or (lambda: False)
        self._compact_fn = compact_fn
        self._log = log or logger
        self._last_drain: float | None = None
        self._last_sweep: float | None = None
        self._last_compact: float | None = None
        self._stop = threading.Event()

    def _run_was_interrupted(self) -> bool:
        """True when the previous run ended without completing. Never raises —
        an unreadable supervisor state must not turn every boot into a sweep."""
        try:
            return bool(self._run_was_interrupted_fn())
        except Exception as exc:
            self._log.warning("interrupted-run check failed: %s", exc)
            return False

    def seed(self, now: float) -> "IndexScheduler":
        """Suppress an immediate full sweep on boot (it is heavy and would run on
        every container restart). The drain is intentionally left due so a boot
        backlog clears within one drain interval, and so is the compaction: a
        just-booted container is the idlest window there is, and the daily marker
        already stops it repeating on a restart loop.

        Exception: a boot that follows an *interrupted* run leaves the sweep due
        (#1153). A restart landing on a live run — the container's memory cgroup
        OOM-killing the server is the observed cause — otherwise parks that run's
        remaining documents for a whole sweep interval, and /health stays 503 on
        the unresolved terminal state for just as long. Resuming here re-scans
        and re-queues only what is still outstanding; documents the dead run
        already indexed diff out. Boot is also the cheapest moment to sweep: the
        server process has just restarted, so the container is at its memory
        floor rather than its ceiling.
        """
        self._last_sweep = None if self._run_was_interrupted() else now
        return self

    @staticmethod
    def _due(last: float | None, interval: float, now: float) -> bool:
        return interval > 0 and (last is None or now - last >= interval)

    def tick(self, now: float) -> list[tuple[str, Any]]:
        """Run whichever jobs are due. Pure except for the injected job fns;
        never raises — a failing job is logged and reported, not propagated,
        so one bad run cannot wedge the loop."""
        actions: list[tuple[str, Any]] = []

        if self._due(self._last_drain, self.drain_interval_s, now):
            self._last_drain = now
            try:
                actions.append(("drain", self._drain_fn()))
            except Exception as exc:
                self._log.warning("scheduled drain failed: %s", exc)
                actions.append(("drain_error", str(exc)))

        if self._due(self._last_sweep, self.sweep_interval_s, now):
            # Advance the clock even when skipping, so a running sweep does not
            # cause a busy-retry on every tick.
            self._last_sweep = now
            if self._sweep_running_fn():
                actions.append(("sweep_skipped_running", None))
            else:
                try:
                    actions.append(("sweep", self._sweep_fn()))
                except Exception as exc:
                    self._log.warning("scheduled sweep failed: %s", exc)
                    actions.append(("sweep_error", str(exc)))

        if self._due(self._last_compact, self.compact_interval_s, now):
            self._last_compact = now
            try:
                actions.append(("compact", self._compact_fn()))
            except Exception as exc:
                self._log.warning("scheduled compaction failed: %s", exc)
                actions.append(("compact_error", str(exc)))

        return actions

    def stop(self) -> None:
        self._stop.set()

    def run_forever(
        self,
        *,
        poll_s: float = 5.0,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.seed(clock())
        self._log.info(
            "index scheduler started (drain=%ss, sweep=%ss, compact=%ss)",
            self.drain_interval_s,
            self.sweep_interval_s,
            self.compact_interval_s,
        )
        while not self._stop.is_set():
            try:
                self.tick(clock())
            except Exception:  # defensive: tick already swallows job errors
                self._log.exception("index scheduler tick crashed; continuing")
            sleep(poll_s)

    def start_thread(self, **run_kwargs: Any) -> threading.Thread:
        thread = threading.Thread(
            target=self.run_forever,
            kwargs=run_kwargs,
            name="index-scheduler",
            daemon=True,
        )
        thread.start()
        return thread
