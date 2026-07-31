"""The one place doc-organizer configures logging — and the guarantee that
**one log record is one physical line**.

Why this module exists (#0546): a provider exception's ``str()`` can contain a
newline (litellm's ``RateLimitError`` ends with ``\\nFor more information check:
<mdn url>``). Interpolated verbatim into a retry warning, every rate-limit retry
wrote a second *untimestamped, unattributable* physical line into
``indexer.log`` — no timestamp, no level, no logger. Those orphans are invisible
to the in-container pattern watcher, to ``grep '^2026-'`` and to any line-based
parser, and during a 429 storm they were 27% of a nightly log capture, evicting
most of the reviewable window. Truncating the *formatted* message (the previous
mitigation) mangled the payload mid-URL and did not remove the split.

Three seams, cheapest first:

1. :func:`collapse` — flatten whitespace *then* truncate, at the site that
   interpolates untrusted text (an exception, a provider body) into a message.
   Order matters: truncate-first can leave the newline inside the prefix.
2. :class:`SingleLineFormatter` — the backstop. Whatever the source (message,
   traceback, third-party logger), the *formatted* record is joined into one
   line, so the invariant holds for code that forgot seam 1.
3. :class:`RepeatCollapseFilter` — a flood of the same failure collapses to one
   line per window with a count, so a provider outage costs a couple of lines
   instead of thousands.

:func:`configure_logging` wires all three plus ``logging.captureWarnings`` (so
library warnings — pydub's missing-ffmpeg ``RuntimeWarning`` — become normal
timestamped records instead of raw stderr). Every entrypoint calls it instead of
``logging.basicConfig``.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# Joiner for a record that was multi-line before collapsing (e.g. a traceback):
# keeps the boundary visible while staying greppable.
LINE_JOIN = " | "

# Cap for untrusted text interpolated into a message (provider error bodies run
# to kilobytes). Generous enough that a typical 429/5xx body survives whole.
MAX_ERROR_CHARS = 500

# Repeat-collapse window: one line per (site, key) per minute during a storm.
DEFAULT_DEDUP_WINDOW = 60.0

# ``extra=`` attribute a call site sets to opt into repeat collapsing.
DEDUP_KEY = "dedup_key"

_TRUNCATION_MARK = "..."


def collapse(text: object, limit: int | None = None) -> str:
    """Flatten every run of whitespace in ``text`` to a single space, then truncate.

    Use at any site that interpolates text you do not control (an exception, an
    HTTP body) into a log message. ``text`` may be an exception — it is stringified.
    """
    flat = " ".join(str(text).split())
    if limit is not None and len(flat) > limit:
        flat = flat[:limit].rstrip() + _TRUNCATION_MARK
    return flat


class SingleLineFormatter(logging.Formatter):
    """Formatter that guarantees one record renders as exactly one physical line.

    Newlines anywhere in the formatted record — message, exception text, stack —
    become ``LINE_JOIN``. Blank lines are dropped and each fragment stripped, so a
    traceback becomes one dense, greppable line rather than N orphan lines.
    """

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        if "\n" in formatted or "\r" in formatted:
            formatted = LINE_JOIN.join(
                part for part in (line.strip() for line in formatted.splitlines()) if part
            )
        return formatted


class RepeatCollapseFilter(logging.Filter):
    """Collapse a flood of identical records into one line per window, with a count.

    A record opts in by carrying a ``dedup_key`` (``logger.warning(..., extra={
    "dedup_key": (label, type(exc).__name__)})``) — the key is what "identical"
    means for that site, e.g. (provider, op, error class), deliberately ignoring
    the attempt number and backoff delay. The first record of a window is emitted;
    the rest are counted, and the count rides out on the next record that passes:
    ``… — retrying in 2s (+143 suppressed in the last 61s)``.

    Records with no ``dedup_key`` are never suppressed: per-doc chatter stays
    per-doc, and a site must ask to be collapsed.
    """

    def __init__(
        self,
        window_seconds: float = DEFAULT_DEDUP_WINDOW,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        self._window = float(window_seconds)
        self._clock = clock
        self._windows: dict[object, tuple[float, int]] = {}
        self._lock = threading.Lock()

    def filter(self, record: logging.LogRecord) -> bool:
        key = getattr(record, DEDUP_KEY, None)
        if key is None:
            return True
        now = self._clock()
        with self._lock:
            opened, suppressed = self._windows.get(key, (None, 0))
            if opened is not None and now - opened < self._window:
                self._windows[key] = (opened, suppressed + 1)
                return False
            self._windows[key] = (now, 0)
        if suppressed:
            record.msg = "%s (+%d suppressed in the last %.0fs)" % (
                record.getMessage(), suppressed, now - opened,
            )
            record.args = ()
        return True


def configure_logging(
    level: int | str = logging.WARNING,
    *,
    dedup_window_seconds: float = DEFAULT_DEDUP_WINDOW,
    stream=None,
    fmt: str = DEFAULT_FORMAT,
) -> None:
    """Install the one-record-one-line root handler. Call once, at entrypoint start.

    Replaces ``logging.basicConfig`` everywhere: same job, plus the single-line
    invariant, repeat collapsing and ``warnings`` capture. Re-callable (later calls
    replace the root handlers) so a test can point it at a ``StringIO``.
    """
    if isinstance(level, str):
        level = getattr(logging, level.strip().upper(), logging.WARNING)

    handler = logging.StreamHandler(stream)
    handler.setFormatter(SingleLineFormatter(fmt))
    handler.addFilter(RepeatCollapseFilter(dedup_window_seconds))
    # force=True: an imported library may have called basicConfig first, which
    # would otherwise make ours a no-op and silently lose the invariant.
    logging.basicConfig(level=level, handlers=[handler], force=True)

    # Library warnings (pydub's missing-ffmpeg RuntimeWarning et al.) become
    # timestamped records on the "py.warnings" logger instead of raw stderr.
    logging.captureWarnings(True)


def configure_logging_from_config(config: dict, **kwargs) -> None:
    """``configure_logging`` driven by the ``logging:`` block of a loaded config."""
    log_cfg = config.get("logging") or {}
    kwargs.setdefault(
        "dedup_window_seconds",
        float(log_cfg.get("dedup_window_seconds", DEFAULT_DEDUP_WINDOW)),
    )
    configure_logging(log_cfg.get("level", "WARNING"), **kwargs)
