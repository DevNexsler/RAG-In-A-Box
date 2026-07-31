"""One log record == one physical line (#0546).

A provider exception whose ``str()`` contains a newline used to write an extra
untimestamped, unattributable physical line into indexer.log — invisible to
``grep '^2026-'`` and to the in-container pattern watcher. These tests pin the
three seams that guarantee it can't happen again, wherever the newline comes
from: the collapse helper, the formatter backstop, and warnings capture.
"""

import io
import logging
import time
import warnings

import pytest

from core.logging_setup import (
    DEFAULT_FORMAT,
    RepeatCollapseFilter,
    SingleLineFormatter,
    collapse,
    configure_logging,
)


@pytest.fixture(autouse=True)
def _restore_root_logging():
    """configure_logging mutates the root logger — don't leak it into other tests."""
    root = logging.getLogger()
    saved, saved_level = root.handlers[:], root.level
    yield
    root.handlers[:] = saved
    root.setLevel(saved_level)
    logging.captureWarnings(False)


# The real shape from the flood: a litellm RateLimitError's str().
RATE_LIMIT_STR = (
    "Error code: 429 - {'error': {'message': 'rate limit exceeded', 'type': "
    "'rate_limit_error'}}\n"
    "For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/429"
)


# --------------------------------------------------------------------------- collapse


def test_collapse_flattens_every_kind_of_whitespace():
    flat = collapse(RATE_LIMIT_STR)
    assert "\n" not in flat
    assert flat.count("For more information check") == 1
    assert flat.endswith("Status/429")            # payload intact, just flattened
    assert collapse("a\r\n\tb   c\n") == "a b c"


def test_collapse_truncates_after_flattening_not_before():
    """The #0546 defect was truncating the *formatted message* first, which mangled
    the payload mid-URL and still left the newline inside the surviving prefix."""
    limit = RATE_LIMIT_STR.index("\n") + 5
    out = collapse(RATE_LIMIT_STR, limit)
    assert "\n" not in out
    assert out.endswith("...")
    assert len(out) <= limit + 3


def test_collapse_accepts_an_exception_directly():
    assert collapse(RuntimeError(RATE_LIMIT_STR)).count("\n") == 0


# ------------------------------------------------------------------- formatter


def _format(record_kwargs, **fmt_kwargs):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(SingleLineFormatter(DEFAULT_FORMAT, **fmt_kwargs))
    logger = logging.getLogger("test.singleline")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.log(record_kwargs.pop("level", logging.WARNING), *record_kwargs.pop("args"),
               **record_kwargs)
    return stream.getvalue()


def test_formatter_keeps_a_multiline_message_on_one_line():
    out = _format({"args": ("provider %s failed: %s", "litellm", RATE_LIMIT_STR)})
    assert out.count("\n") == 1                   # only the handler's terminator
    assert "For more information check" in out
    assert out.startswith("20")                   # timestamped


def test_formatter_keeps_a_traceback_on_one_line():
    try:
        raise ValueError("boom\nsecond line")
    except ValueError:
        out = _format({"args": ("embed failed",), "exc_info": True})
    assert out.count("\n") == 1
    assert "Traceback" in out and "ValueError" in out


# ---------------------------------------------------------------- repeat collapse


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _emit(logger, key, n=1, msg="429 from litellm"):
    for _ in range(n):
        logger.warning(msg, extra={"dedup_key": key})


def _dedup_logger(window=60.0, clock=None):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.addFilter(RepeatCollapseFilter(window, clock=clock or time.monotonic))
    logger = logging.getLogger(f"test.dedup.{id(stream)}")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    return logger, stream


def test_repeat_collapse_emits_once_per_window_and_keeps_the_count():
    clock = _Clock()
    logger, stream = _dedup_logger(60.0, clock)

    _emit(logger, ("litellm", "ocr", "RateLimitError"), n=144)
    assert stream.getvalue().count("\n") == 1     # 143 suppressed

    clock.now += 61
    _emit(logger, ("litellm", "ocr", "RateLimitError"))
    lines = stream.getvalue().strip().split("\n")
    assert len(lines) == 2
    assert "143 suppressed" in lines[1]
    assert "61s" in lines[1]


def test_repeat_collapse_does_not_conflate_distinct_keys():
    logger, stream = _dedup_logger()
    _emit(logger, ("litellm", "ocr", "RateLimitError"), n=3)
    _emit(logger, ("openrouter", "embed", "RateLimitError"), n=3)
    assert stream.getvalue().count("\n") == 2


def test_records_without_a_dedup_key_are_never_suppressed():
    logger, stream = _dedup_logger()
    for _ in range(5):
        logger.warning("per-doc chatter")
    assert stream.getvalue().count("\n") == 5


# ------------------------------------------------------------- configure_logging


def test_configure_logging_routes_library_warnings_through_logging():
    """pydub's ffmpeg RuntimeWarning used to land on raw stderr, untimestamped."""
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    try:
        warnings.warn("Couldn't find ffmpeg or avconv", RuntimeWarning)
        out = stream.getvalue()
    finally:
        logging.captureWarnings(False)
    assert "RuntimeWarning" in out
    assert "ffmpeg" in out
    assert out.count("\n") == 1                   # the two-line warning, collapsed
    assert out.startswith("20")


def test_configure_logging_is_the_single_place_entrypoints_call():
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    logging.getLogger("some.module").info("hello\nworld")
    assert stream.getvalue().count("\n") == 1
    assert logging.getLogger().level == logging.INFO


@pytest.mark.parametrize("level,expected", [
    ("debug", logging.DEBUG), ("WARNING", logging.WARNING),
    ("nonsense", logging.WARNING), (logging.ERROR, logging.ERROR),
])
def test_configure_logging_level_parsing(level, expected):
    configure_logging(level, stream=io.StringIO())
    assert logging.getLogger().level == expected
