"""Cross-process exclusion for full-sweep and targeted index writers."""

import multiprocessing
import threading
from unittest.mock import patch

import pytest


def _hold_index_lock(index_root, acquired, release):
    from core.index_write_lock import index_write_lock

    with index_write_lock(index_root, "chunks"):
        acquired.set()
        if not release.wait(10):
            raise TimeoutError("parent did not release index write lock")


def _try_index_lock(index_root, busy):
    from core.index_write_lock import IndexWriteLockBusy, index_write_lock

    try:
        with index_write_lock(index_root, "chunks", blocking=False):
            return
    except IndexWriteLockBusy:
        busy.set()


def _take_nested_lock(index_root):
    from core.index_write_lock import index_write_lock

    with index_write_lock(index_root, "chunks"):
        with index_write_lock(index_root, "chunks"):
            return


def test_index_write_lock_excludes_another_process(tmp_path):
    context = multiprocessing.get_context("spawn")
    acquired = context.Event()
    release = context.Event()
    busy = context.Event()
    holder = context.Process(
        target=_hold_index_lock,
        args=(str(tmp_path), acquired, release),
    )
    contender = context.Process(
        target=_try_index_lock,
        args=(str(tmp_path), busy),
    )
    holder.start()
    try:
        assert acquired.wait(10)
        contender.start()
        contender.join(10)
        assert contender.exitcode == 0
        assert busy.is_set()
    finally:
        release.set()
        holder.join(10)
        for process in (holder, contender):
            if process.pid is not None and process.is_alive():
                process.terminate()
                process.join(5)

    assert holder.exitcode == 0


def test_index_write_lock_nested_acquisition_reuses_outer_flock(tmp_path):
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=_take_nested_lock, args=(str(tmp_path),))
    process.start()
    process.join(3)
    try:
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)


def test_index_write_lock_excludes_another_thread(tmp_path):
    from core.index_write_lock import IndexWriteLockBusy, index_write_lock

    acquired = threading.Event()
    release = threading.Event()

    def hold():
        with index_write_lock(tmp_path, "chunks"):
            acquired.set()
            release.wait(5)

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        assert acquired.wait(3)
        with pytest.raises(IndexWriteLockBusy):
            with index_write_lock(tmp_path, "chunks", blocking=False):
                pass
    finally:
        release.set()
        holder.join(5)


def test_index_write_lock_keeps_distinct_tables_independent(tmp_path):
    from core.index_write_lock import index_write_lock

    with index_write_lock(tmp_path, "chunks"):
        with index_write_lock(tmp_path, "other_chunks", blocking=False):
            pass


def test_index_write_lock_releases_process_lock_when_unlock_fails(tmp_path):
    from core.index_write_lock import index_write_lock

    real_flock = __import__("fcntl").flock
    calls = 0

    def fail_unlock(fd, flags):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("unlock failed")
        return real_flock(fd, flags)

    with pytest.raises(OSError, match="unlock failed"):
        with patch("fcntl.flock", side_effect=fail_unlock):
            with index_write_lock(tmp_path, "chunks"):
                pass

    with index_write_lock(tmp_path, "chunks", blocking=False):
        pass
