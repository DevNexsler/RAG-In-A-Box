"""Cross-process exclusion for full-sweep and targeted index writers."""

from __future__ import annotations

import fcntl
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import Iterator


class IndexWriteLockBusy(RuntimeError):
    """Raised when a non-blocking index writer cannot acquire the table."""


@dataclass
class _ProcessLockState:
    lock: threading.RLock
    depth: int = 0
    lock_file: TextIOWrapper | None = None


_PROCESS_LOCKS: dict[str, _ProcessLockState] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()


@contextmanager
def index_write_lock(
    index_root: str | Path,
    table_name: str,
    *,
    blocking: bool = True,
) -> Iterator[None]:
    """Exclude another full or targeted writer for one Lance table."""
    lock_root = Path(index_root) / ".lancedb-write-locks"
    lock_path = lock_root / f"{table_name}-session.lock"
    key = str(lock_path.resolve())
    with _PROCESS_LOCKS_GUARD:
        state = _PROCESS_LOCKS.setdefault(
            key,
            _ProcessLockState(lock=threading.RLock()),
        )

    if not state.lock.acquire(blocking=blocking):
        raise IndexWriteLockBusy(f"Index table {table_name!r} already has a writer")

    try:
        if state.depth == 0:
            lock_root.mkdir(parents=True, exist_ok=True)
            lock_file = lock_path.open("a+")
            flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            try:
                fcntl.flock(lock_file.fileno(), flags)
            except BlockingIOError as exc:
                lock_file.close()
                raise IndexWriteLockBusy(
                    f"Index table {table_name!r} already has a writer"
                ) from exc
            except BaseException:
                lock_file.close()
                raise
            state.lock_file = lock_file
        state.depth += 1
        try:
            yield
        finally:
            state.depth -= 1
            if state.depth == 0:
                lock_file = state.lock_file
                state.lock_file = None
                if lock_file is not None:
                    cleanup_error: BaseException | None = None
                    try:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                    except BaseException as exc:
                        cleanup_error = exc
                    try:
                        lock_file.close()
                    except BaseException as exc:
                        if cleanup_error is None:
                            cleanup_error = exc
                    if cleanup_error is not None:
                        raise cleanup_error
    finally:
        state.lock.release()
