"""Serialize local schema initialization and retry SQLite's WAL lock transition."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import sqlite3
import threading
import time

_INITIALIZE_LOCK = threading.Lock()
_RETRY_DELAYS = (.01, .05, .1, .25, .5)


@contextmanager
def initializing_database(connect: Callable[[], sqlite3.Connection]) -> Iterator[sqlite3.Connection]:
    """Own one immediate schema transaction; always commit/rollback and close.

    SQLite's journal-mode transition can return SQLITE_BUSY immediately despite
    busy_timeout. The process lock prevents local races; bounded retries handle
    peer processes entering WAL at the same time.
    """
    with _INITIALIZE_LOCK:
        for attempt in range(len(_RETRY_DELAYS) + 1):
            connection = None
            ready = False
            try:
                connection = connect()
                connection.execute('PRAGMA journal_mode=WAL')
                connection.execute('BEGIN IMMEDIATE')
                ready = True
                break
            except sqlite3.OperationalError as error:
                if 'locked' not in str(error).lower() or attempt == len(_RETRY_DELAYS):
                    raise
                # Close before sleeping, releasing any lock this attempt owns.
                if connection is not None:
                    connection.close()
                    connection = None
                time.sleep(_RETRY_DELAYS[attempt])
            finally:
                if connection is not None and not ready:
                    connection.close()
        try:
            with connection:
                yield connection
        finally:
            connection.close()
