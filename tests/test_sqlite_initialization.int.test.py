"""Failed durable-queue initialization must release real SQLite resources."""
import gc
import os
import sqlite3

import pytest

from core.hook_outbox import HookOutbox
from core.index_request_queue import IndexRequestQueue
from core.sqlite_init import initializing_database


def test_database_authorizer_rejection_closes_initialization_connection(tmp_path):
    connection = sqlite3.connect(tmp_path / 'queue.sqlite3')
    connection.set_authorizer(lambda action, *_: sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_PRAGMA else sqlite3.SQLITE_OK)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            with initializing_database(lambda: connection):
                pytest.fail('rejected initialization cannot yield a connection')
        with pytest.raises(sqlite3.ProgrammingError, match='closed'):
            connection.execute('SELECT 1')
    finally:
        connection.close()


@pytest.mark.parametrize('queue_type,filename', [(HookOutbox, 'hook-outbox.sqlite3'), (IndexRequestQueue, 'index-requests.sqlite3')])
def test_corrupt_database_startup_does_not_leak_file_handles(tmp_path, queue_type, filename):
    path = tmp_path / filename
    path.write_bytes(b'not a SQLite database' * 100)
    def descriptors():
        matches = []
        for name in os.listdir('/proc/self/fd'):
            try:
                if os.readlink('/proc/self/fd/' + name) == str(path):
                    matches.append(name)
            except FileNotFoundError:
                pass
        return matches
    gc.collect()
    gc.disable()
    try:
        for _ in range(5):
            with pytest.raises(sqlite3.DatabaseError):
                queue_type(tmp_path)
        assert descriptors() == [], 'failed connection setup retained database handles'
    finally:
        gc.enable()
        gc.collect()
