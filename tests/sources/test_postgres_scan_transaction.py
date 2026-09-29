"""PostgresSource.scan must not leave its connection inside a transaction.

psycopg connections are not autocommit, so the scan's first statement opens
an implicit transaction that holds ACCESS SHARE on every table it read. The
source object (and its connection) outlives the scan for the rest of the
index run, so an unterminated transaction sat `idle in transaction` on the
Comm-Data-Store server for minutes and blocked ACCESS EXCLUSIVE DDL on
`messages` (#3653). The fake below models only that psycopg rule: executing
while idle begins a transaction; a `transaction()` block, `commit()` or
`rollback()` ends it.
"""

from contextlib import contextmanager
from datetime import UTC, datetime

from sources.postgres import PostgresSource, TableSpec


class _Cursor:
    def __init__(self, connection, rows):
        self._connection = connection
        self._rows = rows
        self.itersize = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def execute(self, _query):
        self._connection.in_transaction = True

    def __iter__(self):
        return iter(self._rows)


class _Connection:
    def __init__(self, rows):
        self._rows = rows
        self.in_transaction = False

    def cursor(self, name=None):
        return _Cursor(self, self._rows if name else [])

    @contextmanager
    def transaction(self):
        self.in_transaction = True
        try:
            yield
        finally:
            self.in_transaction = False

    def commit(self):
        self.in_transaction = False

    def rollback(self):
        self.in_transaction = False


def _row(message_id):
    return {
        "source": "zoho_cliq",
        "source_message_id": message_id,
        "updated_at": datetime(2026, 9, 28, tzinfo=UTC),
        "_text": f"body {message_id}",
    }


def _source(rows):
    spec = TableSpec(
        source_type="pg_transcript",
        query="SELECT transcript rows",
        id_template="{source}/{source_message_id}",
        text_column="_text",
        mtime_column="updated_at",
        text_normalizer="zoho_cliq_mentions",
    )
    source = PostgresSource("comm_messages", "postgresql://unused", [spec, spec])
    connection = _Connection(rows)
    source._get_conn = lambda: connection
    return source, connection


def test_exhausted_scan_ends_its_transaction():
    source, connection = _source([_row("a"), _row("b")])

    records = list(source.scan())

    assert len(records) == 4
    assert connection.in_transaction is False


def test_abandoned_scan_ends_its_transaction():
    source, connection = _source([_row("a"), _row("b")])
    scan = source.scan()
    next(scan)
    assert connection.in_transaction is True

    scan.close()

    assert connection.in_transaction is False
