"""PostgresSource: read indexable text from PostgreSQL tables.

Each TableSpec declares one SELECT query, the column holding the text,
the column to use as mtime, and a natural-key template. The source opens
one connection lazily and reuses it for the lifetime of the instance.
Each scan() uses a server-side cursor to stream rows so large tables
don't OOM, and yields a SourceRecord per row.
"""

import os
from dataclasses import dataclass, field
from typing import Iterator

import psycopg
from psycopg.rows import dict_row

from sources.base import SourceRecord
from extractors import ExtractionResult


@dataclass
class TableSpec:
    source_type: str            # "pg_message", "pg_transcript", ...
    query: str                  # SELECT with all columns referenced below
    id_template: str            # Python .format() template, e.g. "{source}/{source_message_id}"
    text_column: str
    """Result-set column name for the indexable text.

    If the source column has a different name than you want in metadata,
    alias it in the query. Example:

        query = "SELECT m.body AS _text FROM messages m"
        text_column = "_text"

    The convention "_text" signals that the column is a query-time alias,
    not a real table column, and the value is also cached on
    SourceRecord.metadata for zero-I/O extract.
    """
    mtime_column: str
    """Column holding the timestamptz used for incremental diffing.

    Must be a TIMESTAMP or TIMESTAMPTZ column (not DATE — plain DATE
    objects lack a .timestamp() method and will raise AttributeError).
    Comm-Data-Store's messages/transcripts both have updated_at
    TIMESTAMPTZ columns (added in migration 004).
    """
    metadata_columns: list[str] = field(default_factory=list)
    """List of column names to include verbatim in SourceRecord.metadata.

    Reserved keys (do not include):
        _text  — overwritten internally with the cached text column value
    """


# Satisfies the Source protocol from sources.base structurally — no explicit
# inheritance needed. Runtime isinstance checks against Source() work because
# the protocol is @runtime_checkable.
class PostgresSource:
    def __init__(self, name: str, dsn: str, tables: list[TableSpec]):
        self.name = name
        if dsn.startswith("${") and dsn.endswith("}"):
            env_name = dsn[2:-1]
            dsn = os.environ.get(env_name, "")
            if not dsn:
                raise ValueError(
                    f"DSN env var {env_name!r} not set for postgres source '{name}'"
                )
        self._dsn = dsn
        self._tables = tables
        self._conn: psycopg.Connection | None = None

    def _get_conn(self) -> psycopg.Connection:
        # Reconnect if we don't have a connection, if it's closed, or if a
        # prior scan left it in an INERROR transaction state (e.g. a failed
        # SQL statement). INERROR connections silently reject new commands
        # with "current transaction is aborted" — the only clean recovery
        # is to close and reopen.
        from psycopg.pq import TransactionStatus
        stale = (
            self._conn is None
            or self._conn.closed
            or self._conn.pgconn.transaction_status == TransactionStatus.INERROR
        )
        if stale:
            if self._conn is not None and not self._conn.closed:
                try:
                    self._conn.close()
                except Exception:
                    pass
            self._conn = psycopg.connect(self._dsn, row_factory=dict_row)
        return self._conn

    def scan(self) -> Iterator[SourceRecord]:
        conn = self._get_conn()
        for spec in self._tables:
            # Server-side cursor for streaming; named cursors stream in
            # batches of itersize rather than fetching everything.
            with conn.cursor(name=f"scan_{self.name}_{spec.source_type}") as cur:
                cur.itersize = 500
                cur.execute(spec.query)
                for row in cur:
                    doc_id = spec.id_template.format(**row)
                    text = row.get(spec.text_column) or ""
                    mtime_val = row[spec.mtime_column]
                    mtime = mtime_val.timestamp() if mtime_val else 0.0
                    metadata = {c: row[c] for c in spec.metadata_columns if c in row}
                    metadata["_text"] = text  # cache for zero-IO extract
                    yield SourceRecord(
                        doc_id=doc_id,
                        source_type=spec.source_type,
                        natural_key=doc_id,
                        mtime=mtime,
                        size=len(text.encode("utf-8")),
                        metadata=metadata,
                    )

    def extract(self, record: SourceRecord) -> ExtractionResult:
        # Text was cached by scan() — no second round-trip needed.
        text = record.metadata.get("_text", "")
        frontmatter = {k: v for k, v in record.metadata.items() if k != "_text"}
        return ExtractionResult.from_text(text, frontmatter=frontmatter)

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        self._conn = None
