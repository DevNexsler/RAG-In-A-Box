"""Crash-safe queue for targeted indexing requests."""

from __future__ import annotations

from contextlib import closing
import posixpath
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from core.sqlite_init import initializing_database

_QUEUE_FILENAME = "index-requests.sqlite3"
_BUSY_TIMEOUT_MS = 5_000


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_target(target: str) -> str:
    """Normalize separators and dot segments in a source-relative queue key."""
    value = str(target).strip().replace("\\", "/")
    if not value:
        raise ValueError("target must not be empty")
    normalized = posixpath.normpath(value)
    if normalized in {"", "."}:
        raise ValueError("target must not be empty")
    return normalized


def pending_backlog(index_root: str | Path, table_name: str) -> dict:
    """Pending depth and the oldest pending request's creation time.

    Read-only and side-effect free on purpose: ``/health`` asks this on every
    poll, and constructing an ``IndexRequestQueue`` would create and migrate the
    database as a side effect of a probe. No queue file yet means no backlog.
    """
    path = Path(index_root) / _QUEUE_FILENAME
    empty = {"pending": 0, "oldest_created_at": None}
    if not path.exists():
        return empty
    try:
        connection = sqlite3.connect(
            f"file:{path}?mode=ro",
            uri=True,
            timeout=_BUSY_TIMEOUT_MS / 1_000,
        )
    except sqlite3.Error:
        return empty
    try:
        row = connection.execute(
            """
            SELECT COUNT(*) AS pending, MIN(created_at) AS oldest_created_at
            FROM index_requests
            WHERE table_name = ? AND status = 'pending'
            """,
            (str(table_name).strip(),),
        ).fetchone()
    except sqlite3.Error:
        return empty
    finally:
        connection.close()
    if row is None:
        return empty
    return {"pending": int(row[0] or 0), "oldest_created_at": row[1]}


@dataclass(frozen=True)
class IndexRequest:
    id: int
    table_name: str
    source_name: str
    target: str
    force: bool
    status: str
    attempts: int
    revision: int
    created_at: str
    updated_at: str
    last_error: str | None
    incarnation: str


class IndexRequestQueue:
    """SQLite-backed, revision-safe queue scoped to one index root."""

    def __init__(self, index_root: str | Path) -> None:
        root = Path(index_root)
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / _QUEUE_FILENAME
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=_BUSY_TIMEOUT_MS / 1_000,
        )
        try:
            connection.row_factory = sqlite3.Row
            connection.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            connection.execute("PRAGMA synchronous=FULL")
        except BaseException:
            connection.close()
            raise
        return connection

    def _initialize(self) -> None:
        with initializing_database(self._connect) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS index_requests (
                    id INTEGER PRIMARY KEY,
                    table_name TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    target TEXT NOT NULL,
                    force INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    revision INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_error TEXT,
                    incarnation TEXT NOT NULL DEFAULT '',
                    UNIQUE (table_name, source_name, target)
                )
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(index_requests)")}
            if "incarnation" not in columns:
                connection.execute("ALTER TABLE index_requests ADD COLUMN incarnation TEXT NOT NULL DEFAULT ''")
            connection.execute(
                "UPDATE index_requests SET incarnation = lower(hex(randomblob(16))) WHERE incarnation = ''"
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_index_requests_pending
                ON index_requests (table_name, status, created_at, id)
                """
            )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> IndexRequest:
        return IndexRequest(
            id=int(row["id"]),
            table_name=str(row["table_name"]),
            source_name=str(row["source_name"]),
            target=str(row["target"]),
            force=bool(row["force"]),
            status=str(row["status"]),
            attempts=int(row["attempts"]),
            revision=int(row["revision"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            last_error=(
                None if row["last_error"] is None else str(row["last_error"])
            ),
            incarnation=str(row["incarnation"]),
        )

    def enqueue(
        self,
        table_name: str,
        source_name: str,
        target: str,
        *,
        force: bool = False,
    ) -> IndexRequest:
        table_name = str(table_name).strip()
        source_name = str(source_name).strip()
        if not table_name or not source_name:
            raise ValueError("table_name and source_name must not be empty")
        target = normalize_target(target)
        now = _utc_now()
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                INSERT INTO index_requests (
                    table_name, source_name, target, force, status,
                    attempts, revision, created_at, updated_at, last_error, incarnation
                ) VALUES (?, ?, ?, ?, 'pending', 0, 1, ?, ?, NULL, ?)
                ON CONFLICT (table_name, source_name, target) DO UPDATE SET
                    force = MAX(index_requests.force, excluded.force),
                    status = 'pending',
                    revision = index_requests.revision + 1,
                    updated_at = excluded.updated_at,
                    last_error = NULL
                RETURNING *
                """,
                (table_name, source_name, target, int(force), now, now, uuid.uuid4().hex),
            ).fetchone()
            connection.commit()
        assert row is not None
        return self._from_row(row)

    def pending(
        self,
        table_name: str,
        *,
        limit: int,
        prioritize: tuple[str, str] | None = None,
    ) -> list[IndexRequest]:
        if limit <= 0:
            return []
        priority_source, priority_target = prioritize or ("", "")
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                """
                SELECT * FROM index_requests
                WHERE table_name = ? AND status = 'pending'
                ORDER BY
                    CASE WHEN source_name = ? AND target = ? THEN 0 ELSE 1 END,
                    created_at,
                    id
                LIMIT ?
                """,
                (
                    str(table_name).strip(),
                    priority_source,
                    normalize_target(priority_target) if priority_target else "",
                    int(limit),
                ),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def complete(self, request: IndexRequest) -> bool:
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "DELETE FROM index_requests WHERE id = ? AND revision = ? AND incarnation = ?",
                (request.id, request.revision, request.incarnation),
            )
        return cursor.rowcount == 1

    def fail(self, request: IndexRequest, error: str) -> bool:
        now = _utc_now()
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE index_requests
                SET attempts = attempts + 1,
                    updated_at = ?,
                    last_error = ?,
                    status = 'pending'
                WHERE id = ? AND revision = ? AND incarnation = ?
                """,
                (now, str(error), request.id, request.revision, request.incarnation),
            )
        return cursor.rowcount == 1
