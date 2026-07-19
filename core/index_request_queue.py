"""Crash-safe queue for targeted indexing requests."""

from __future__ import annotations

import posixpath
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_QUEUE_FILENAME = "index-requests.sqlite3"
_BUSY_TIMEOUT_MS = 5_000
_INITIALIZE_RETRY_DELAYS = (0.01, 0.05, 0.1, 0.25, 0.5)
_INITIALIZE_LOCK = threading.Lock()


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
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self) -> None:
        with _INITIALIZE_LOCK:
            for attempt in range(len(_INITIALIZE_RETRY_DELAYS) + 1):
                try:
                    with self._connect() as connection:
                        connection.execute("PRAGMA journal_mode=WAL")
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
                                UNIQUE (table_name, source_name, target)
                            )
                            """
                        )
                        connection.execute(
                            """
                            CREATE INDEX IF NOT EXISTS idx_index_requests_pending
                            ON index_requests (table_name, status, created_at, id)
                            """
                        )
                    return
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc).lower() or attempt >= len(
                        _INITIALIZE_RETRY_DELAYS
                    ):
                        raise
                    time.sleep(_INITIALIZE_RETRY_DELAYS[attempt])

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
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                INSERT INTO index_requests (
                    table_name, source_name, target, force, status,
                    attempts, revision, created_at, updated_at, last_error
                ) VALUES (?, ?, ?, ?, 'pending', 0, 1, ?, ?, NULL)
                ON CONFLICT (table_name, source_name, target) DO UPDATE SET
                    force = MAX(index_requests.force, excluded.force),
                    status = 'pending',
                    revision = index_requests.revision + 1,
                    updated_at = excluded.updated_at,
                    last_error = NULL
                RETURNING *
                """,
                (table_name, source_name, target, int(force), now, now),
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
        with self._connect() as connection:
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
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM index_requests WHERE id = ? AND revision = ?",
                (request.id, request.revision),
            )
        return cursor.rowcount == 1

    def fail(self, request: IndexRequest, error: str) -> bool:
        now = _utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE index_requests
                SET attempts = attempts + 1,
                    updated_at = ?,
                    last_error = ?,
                    status = 'pending'
                WHERE id = ? AND revision = ?
                """,
                (now, str(error), request.id, request.revision),
            )
        return cursor.rowcount == 1
