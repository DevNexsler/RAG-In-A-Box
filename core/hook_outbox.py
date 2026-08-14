"""Durable, SQLite-backed outbox for hook deliveries."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


RETRY_DELAYS_SECONDS = (1, 5, 30, 120)
_DATABASE_FILENAME = "hook-outbox.sqlite3"
_TERMINAL_STATUSES = {"completed", "redrive_required"}
_PERSISTED_HOOK_FIELDS = {
    "name",
    "type",
    "url",
    "events",
    "timeout_seconds",
    "secret_env",
    "accepted_statuses",
}


@dataclass(frozen=True)
class HookDelivery:
    id: int
    event_id: str
    hook_name: str
    event: dict[str, Any]
    hook: dict[str, Any]
    status: str
    attempts: int
    next_attempt_at: float
    last_outcome: str | None
    last_error: str | None
    created_at: float
    updated_at: float
    revision: int


def _sanitize_hook(value: dict[str, Any]) -> dict[str, Any]:
    """Persist only hook delivery settings that cannot carry credentials."""
    sanitized: dict[str, Any] = {}
    for key, item in value.items():
        if key not in _PERSISTED_HOOK_FIELDS:
            continue
        if key in {"name", "type"} and isinstance(item, str):
            sanitized[key] = item
        elif key == "secret_env" and isinstance(item, str) and item.isidentifier():
            sanitized[key] = item
        elif key == "url" and isinstance(item, str):
            if item.startswith("${") and item.endswith("}") and item[2:-1].isidentifier():
                sanitized[key] = item
            else:
                parsed = urlsplit(item)
                if (
                    parsed.scheme in {"http", "https"}
                    and parsed.netloc
                    and not parsed.username
                    and not parsed.password
                    and not parsed.query
                    and not parsed.fragment
                ):
                    sanitized[key] = item
        elif key in {"events", "accepted_statuses"} and isinstance(item, list):
            sanitized[key] = [entry for entry in item if isinstance(entry, str)]
        elif key == "timeout_seconds" and isinstance(item, (int, float)) and not isinstance(item, bool):
            sanitized[key] = item
    return sanitized


class HookOutbox:
    """Revision-safe hook delivery state persisted below one index root."""

    def __init__(self, index_root: str | Path) -> None:
        root = Path(index_root)
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / _DATABASE_FILENAME
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS hook_deliveries (
                    id INTEGER PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    hook_name TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    hook_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL,
                    next_attempt_at REAL NOT NULL,
                    last_outcome TEXT,
                    last_error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    revision INTEGER NOT NULL,
                    UNIQUE (event_id, hook_name)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_hook_deliveries_due
                ON hook_deliveries (status, next_attempt_at, id)
                """
            )

    @staticmethod
    def _delivery(row: sqlite3.Row) -> HookDelivery:
        return HookDelivery(
            id=int(row["id"]),
            event_id=str(row["event_id"]),
            hook_name=str(row["hook_name"]),
            event=json.loads(str(row["event_json"])),
            hook=json.loads(str(row["hook_json"])),
            status=str(row["status"]),
            attempts=int(row["attempts"]),
            next_attempt_at=float(row["next_attempt_at"]),
            last_outcome=None if row["last_outcome"] is None else str(row["last_outcome"]),
            last_error=None if row["last_error"] is None else str(row["last_error"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            revision=int(row["revision"]),
        )

    @staticmethod
    def _safe_error(error: object) -> str:
        """Persist no untrusted error text: it can contain a credential."""
        _ = error
        return "delivery_error"

    def enqueue(self, event: dict[str, Any], hook: dict[str, Any]) -> HookDelivery:
        event_id = str(event.get("event_id") or "").strip()
        hook_name = str(hook.get("name") or "").strip()
        if not event_id or not hook_name:
            raise ValueError("event_id and hook name must not be empty")
        event_json = json.dumps(event, separators=(",", ":"))
        hook_json = json.dumps(_sanitize_hook(hook), separators=(",", ":"))
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT OR IGNORE INTO hook_deliveries (
                    event_id, hook_name, event_json, hook_json, status, attempts,
                    next_attempt_at, last_outcome, last_error, created_at, updated_at, revision
                ) VALUES (?, ?, ?, ?, 'pending', 0, ?, NULL, NULL, ?, ?, 1)
                """,
                (event_id, hook_name, event_json, hook_json, now, now, now),
            )
            row = connection.execute(
                "SELECT * FROM hook_deliveries WHERE event_id = ? AND hook_name = ?",
                (event_id, hook_name),
            ).fetchone()
            connection.commit()
        assert row is not None
        return self._delivery(row)

    def due(self, limit: int, now: float | None = None) -> list[HookDelivery]:
        if limit <= 0:
            return []
        due_at = time.time() if now is None else float(now)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM hook_deliveries
                WHERE status = 'pending' AND next_attempt_at <= ?
                ORDER BY next_attempt_at, id
                LIMIT ?
                """,
                (due_at, int(limit)),
            ).fetchall()
        return [self._delivery(row) for row in rows]

    def complete(self, delivery: HookDelivery) -> HookDelivery | None:
        return self._transition(delivery, "completed", delivery.last_outcome, delivery.last_error)

    def retry(
        self,
        delivery: HookDelivery,
        outcome: str,
        error: object,
        now: float | None = None,
    ) -> HookDelivery | None:
        attempt = delivery.attempts + 1
        if attempt >= 5:
            return self.redrive_required(delivery, outcome, error, attempts=attempt)
        base = time.time() if now is None else float(now)
        return self._transition(
            delivery,
            "pending",
            outcome,
            self._safe_error(error),
            attempts=attempt,
            next_attempt_at=base + RETRY_DELAYS_SECONDS[attempt - 1],
        )

    def redrive_required(
        self,
        delivery: HookDelivery,
        outcome: str,
        error: object,
        *,
        attempts: int | None = None,
    ) -> HookDelivery | None:
        return self._transition(
            delivery,
            "redrive_required",
            outcome,
            self._safe_error(error),
            attempts=delivery.attempts if attempts is None else attempts,
        )

    def _transition(
        self,
        delivery: HookDelivery,
        status: str,
        outcome: str | None,
        error: str | None,
        *,
        attempts: int | None = None,
        next_attempt_at: float | None = None,
    ) -> HookDelivery | None:
        now = time.time()
        next_due = delivery.next_attempt_at if next_attempt_at is None else next_attempt_at
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE hook_deliveries
                SET status = ?, attempts = ?, next_attempt_at = ?, last_outcome = ?,
                    last_error = ?, updated_at = ?, revision = revision + 1
                WHERE id = ? AND revision = ? AND status = 'pending'
                """,
                (
                    status,
                    delivery.attempts if attempts is None else attempts,
                    next_due,
                    outcome,
                    error,
                    now,
                    delivery.id,
                    delivery.revision,
                ),
            )
            if cursor.rowcount != 1:
                connection.commit()
                return None
            row = connection.execute("SELECT * FROM hook_deliveries WHERE id = ?", (delivery.id,)).fetchone()
            connection.commit()
        assert row is not None
        return self._delivery(row)
