"""Persistent document ID registry backed by SQLite.

Each document gets a unique 5-character base-62 ID embedded in its filename
as an @XXXXX@ suffix (e.g., recipe@00001@.md). IDs survive file moves across
different memory systems.

Counter-based generation ensures sequential, collision-free IDs.

Includes an append-only audit log table for tracking every lifecycle event
(registered, moved, deleted, collision, rename_failed, migrated) so that
changes are traceable and recoverable.
"""

import re
import sqlite3
import time
from pathlib import Path
from typing import Literal

# Base-62 charset: digits, lowercase, uppercase
_BASE62 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_BASE = len(_BASE62)  # 62
_ID_LEN = 5           # 62^5 = 916,132,832 unique IDs

_ID_PATTERN = re.compile(r"@([0-9a-zA-Z]{5})@")


def _int_to_base62(n: int) -> str:
    """Encode non-negative integer as zero-padded 5-char base-62 string."""
    if n < 0:
        raise ValueError(f"Cannot encode negative integer: {n}")
    chars = []
    for _ in range(_ID_LEN):
        chars.append(_BASE62[n % _BASE])
        n //= _BASE
    return "".join(reversed(chars))


def _base62_to_int(s: str) -> int:
    """Decode a base-62 string back to an integer."""
    n = 0
    for ch in s:
        idx = _BASE62.index(ch)
        n = n * _BASE + idx
    return n


def extract_id_from_filename(filename: str) -> str | None:
    """Extract the @XXXXX@ doc ID from a filename, or None if absent.

    >>> extract_id_from_filename("recipe@00001@.md")
    '00001'
    >>> extract_id_from_filename("recipe.md") is None
    True
    """
    m = _ID_PATTERN.search(filename)
    return m.group(1) if m else None


def strip_id_from_filename(filename: str) -> str:
    """Remove the @XXXXX@ doc ID from a filename, returning the clean name.

    >>> strip_id_from_filename("recipe@00001@.md")
    'recipe.md'
    >>> strip_id_from_filename("recipe.md")
    'recipe.md'
    """
    return _ID_PATTERN.sub("", filename)


def inject_id_into_filename(filename: str, doc_id: str) -> str:
    """Insert @XXXXX@ before the file extension (or at end if no extension).

    >>> inject_id_into_filename("recipe.md", "00001")
    'recipe@00001@.md'
    >>> inject_id_into_filename("README", "0000a")
    'README@0000a@'
    """
    p = Path(filename)
    # Handle compound extensions like .tar.gz — only split on the last suffix
    stem = p.stem
    ext = p.suffix  # e.g. ".md", ".pdf", ""
    return f"{stem}@{doc_id}@{ext}"


class DocIDStore:
    """SQLite-backed persistent document ID registry.

    Stores a monotonic counter and a mapping from 5-char base-62 IDs to
    vault-relative file paths.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    # Event types for the audit log
    REGISTERED = "registered"        # new file got an ID assigned
    MOVED = "moved"                  # file moved, rel_path updated
    DELETED = "deleted"              # file removed from vault
    COLLISION = "collision"          # duplicate ID detected, re-assigned
    RENAME_FAILED = "rename_failed"  # OS refused the rename
    MIGRATED = "migrated"            # legacy path-based ID migrated

    def _init_schema(self) -> None:
        c = self._conn
        c.execute("""
            CREATE TABLE IF NOT EXISTS doc_registry (
                doc_id      TEXT PRIMARY KEY,
                rel_path    TEXT NOT NULL,
                created     REAL NOT NULL,
                source_name TEXT NOT NULL DEFAULT 'documents'
            )
        """)
        # Migration: add source_name column to pre-existing tables (idempotent).
        # SQLite ALTER TABLE ADD COLUMN raises OperationalError if the column
        # already exists, so we detect and swallow.
        try:
            c.execute(
                "ALTER TABLE doc_registry ADD COLUMN source_name "
                "TEXT NOT NULL DEFAULT 'documents'"
            )
        except sqlite3.OperationalError:
            pass  # Column already exists (fresh DB from CREATE above, or already migrated)
        c.execute("""
            CREATE TABLE IF NOT EXISTS counter (
                id    INTEGER PRIMARY KEY CHECK (id = 1),
                value INTEGER NOT NULL DEFAULT 0
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                rowid    INTEGER PRIMARY KEY AUTOINCREMENT,
                ts       REAL    NOT NULL,
                event    TEXT    NOT NULL,
                doc_id   TEXT    NOT NULL,
                rel_path TEXT    NOT NULL DEFAULT '',
                old_path TEXT    NOT NULL DEFAULT '',
                detail   TEXT    NOT NULL DEFAULT ''
            )
        """)
        # Retired IDs: deleted IDs that must never be reused.
        # When a file is deleted its ID goes here so that copy-pasted files
        # carrying a stale @XXXXX@ suffix get assigned a fresh ID instead
        # of silently inheriting a dead document's identity.
        c.execute("""
            CREATE TABLE IF NOT EXISTS retired_ids (
                doc_id       TEXT PRIMARY KEY,
                retired_at   REAL NOT NULL,
                last_path    TEXT NOT NULL DEFAULT ''
            )
        """)
        # Seed counter if not present
        c.execute("INSERT OR IGNORE INTO counter (id, value) VALUES (1, 0)")
        c.commit()

    def next_id(self) -> str:
        """Atomically increment counter and return the next base-62 5-char ID."""
        c = self._conn
        c.execute("UPDATE counter SET value = value + 1 WHERE id = 1")
        row = c.execute("SELECT value FROM counter WHERE id = 1").fetchone()
        c.commit()
        return _int_to_base62(row[0])

    def register(
        self,
        doc_id: str,
        rel_path: str,
        *,
        event: str = "",
        detail: str = "",
        source_name: str | None = None,
    ) -> None:
        """Insert or update a doc_id → rel_path mapping.

        If *event* is provided it is written to the audit log. Otherwise a
        ``registered`` event is logged automatically for new entries, and a
        ``moved`` event for path changes.

        Args:
            source_name: Which Source this doc_id belongs to. Defaults to None,
                meaning "do not touch source_name on conflict — preserve any
                existing value, or insert 'documents' for new rows." Pass an
                explicit string when the caller owns the namespace for this row
                (e.g. FilesystemSource always passes its own name).
        """
        old_path = ""
        if not event:
            existing = self._conn.execute(
                "SELECT rel_path FROM doc_registry WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if existing is None:
                event = self.REGISTERED
            elif existing[0] != rel_path:
                event = self.MOVED
                old_path = existing[0]
            # else: unchanged — no audit entry needed

        c = self._conn
        if source_name is None:
            # Caller didn't specify — insert 'documents' for new rows,
            # preserve existing source_name on conflict.
            c.execute(
                "INSERT INTO doc_registry (doc_id, rel_path, created, source_name) "
                "VALUES (?, ?, ?, 'documents') "
                "ON CONFLICT(doc_id) DO UPDATE SET rel_path=excluded.rel_path",
                (doc_id, rel_path, time.time()),
            )
        else:
            # Caller specified — insert or overwrite source_name.
            c.execute(
                "INSERT INTO doc_registry (doc_id, rel_path, created, source_name) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(doc_id) DO UPDATE SET "
                "    rel_path=excluded.rel_path, "
                "    source_name=excluded.source_name",
                (doc_id, rel_path, time.time(), source_name),
            )
        if event:
            self._log(event, doc_id, rel_path, old_path=old_path, detail=detail)
        self._conn.commit()

    def lookup_path(self, doc_id: str) -> str | None:
        """Return the rel_path for a doc_id, or None if not found."""
        row = self._conn.execute(
            "SELECT rel_path FROM doc_registry WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return row[0] if row else None

    def lookup_id(self, rel_path: str) -> str | None:
        """Return the doc_id for a rel_path, or None if not found."""
        row = self._conn.execute(
            "SELECT doc_id FROM doc_registry WHERE rel_path = ?", (rel_path,)
        ).fetchone()
        return row[0] if row else None

    def update_path(self, doc_id: str, new_rel_path: str) -> None:
        """Update the rel_path for an existing doc_id."""
        self._conn.execute(
            "UPDATE doc_registry SET rel_path = ? WHERE doc_id = ?",
            (new_rel_path, doc_id),
        )
        self._conn.commit()

    def delete(self, doc_id: str) -> None:
        """Remove a doc_id entry from the registry, retire it, and log a ``deleted`` event.

        The ID is moved to the ``retired_ids`` table so it can never be
        reused by a different file — even if someone copies a file that
        still carries the old @XXXXX@ suffix.
        """
        row = self._conn.execute(
            "SELECT rel_path FROM doc_registry WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        self._conn.execute("DELETE FROM doc_registry WHERE doc_id = ?", (doc_id,))
        if row:
            self._conn.execute(
                "INSERT OR REPLACE INTO retired_ids (doc_id, retired_at, last_path) "
                "VALUES (?, ?, ?)",
                (doc_id, time.time(), row[0]),
            )
            self._log(self.DELETED, doc_id, row[0])
        self._conn.commit()

    def is_retired(self, doc_id: str) -> bool:
        """Return True if this ID was previously used and then deleted.

        Retired IDs must not be reused — a copy-pasted file carrying a
        retired @XXXXX@ suffix should get a fresh ID instead.
        """
        row = self._conn.execute(
            "SELECT 1 FROM retired_ids WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return row is not None

    def retired_info(self, doc_id: str) -> dict | None:
        """Return retirement details for a doc_id, or None if not retired."""
        row = self._conn.execute(
            "SELECT retired_at, last_path FROM retired_ids WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if row:
            return {"doc_id": doc_id, "retired_at": row[0], "last_path": row[1]}
        return None

    def all_mappings(self) -> dict[str, str]:
        """Return {doc_id: rel_path} for all registered documents."""
        rows = self._conn.execute("SELECT doc_id, rel_path FROM doc_registry").fetchall()
        return {r[0]: r[1] for r in rows}

    def distinct_source_names(self) -> set[str]:
        """Return the set of source_name values currently in the registry.

        Used by mcp_server to validate the `source_name` filter parameter
        on MCP tool calls without hardcoding the valid set.
        """
        cur = self._conn.execute("SELECT DISTINCT source_name FROM doc_registry")
        return {row[0] for row in cur.fetchall()}

    def count(self) -> int:
        """Return total number of registered documents."""
        row = self._conn.execute("SELECT COUNT(*) FROM doc_registry").fetchone()
        return row[0]

    # --- Audit log ---

    def _log(
        self,
        event: str,
        doc_id: str,
        rel_path: str = "",
        *,
        old_path: str = "",
        detail: str = "",
    ) -> None:
        """Append a row to the audit log (no commit — caller commits)."""
        self._conn.execute(
            "INSERT INTO audit_log (ts, event, doc_id, rel_path, old_path, detail) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), event, doc_id, rel_path, old_path, detail),
        )

    def log_event(
        self,
        event: str,
        doc_id: str,
        rel_path: str = "",
        *,
        old_path: str = "",
        detail: str = "",
    ) -> None:
        """Write an audit log entry (public API for callers outside DocIDStore)."""
        self._log(event, doc_id, rel_path, old_path=old_path, detail=detail)
        self._conn.commit()

    def audit_log(
        self,
        *,
        doc_id: str | None = None,
        event: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict]:
        """Query the audit log with optional filters.

        Returns rows newest-first as dicts with keys:
        ts, event, doc_id, rel_path, old_path, detail.
        """
        sql = "SELECT ts, event, doc_id, rel_path, old_path, detail FROM audit_log"
        params: list = []
        clauses: list[str] = []
        if doc_id:
            clauses.append("doc_id = ?")
            params.append(doc_id)
        if event:
            clauses.append("event = ?")
            params.append(event)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY rowid DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self._conn.execute(sql, params).fetchall()
        return [
            {
                "ts": r[0],
                "event": r[1],
                "doc_id": r[2],
                "rel_path": r[3],
                "old_path": r[4],
                "detail": r[5],
            }
            for r in rows
        ]

    def audit_log_count(self, *, doc_id: str | None = None, event: str | None = None) -> int:
        """Return total number of audit log entries matching the filters."""
        sql = "SELECT COUNT(*) FROM audit_log"
        params: list = []
        clauses: list[str] = []
        if doc_id:
            clauses.append("doc_id = ?")
            params.append(doc_id)
        if event:
            clauses.append("event = ?")
            params.append(event)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return self._conn.execute(sql, params).fetchone()[0]

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()
