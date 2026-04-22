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
        # check_same_thread=False + an internal lock lets worker threads in
        # the indexer call register/delete concurrently. All writes are still
        # serialized under self._lock so SQLite never sees cross-thread use.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        import threading as _threading
        self._lock = _threading.RLock()
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
                doc_id         TEXT PRIMARY KEY,
                rel_path       TEXT NOT NULL,
                created        REAL NOT NULL,
                source_name    TEXT NOT NULL DEFAULT 'documents',
                size_bytes     INTEGER,
                content_hash   BLOB,
                hash_algo      TEXT,
                dedupe_status  TEXT NOT NULL DEFAULT 'canonical',
                canonical_doc_id TEXT,
                archive_path   TEXT,
                duplicate_reason TEXT,
                duplicate_of_doc_id TEXT,
                first_seen_at  REAL,
                last_seen_at   REAL
            )
        """)
        existing_columns = {
            row[1] for row in c.execute("PRAGMA table_info(doc_registry)").fetchall()
        }
        for column_sql, column_name in (
            ("ALTER TABLE doc_registry ADD COLUMN source_name TEXT NOT NULL DEFAULT 'documents'", "source_name"),
            ("ALTER TABLE doc_registry ADD COLUMN size_bytes INTEGER", "size_bytes"),
            ("ALTER TABLE doc_registry ADD COLUMN content_hash BLOB", "content_hash"),
            ("ALTER TABLE doc_registry ADD COLUMN hash_algo TEXT", "hash_algo"),
            ("ALTER TABLE doc_registry ADD COLUMN dedupe_status TEXT NOT NULL DEFAULT 'canonical'", "dedupe_status"),
            ("ALTER TABLE doc_registry ADD COLUMN canonical_doc_id TEXT", "canonical_doc_id"),
            ("ALTER TABLE doc_registry ADD COLUMN archive_path TEXT", "archive_path"),
            ("ALTER TABLE doc_registry ADD COLUMN duplicate_reason TEXT", "duplicate_reason"),
            ("ALTER TABLE doc_registry ADD COLUMN duplicate_of_doc_id TEXT", "duplicate_of_doc_id"),
            ("ALTER TABLE doc_registry ADD COLUMN first_seen_at REAL", "first_seen_at"),
            ("ALTER TABLE doc_registry ADD COLUMN last_seen_at REAL", "last_seen_at"),
        ):
            if column_name not in existing_columns:
                c.execute(column_sql)
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
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_doc_registry_size_hash "
            "ON doc_registry(size_bytes, content_hash)"
        )
        c.execute(
            "UPDATE doc_registry SET first_seen_at = created "
            "WHERE first_seen_at IS NULL"
        )
        # Seed counter if not present
        c.execute("INSERT OR IGNORE INTO counter (id, value) VALUES (1, 0)")
        c.commit()

    def next_id(self) -> str:
        """Atomically increment counter and return the next base-62 5-char ID."""
        with self._lock:
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
        with self._lock:
            ts = time.time()
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
                    "INSERT INTO doc_registry (doc_id, rel_path, created, source_name, first_seen_at) "
                    "VALUES (?, ?, ?, 'documents', ?) "
                    "ON CONFLICT(doc_id) DO UPDATE SET rel_path=excluded.rel_path",
                    (doc_id, rel_path, ts, ts),
                )
            else:
                # Caller specified — insert or overwrite source_name.
                c.execute(
                    "INSERT INTO doc_registry (doc_id, rel_path, created, source_name, first_seen_at) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(doc_id) DO UPDATE SET "
                    "    rel_path=excluded.rel_path, "
                    "    source_name=excluded.source_name",
                    (doc_id, rel_path, ts, source_name, ts),
                )
            if event:
                self._log(event, doc_id, rel_path, old_path=old_path, detail=detail)
            self._conn.commit()

    def lookup_path(self, doc_id: str) -> str | None:
        """Return the rel_path for a doc_id, or None if not found."""
        with self._lock:
            row = self._conn.execute(
                "SELECT rel_path FROM doc_registry WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            return row[0] if row else None

    def lookup_id(self, rel_path: str) -> str | None:
        """Return the doc_id for a rel_path, or None if not found."""
        with self._lock:
            row = self._conn.execute(
                "SELECT doc_id FROM doc_registry WHERE rel_path = ?", (rel_path,)
            ).fetchone()
            return row[0] if row else None

    def update_path(self, doc_id: str, new_rel_path: str) -> None:
        """Update the rel_path for an existing doc_id."""
        with self._lock:
            self._conn.execute(
                "UPDATE doc_registry SET rel_path = ? WHERE doc_id = ?",
                (new_rel_path, doc_id),
            )
            self._conn.commit()

    def set_source_name(self, doc_id: str, source_name: str) -> None:
        """Explicitly set source_name for an already-registered doc_id.

        Used by FilesystemSource after scan_vault_task registers rows via
        the None-sentinel path (which defaults to 'documents'). When the
        source instance name is something else, this corrects the row.

        No-op if doc_id is not present in doc_registry.
        """
        with self._lock:
            c = self._conn
            c.execute(
                "UPDATE doc_registry SET source_name = ? WHERE doc_id = ?",
                (source_name, doc_id),
            )
            c.commit()

    def delete(self, doc_id: str) -> None:
        """Remove a doc_id entry from the registry, retire it, and log a ``deleted`` event.

        The ID is moved to the ``retired_ids`` table so it can never be
        reused by a different file — even if someone copies a file that
        still carries the old @XXXXX@ suffix.

        Accepts both the exact registered doc_id and the namespaced form
        returned by ``all_mappings()``.  The lookup strategy is:

        1. Try the exact ``doc_id`` passed in (handles the new multi-source
           flow where ``"alpha::00001"`` is stored verbatim in the registry).
        2. If no row is found **and** the caller passed a namespaced ID
           (``"::"`` present), fall back to the bare suffix (``"00001"``).
           This keeps backward compatibility with legacy callers that pass
           ``all_mappings()`` keys — those keys are prefixed by that method
           even though the registry row itself stores only the bare 5-char ID.

        In either case the *actual* stored doc_id (bare or namespaced) is used
        when writing to ``retired_ids`` and the audit log, so that
        ``is_retired()`` queries remain consistent with ``all_mappings()``.
        """
        with self._lock:
            # Step 1: exact match
            row = self._conn.execute(
                "SELECT rel_path FROM doc_registry WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            # Step 2: bare-suffix fallback for legacy bare IDs accessed via
            #         all_mappings() keys (e.g. "documents::abc12" → "abc12")
            stored_key = doc_id
            if row is None and "::" in doc_id:
                bare = doc_id.split("::", 1)[-1]
                row = self._conn.execute(
                    "SELECT rel_path FROM doc_registry WHERE doc_id = ?", (bare,)
                ).fetchone()
                if row is not None:
                    stored_key = bare
            self._conn.execute("DELETE FROM doc_registry WHERE doc_id = ?", (stored_key,))
            if row:
                self._conn.execute(
                    "INSERT OR REPLACE INTO retired_ids (doc_id, retired_at, last_path) "
                    "VALUES (?, ?, ?)",
                    (stored_key, time.time(), row[0]),
                )
                self._log(self.DELETED, stored_key, row[0])
            self._conn.commit()

    def is_retired(self, doc_id: str) -> bool:
        """Return True if this ID was previously used and then deleted.

        Retired IDs must not be reused — a copy-pasted file carrying a
        retired @XXXXX@ suffix should get a fresh ID instead.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM retired_ids WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            return row is not None

    def retired_info(self, doc_id: str) -> dict | None:
        """Return retirement details for a doc_id, or None if not retired."""
        with self._lock:
            row = self._conn.execute(
                "SELECT retired_at, last_path FROM retired_ids WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if row:
                return {"doc_id": doc_id, "retired_at": row[0], "last_path": row[1]}
            return None

    def all_mappings(self) -> dict[str, str]:
        """Return {namespaced_doc_id: rel_path} for all registered documents.

        The key is prefixed with source_name (e.g. ``documents::00001``) so
        it matches what LanceDB stores in its doc_id column after the Task-8
        multi-source namespacing.  If the stored doc_id already contains
        ``::`` (i.e. it was registered as a pre-namespaced ID by the flow),
        it is returned as-is without an additional prefix.

        Callers that need the raw 5-char ID can split on ``"::"`` and take
        the last part.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT doc_id, rel_path, source_name FROM doc_registry"
            ).fetchall()
        result: dict[str, str] = {}
        for doc_id, rel_path, source_name in rows:
            if "::" in doc_id:
                # Already namespaced — store as-is
                result[doc_id] = rel_path
            else:
                # Legacy bare ID — prefix with source_name
                sn = source_name or "documents"
                result[f"{sn}::{doc_id}"] = rel_path
        return result

    def distinct_source_names(self) -> set[str]:
        """Return the set of source_name values currently in the registry.

        Used by mcp_server to validate the `source_name` filter parameter
        on MCP tool calls without hardcoding the valid set.
        """
        with self._lock:
            cur = self._conn.execute("SELECT DISTINCT source_name FROM doc_registry")
            return {row[0] for row in cur.fetchall()}
    @staticmethod
    def _registry_row_to_dict(row: tuple) -> dict:
        return {
            "doc_id": row[0],
            "rel_path": row[1],
            "created": row[2],
            "source_name": row[3],
            "size_bytes": row[4],
            "content_hash": row[5],
            "hash_algo": row[6],
            "dedupe_status": row[7],
            "canonical_doc_id": row[8],
            "archive_path": row[9],
            "duplicate_reason": row[10],
            "duplicate_of_doc_id": row[11],
            "first_seen_at": row[12],
            "last_seen_at": row[13],
        }

    def _reject_stranding_canonical_move(
        self,
        row: tuple,
        *,
        new_size_bytes: int,
        new_content_hash: bytes,
        new_hash_algo: str,
    ) -> None:
        current_size_bytes = row[4]
        current_content_hash = row[5]
        current_hash_algo = row[6]
        if row[7] != "canonical":
            return
        if (
            current_size_bytes is None
            or current_content_hash is None
            or current_hash_algo is None
        ):
            return
        if (
            current_size_bytes == new_size_bytes
            and current_content_hash == new_content_hash
            and current_hash_algo == new_hash_algo
        ):
            return
        has_old_siblings = self._conn.execute(
            """
            SELECT 1
            FROM doc_registry
            WHERE canonical_doc_id = ? AND dedupe_status = 'duplicate'
            LIMIT 1
            """,
            (row[0],),
        ).fetchone()
        if has_old_siblings is not None:
            raise ValueError(
                f"cannot move canonical {row[0]} to a new hash while old cohort still points to it"
            )

    def update_dedupe_identity(
        self,
        doc_id: str,
        *,
        size_bytes: int,
        content_hash: bytes,
        hash_algo: str,
        dedupe_status: str,
        canonical_doc_id: str | None,
        archive_path: str | None = None,
        duplicate_reason: str | None = None,
    ) -> None:
        """Store exact-duplicate identity metadata for an existing row."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    """
                    SELECT doc_id, rel_path, created, source_name, size_bytes, content_hash,
                           hash_algo, dedupe_status, canonical_doc_id, archive_path,
                           duplicate_reason, duplicate_of_doc_id, first_seen_at, last_seen_at
                    FROM doc_registry
                    WHERE doc_id = ?
                    """,
                    (doc_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(doc_id)
                self._reject_stranding_canonical_move(
                    row,
                    new_size_bytes=size_bytes,
                    new_content_hash=content_hash,
                    new_hash_algo=hash_algo,
                )
                now = time.time()
                if dedupe_status == "canonical":
                    rows = self._conn.execute(
                        """
                        SELECT doc_id, rel_path, created, source_name, size_bytes, content_hash,
                               hash_algo, dedupe_status, canonical_doc_id, archive_path,
                               duplicate_reason, duplicate_of_doc_id, first_seen_at, last_seen_at
                        FROM doc_registry
                        WHERE size_bytes = ? AND content_hash = ? AND hash_algo = ?
                        ORDER BY COALESCE(first_seen_at, created), created, doc_id
                        """,
                        (size_bytes, content_hash, hash_algo),
                    ).fetchall()
                    if all(existing[0] != doc_id for existing in rows):
                        rows.append(row)
                        rows.sort(key=lambda entry: (
                            entry[12] if entry[12] is not None else entry[2],
                            entry[2],
                            entry[0],
                        ))
                    if rows:
                        winner = rows[0][0]
                        if winner != doc_id:
                            raise ValueError(
                                f"first-seen canonical is {winner}, not {doc_id}"
                            )
                        other_canonical = next(
                            (
                                existing[0]
                                for existing in rows
                                if existing[0] != doc_id and existing[7] == "canonical"
                            ),
                            None,
                        )
                        if other_canonical is not None:
                            raise ValueError(
                                f"canonical already exists for ({size_bytes}, content_hash)"
                            )
                        for existing in rows:
                            is_winner = existing[0] == doc_id
                            self._conn.execute(
                                """
                                UPDATE doc_registry
                                SET size_bytes = ?,
                                    content_hash = ?,
                                    hash_algo = ?,
                                    dedupe_status = ?,
                                    canonical_doc_id = ?,
                                    archive_path = ?,
                                    duplicate_reason = ?,
                                    duplicate_of_doc_id = ?,
                                    first_seen_at = COALESCE(first_seen_at, ?),
                                    last_seen_at = ?
                                WHERE doc_id = ?
                                """,
                                (
                                    size_bytes,
                                    content_hash,
                                    hash_algo,
                                    "canonical" if is_winner else "duplicate",
                                    None if is_winner else doc_id,
                                    None if is_winner else existing[9],
                                    None if is_winner else existing[10],
                                    None if is_winner else doc_id,
                                    existing[12] if existing[12] is not None else existing[2],
                                    now,
                                    existing[0],
                                ),
                            )
                        self._conn.commit()
                        return
                self._conn.execute(
                    """
                    UPDATE doc_registry
                    SET size_bytes = ?,
                        content_hash = ?,
                        hash_algo = ?,
                        dedupe_status = ?,
                        canonical_doc_id = ?,
                        archive_path = ?,
                        duplicate_reason = ?,
                        duplicate_of_doc_id = ?,
                        first_seen_at = COALESCE(first_seen_at, ?),
                        last_seen_at = ?
                    WHERE doc_id = ?
                    """,
                    (
                        size_bytes,
                        content_hash,
                        hash_algo,
                        dedupe_status,
                        None if dedupe_status == "canonical" else canonical_doc_id,
                        None if dedupe_status == "canonical" else archive_path,
                        None if dedupe_status == "canonical" else duplicate_reason,
                        None if dedupe_status == "canonical" else canonical_doc_id,
                        row[12] if row[12] is not None else row[2],
                        now,
                        doc_id,
                    ),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def find_canonical_by_exact_hash(
        self,
        size_bytes: int,
        content_hash: bytes,
        hash_algo: str,
    ) -> dict | None:
        """Return the canonical registry row for an exact content hash match."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT doc_id, rel_path, created, source_name, size_bytes, content_hash,
                       hash_algo, dedupe_status, canonical_doc_id, archive_path,
                       duplicate_reason, duplicate_of_doc_id, first_seen_at, last_seen_at
                FROM doc_registry
                WHERE size_bytes = ?
                  AND content_hash = ?
                  AND hash_algo = ?
                  AND dedupe_status = 'canonical'
                ORDER BY COALESCE(first_seen_at, created), created
                LIMIT 1
                """,
                (size_bytes, content_hash, hash_algo),
            ).fetchone()
        if row is None:
            return None
        return self._registry_row_to_dict(row)

    def claim_canonical_by_exact_hash(
        self,
        doc_id: str,
        size_bytes: int,
        content_hash: bytes,
        *,
        hash_algo: str,
        archive_path: str | None = None,
        duplicate_reason: str | None = None,
    ) -> dict | None:
        """Atomically resolve the first-seen canonical row for an exact hash.

        All rows matching ``(size_bytes, content_hash, hash_algo)`` participate
        in one first-seen election ordered by ``first_seen_at`` then ``created``.
        The earliest row remains or becomes canonical; every other matching row
        becomes a duplicate of that winner. The winning canonical row is
        returned.
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                candidates = self._conn.execute(
                    """
                    SELECT doc_id, rel_path, created, source_name, size_bytes, content_hash,
                           hash_algo, dedupe_status, canonical_doc_id, archive_path,
                           duplicate_reason, duplicate_of_doc_id, first_seen_at, last_seen_at
                    FROM doc_registry
                    WHERE size_bytes = ? AND content_hash = ? AND hash_algo = ?
                    ORDER BY COALESCE(first_seen_at, created), created, doc_id
                    """,
                    (size_bytes, content_hash, hash_algo),
                ).fetchall()
                candidate_ids = {row[0] for row in candidates}
                caller = self._conn.execute(
                    """
                    SELECT doc_id, rel_path, created, source_name, size_bytes, content_hash,
                           hash_algo, dedupe_status, canonical_doc_id, archive_path,
                           duplicate_reason, duplicate_of_doc_id, first_seen_at, last_seen_at
                    FROM doc_registry
                    WHERE doc_id = ?
                    """,
                    (doc_id,),
                ).fetchone()
                if caller is None:
                    raise KeyError(doc_id)
                self._reject_stranding_canonical_move(
                    caller,
                    new_size_bytes=size_bytes,
                    new_content_hash=content_hash,
                    new_hash_algo=hash_algo,
                )
                if doc_id not in candidate_ids:
                    candidates.append(caller)
                candidates.sort(key=lambda row: (
                    row[12] if row[12] is not None else row[2],
                    row[2],
                    row[0],
                ))
                winner = candidates[0]
                now = time.time()
                for row in candidates:
                    row_doc_id = row[0]
                    is_winner = row_doc_id == winner[0]
                    is_caller = row_doc_id == doc_id
                    archive_value = (
                        None
                        if is_winner
                        else (archive_path if is_caller and archive_path is not None else row[9])
                    )
                    reason_value = (
                        None
                        if is_winner
                        else (duplicate_reason if is_caller and duplicate_reason is not None else row[10])
                    )
                    duplicate_of_value = None if is_winner else winner[0]
                    self._conn.execute(
                        """
                        UPDATE doc_registry
                        SET size_bytes = ?,
                            content_hash = ?,
                            hash_algo = ?,
                            dedupe_status = ?,
                            canonical_doc_id = ?,
                            archive_path = ?,
                            duplicate_reason = ?,
                            duplicate_of_doc_id = ?,
                            first_seen_at = COALESCE(first_seen_at, ?),
                            last_seen_at = ?
                        WHERE doc_id = ?
                        """,
                        (
                            size_bytes,
                            content_hash,
                            hash_algo,
                            "canonical" if is_winner else "duplicate",
                            None if is_winner else winner[0],
                            archive_value,
                            reason_value,
                            duplicate_of_value,
                            row[12] if row[12] is not None else row[2],
                            now,
                            row_doc_id,
                        ),
                    )
                self._conn.commit()
                updated = self._conn.execute(
                    """
                    SELECT doc_id, rel_path, created, source_name, size_bytes, content_hash,
                           hash_algo, dedupe_status, canonical_doc_id, archive_path,
                           duplicate_reason, duplicate_of_doc_id, first_seen_at, last_seen_at
                    FROM doc_registry
                    WHERE doc_id = ?
                    """,
                    (winner[0],),
                ).fetchone()
                return self._registry_row_to_dict(updated)
            except Exception:
                self._conn.rollback()
                raise

    def mark_duplicate(
        self,
        doc_id: str,
        canonical_doc_id: str,
        *,
        archive_path: str | None = None,
        duplicate_reason: str | None = None,
    ) -> None:
        """Mark a row as a duplicate of a canonical doc_id."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                if doc_id == canonical_doc_id:
                    raise ValueError("duplicate cannot self-reference canonical target")
                now = time.time()
                row = self._conn.execute(
                    """
                    SELECT doc_id, rel_path, created, source_name, size_bytes, content_hash,
                           hash_algo, dedupe_status, canonical_doc_id, archive_path,
                           duplicate_reason, duplicate_of_doc_id, first_seen_at, last_seen_at
                    FROM doc_registry
                    WHERE doc_id = ?
                    """,
                    (doc_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(doc_id)
                if row[7] == "canonical":
                    raise ValueError(f"source row {doc_id} is canonical")
                canonical_row = self._conn.execute(
                    """
                    SELECT doc_id, rel_path, created, source_name, size_bytes, content_hash,
                           hash_algo, dedupe_status, canonical_doc_id, archive_path,
                           duplicate_reason, duplicate_of_doc_id, first_seen_at, last_seen_at
                    FROM doc_registry
                    WHERE doc_id = ?
                    """,
                    (canonical_doc_id,),
                ).fetchone()
                if canonical_row is None:
                    raise KeyError(canonical_doc_id)
                if canonical_row[7] != "canonical":
                    raise ValueError(f"canonical target {canonical_doc_id} is not canonical")
                source_has_identity = row[4] is not None and row[5] is not None
                target_has_identity = canonical_row[4] is not None and canonical_row[5] is not None
                if source_has_identity and target_has_identity:
                    same_size = row[4] == canonical_row[4]
                    same_hash = row[5] == canonical_row[5]
                    same_algo = (
                        row[6] == canonical_row[6]
                        or row[6] is None
                        or canonical_row[6] is None
                    )
                    if not (same_size and same_hash and same_algo):
                        raise ValueError(
                            f"source row {doc_id} does not match canonical hash cluster {canonical_doc_id}"
                        )
                self._conn.execute(
                    """
                    UPDATE doc_registry
                    SET dedupe_status = 'duplicate',
                        canonical_doc_id = ?,
                        archive_path = COALESCE(?, archive_path),
                        duplicate_reason = COALESCE(?, duplicate_reason),
                        duplicate_of_doc_id = ?,
                        first_seen_at = COALESCE(first_seen_at, ?),
                        last_seen_at = ?
                    WHERE doc_id = ?
                    """,
                    (
                        canonical_doc_id,
                        archive_path,
                        duplicate_reason,
                        canonical_doc_id,
                        row[12] if row[12] is not None else row[2],
                        now,
                        doc_id,
                    ),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def duplicate_refs_for_canonical(self, canonical_doc_id: str) -> list[dict]:
        """Return duplicate rows that point at a canonical doc_id."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT doc_id, rel_path, created, source_name, size_bytes, content_hash,
                       hash_algo, dedupe_status, canonical_doc_id, archive_path,
                       duplicate_reason, duplicate_of_doc_id, first_seen_at, last_seen_at
                FROM doc_registry
                WHERE canonical_doc_id = ? AND dedupe_status = 'duplicate'
                ORDER BY COALESCE(first_seen_at, created), created
                """,
                (canonical_doc_id,),
            ).fetchall()
        return [
            self._registry_row_to_dict(row)
            for row in rows
        ]

    def count(self) -> int:
        """Return total number of registered documents."""
        with self._lock:
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
        with self._lock:
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

        with self._lock:
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
        with self._lock:
            return self._conn.execute(sql, params).fetchone()[0]

    def close(self) -> None:
        """Close the SQLite connection."""
        with self._lock:
            self._conn.close()
