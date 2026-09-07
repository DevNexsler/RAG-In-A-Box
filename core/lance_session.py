"""One bounded LanceDB session per process, shared by every connection.

LanceDB keeps its index and metadata caches inside the ``Session`` that owns a
connection, and a connection opened without an explicit session gets a fresh
session sized by LanceDB's own defaults — sized for a dedicated host, not for
the doc-organizer's 8 GiB cgroup.  Production paid for that twice (#1157): the
long-lived server's cache grew until the kernel memcg OOM-killer took the
container (and any live index run) down, and every store reconnect — schema
swap, shadow promotion, stale-read recovery — started *another* default-sized
cache instead of reusing the one already warm.

So every LanceDB connection this process opens goes through :func:`connect`,
which shares one explicitly-sized session.  Entrypoints that have the config in
hand call :func:`configure_from_config` (see ``server.py`` and
``index_vault_flow``); everything else — scripts, tests, subprocesses — gets
the bounded defaults below rather than LanceDB's.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Lance's index statistics include the IVF centroids unless told otherwise, and
# it says so with a WARN written straight to stderr on the first index_stats()
# call — one untimestamped line in the line-parsed indexer.log (#0546) every
# sweep, and for a 256-partition x 4096-dim index a needless 4 MB copy per
# health check. Opt into the lean statistics before anything asks for them.
os.environ.setdefault("LANCE_INCLUDE_VECTOR_CENTROIDS", "false")

import lancedb  # noqa: E402

# Room for the serving hot set while leaving the 8 GiB container enough headroom
# for the Python heap and a concurrent index run.  Tune via the ``lancedb``
# config section, not by editing these.
DEFAULT_INDEX_CACHE_MB = 512
DEFAULT_METADATA_CACHE_MB = 128

_MB = 1024 * 1024

_SESSION_LOCK = threading.Lock()
_SESSION: lancedb.Session | None = None


def _new_session(index_cache_mb: int, metadata_cache_mb: int) -> lancedb.Session:
    return lancedb.Session(
        index_cache_size_bytes=int(index_cache_mb) * _MB,
        metadata_cache_size_bytes=int(metadata_cache_mb) * _MB,
    )


def configure(
    index_cache_mb: int = DEFAULT_INDEX_CACHE_MB,
    metadata_cache_mb: int = DEFAULT_METADATA_CACHE_MB,
) -> lancedb.Session:
    """Replace the process-wide session with one sized by these caps."""
    global _SESSION
    session = _new_session(index_cache_mb, metadata_cache_mb)
    with _SESSION_LOCK:
        _SESSION = session
    return session


def configure_from_config(config: Mapping[str, Any] | None) -> lancedb.Session:
    """Size the process-wide session from the ``lancedb`` config section."""
    settings = (config or {}).get("lancedb") or {}
    return configure(
        index_cache_mb=settings.get("index_cache_mb") or DEFAULT_INDEX_CACHE_MB,
        metadata_cache_mb=settings.get("metadata_cache_mb") or DEFAULT_METADATA_CACHE_MB,
    )


def get_session() -> lancedb.Session:
    """Return the process-wide session, building the bounded default once."""
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is None:
            _SESSION = _new_session(DEFAULT_INDEX_CACHE_MB, DEFAULT_METADATA_CACHE_MB)
        return _SESSION


def connect(uri: str | Path) -> lancedb.DBConnection:
    """Open a LanceDB connection bound to the shared session."""
    return lancedb.connect(str(uri), session=get_session())
