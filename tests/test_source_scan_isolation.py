"""Per-source isolation in the scan loop (#2020).

`_scan_and_register_sources` ran every configured source in one unguarded
loop, so one source's `scan()` exception aborted `index_vault_flow` and the
whole cycle — a 21-second Comm-Data-Store Postgres recovery on 2026-09-03 cost
a complete index run, with `queued=unknown processed=0`.

The flow-level guarantees (documents kept, ledger untouched, run marked
partial) are covered by tests/test_source_scan_isolation.int.test.py; these
pin the seam itself.
"""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import flow_index_vault


class _FakeSource:
    """Minimal Source stand-in yielding `count` scannable records."""

    def __init__(self, name: str, count: int, fail_after: int | None = None):
        self.name = name
        self._count = count
        self._fail_after = fail_after

    def scan(self):
        for i in range(self._count):
            if self._fail_after is not None and i >= self._fail_after:
                raise ConnectionError(f"{self.name} unavailable")
            yield SimpleNamespace(
                doc_id=f"{i:06d}",
                natural_key=f"{self.name}/doc-{i}.txt",
                mtime=float(i),
                change_hash=f"h{i}",
                size=10,
                source_type="filesystem",
                metadata={"abs_path": f"/data/{self.name}/doc-{i}.txt", "ext": ".txt"},
            )


class _FakeDocIDStore:
    def __init__(self):
        self.registered: list[tuple] = []

    def register(self, doc_id, natural_key, source_name=None):
        self.registered.append((doc_id, natural_key, source_name))


def _scan(sources, store=None, index_root=None, logger=None):
    return flow_index_vault._scan_and_register_sources(
        sources, store or _FakeDocIDStore(), index_root, logger=logger
    )


def test_failed_source_does_not_stop_the_remaining_sources(tmp_path):
    """The ticket's failure: one source raises, the rest must still be scanned."""
    store = _FakeDocIDStore()
    records, record_map, failed = _scan(
        [
            _FakeSource("comm_messages", count=3, fail_after=0),
            _FakeSource("documents", count=2),
        ],
        store,
        tmp_path,
    )

    assert failed == ["comm_messages"]
    assert [r["doc_id"] for r in records] == ["documents::000000", "documents::000001"]
    assert set(record_map) == {"documents::000000", "documents::000001"}
    assert [r[0] for r in store.registered] == [
        "documents::000000", "documents::000001",
    ]


def test_failed_source_contributes_no_partial_records(tmp_path):
    """A half-scanned source is unscanned, not half-emptied.

    Committing the records it managed to yield would make the missing ones
    look deleted to the diff and to the registry reap — the exact damage the
    per-source guards exist to prevent.
    """
    store = _FakeDocIDStore()
    records, record_map, failed = _scan(
        [_FakeSource("comm_messages", count=10, fail_after=6)], store, tmp_path
    )

    assert failed == ["comm_messages"]
    assert records == []
    assert record_map == {}
    assert store.registered == []


def test_every_source_failing_is_reported_in_order(tmp_path):
    """The caller decides what "all sources failed" means, so it gets them all."""
    _, _, failed = _scan(
        [
            _FakeSource("documents", count=1, fail_after=0),
            _FakeSource("comm_messages", count=1, fail_after=0),
        ],
        index_root=tmp_path,
    )

    assert failed == ["documents", "comm_messages"]


def test_failed_source_is_logged_with_its_name_and_cause(tmp_path):
    logger = Mock(spec=logging.Logger)

    _scan([_FakeSource("comm_messages", count=1, fail_after=0)], index_root=tmp_path,
          logger=logger)

    logger.error.assert_called_once()
    args = logger.error.call_args.args
    assert "comm_messages" in args
    assert "ConnectionError" in args


def test_healthy_scan_still_registers_and_reports_no_failures(tmp_path):
    store = _FakeDocIDStore()
    records, record_map, failed = _scan(
        [_FakeSource("documents", count=2)], store, tmp_path
    )

    assert failed == []
    assert len(records) == len(record_map) == 2
    assert store.registered == [
        ("documents::000000", "documents/doc-0.txt", "documents"),
        ("documents::000001", "documents/doc-1.txt", "documents"),
    ]
