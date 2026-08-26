"""#1254: the daily Lance compaction belongs to the idle window between runs.

Compaction forks a second full-memory Python worker (``core.lance_maintenance``)
into the container's memory cgroup. Run inline in a live index run it doubled
peak memory and drove that cgroup to its 8 GiB ceiling on 2026-08-14 and
2026-08-19; the kernel killed the fattest task in the cgroup — the long-lived
server — the container restarted, and the run in flight was lost with it.

The window is enforced by the table write lock every index run already holds:
taken non-blocking, a busy table means a writer is active and the compaction
defers to the next tick instead of co-residing with it.
"""

import threading
from datetime import date
from unittest.mock import MagicMock, patch

import flow_index_vault as fiv
from core.index_write_lock import index_write_lock


def _config(index_root):
    return {"index_root": str(index_root), "lancedb": {"table": "chunks"}}


def test_compaction_defers_while_an_index_writer_holds_the_table(tmp_path, caplog):
    config = _config(tmp_path / "index")
    store = MagicMock()
    acquired = threading.Event()
    release = threading.Event()

    def hold_the_table():
        with index_write_lock(config["index_root"], "chunks"):
            acquired.set()
            release.wait(10)

    holder = threading.Thread(target=hold_the_table)
    holder.start()
    try:
        assert acquired.wait(5)
        with (
            patch.object(fiv, "load_config", return_value=config),
            patch.object(fiv, "open_store_with_recovery", return_value=store),
            patch.object(fiv, "compaction_is_due", return_value=True),
            caplog.at_level("INFO"),
        ):
            result = fiv.compact_index_if_idle()
    finally:
        release.set()
        holder.join(5)

    assert result["status"] == "writer_busy"
    store.compact_data_files_if_due.assert_not_called()
    assert "compaction deferred" in caplog.text


def test_compaction_runs_when_no_index_writer_holds_the_table(tmp_path):
    config = _config(tmp_path / "index")
    store = MagicMock()
    store.compact_data_files_if_due.return_value = True

    with (
        patch.object(fiv, "load_config", return_value=config),
        patch.object(fiv, "open_store_with_recovery", return_value=store),
        patch.object(fiv, "compaction_is_due", return_value=True),
    ):
        result = fiv.compact_index_if_idle()

    assert result["status"] == "compacted"
    store.compact_data_files_if_due.assert_called_once_with(date.today())


def test_compaction_does_not_open_a_store_when_not_due(tmp_path):
    """The idle tick is cheap when there is nothing to do: no store, no lock."""
    config = _config(tmp_path / "index")

    with (
        patch.object(fiv, "load_config", return_value=config),
        patch.object(fiv, "open_store_with_recovery") as open_store,
        patch.object(fiv, "compaction_is_due", return_value=False),
    ):
        result = fiv.compact_index_if_idle()

    assert result["status"] == "not_due"
    open_store.assert_not_called()
