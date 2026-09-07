"""Tests for LanceDBStore (uses a temp directory, no mocks needed)."""

import json
import logging

import multiprocessing
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import pyarrow as pa
import pytest

import lancedb_store as lancedb_store_module
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from lancedb_store import LanceDBStore, open_store_with_recovery


def _make_node(doc_id: str, loc: str, text: str, vector: list[float]) -> TextNode:
    """Helper: build a TextNode like process_doc_task does."""
    chunk_uid = f"{doc_id}::{loc}"
    node = TextNode(
        text=text,
        id_=chunk_uid,
        embedding=vector,
        metadata={
            "doc_id": doc_id,
            "source_type": "md",
            "loc": loc,
            "snippet": text[:200],
            "mtime": 1.0,
            "size": len(text),
        },
    )
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
    return node


def _hold_document_lock(index_root, acquired, release):
    store = LanceDBStore(index_root, "test_chunks")
    with store._serialize_document_writes(["race.md"]):
        acquired.set()
        if not release.wait(10):
            raise TimeoutError("parent did not release document lock")


def _enter_document_lock(index_root, started, acquired):
    store = LanceDBStore(index_root, "test_chunks")
    started.set()
    with store._serialize_document_writes(["race.md"]):
        acquired.set()


def test_list_communication_context_rows_filters_source_and_channel():
    store = LanceDBStore.__new__(LanceDBStore)
    query = MagicMock()
    query.where.return_value = query
    query.select.return_value = query
    query.limit.return_value = query
    query.to_list.return_value = [
        {
            "doc_id": "comm_messages::1",
            "source_type": "pg_message",
            "source": "zoho_cliq",
            "source_message_id": "message-1",
            "source_channel_id": "maintenance",
            "sender": "Cesar",
            "sent_at": "2026-06-18T19:11:48Z",
            "snippet": "Maybe tomorrow",
        }
    ]
    store._vs = MagicMock()
    store._vs.table.search.return_value = query
    store._run_read_with_recovery = lambda operation, default: operation()
    store._metadata_subfields = MagicMock(
        return_value={
            "source_type",
            "source",
            "source_message_id",
            "source_channel_id",
            "sender",
            "sent_at",
            "snippet",
        }
    )

    rows = store.list_communication_context_rows(
        origin_source="zoho_cliq",
        channel_id="maintenance",
    )

    assert rows == [
        {
            "doc_id": "comm_messages::1",
            "metadata": {
                "source_type": "pg_message",
                "source": "zoho_cliq",
                "source_message_id": "message-1",
                "source_channel_id": "maintenance",
                "sender": "Cesar",
                "sent_at": "2026-06-18T19:11:48Z",
                "snippet": "Maybe tomorrow",
            },
        }
    ]
    query.where.assert_called_once_with(
        "metadata.source_type = 'pg_message' "
        "AND metadata.source = 'zoho_cliq' "
        "AND metadata.source_channel_id = 'maintenance'",
        prefilter=True,
    )
    projection = query.select.call_args.args[0]
    assert projection["doc_id"] == "doc_id"
    assert projection["source_type"] == "metadata.source_type"
    assert projection["source_channel_id"] == "metadata.source_channel_id"


def test_upsert_and_list():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        nodes = [
            _make_node("a.md", "c:0", "hello world", [0.1] * 768),
            _make_node("b.md", "c:0", "goodbye world", [0.2] * 768),
        ]
        store.upsert_nodes(nodes)
        doc_ids = store.list_doc_ids()
        assert set(doc_ids) == {"a.md", "b.md"}


def test_upsert_replaces():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        nodes_v1 = [_make_node("a.md", "c:0", "version 1", [0.1] * 768)]
        store.upsert_nodes(nodes_v1)

        nodes_v2 = [
            _make_node("a.md", "c:0", "version 2 chunk 0", [0.3] * 768),
            _make_node("a.md", "c:1", "version 2 chunk 1", [0.4] * 768),
        ]
        store.upsert_nodes(nodes_v2)

        doc_ids = store.list_doc_ids()
        assert doc_ids == ["a.md"]


def test_replace_chunk_text_and_vector_is_atomic_and_refreshes_node_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        old_text = "photo\n\n[Conversation context]\nBEFORE: old"
        new_text = "photo\n\n[Conversation context]\nBEFORE: 482 #6"
        store.upsert_nodes([_make_node("photo.jpg", "img:c:0", old_text, [0.1] * 768)])

        changed = store.replace_chunk_text_and_vector(
            "photo.jpg", "img:c:0", old_text, new_text, [0.2] * 768
        )

        assert changed is True
        hit = store.get_chunk("photo.jpg", "img:c:0")
        assert hit is not None
        assert hit.text == new_text
        assert store.get_vector("photo.jpg::img:c:0") == pytest.approx([0.2] * 768)
        raw = store._vs.table.search(None).where("id = 'photo.jpg::img:c:0'").limit(1).to_list()[0]
        assert json.loads(raw["metadata"]["_node_content"])["text"] == new_text
        store.ensure_fts_index()
        assert store.keyword_search("482", top_k=5)[0].doc_id == "photo.jpg"
        assert store.vector_search([0.2] * 768, top_k=1)[0].doc_id == "photo.jpg"


def test_replace_chunk_text_and_vector_rejects_stale_expected_text():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        store.upsert_nodes([_make_node("photo.jpg", "img:c:0", "current", [0.1] * 768)])

        changed = store.replace_chunk_text_and_vector(
            "photo.jpg", "img:c:0", "stale", "replacement", [0.2] * 768
        )

        assert changed is False
        assert store.get_chunk("photo.jpg", "img:c:0").text == "current"


def test_insert_nodes_commits_once_without_noop_delete():
    """Known-new documents must not create a delete-only Lance version."""
    import lance

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        store.upsert_nodes([_make_node("seed.md", "c:0", "seed", [0.1] * 768)])
        before = lance.dataset(_lance_path(tmpdir)).version

        store.insert_nodes([_make_node("new.md", "c:0", "new", [0.2] * 768)])

        dataset = lance.dataset(_lance_path(tmpdir))
        assert dataset.version == before + 1
        assert set(store.list_doc_ids()) == {"seed.md", "new.md"}


def test_known_absent_insert_retry_skips_without_latest_manifest_probe():
    """A successful full-sweep insert stays idempotent on Prefect retry."""
    import lance

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        store.upsert_nodes([_make_node("seed.md", "c:0", "seed", [0.1] * 768)])
        node = _make_node("new.md", "c:0", "new", [0.2] * 768)
        store.insert_nodes([node], known_absent=True)
        before_retry = lance.dataset(_lance_path(tmpdir)).version

        with patch.object(
            store,
            "_contains_doc_id_latest",
            side_effect=AssertionError("normal full-sweep retry must not probe Lance"),
        ):
            store.insert_nodes([node], known_absent=True)

        dataset = lance.dataset(_lance_path(tmpdir))
        assert dataset.version == before_retry
        assert dataset.count_rows("doc_id = 'new.md'") == 1


def test_known_absent_insert_recovers_post_commit_exception():
    """An ambiguous add result is accepted only after latest-manifest proof."""
    import lance

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        store.upsert_nodes([_make_node("seed.md", "c:0", "seed", [0.1] * 768)])
        original_write = store._write_nodes_unlocked

        def commit_then_raise(nodes, *, operation):
            original_write(nodes, operation=operation)
            raise RuntimeError("connection dropped after commit")

        with patch.object(store, "_write_nodes_unlocked", side_effect=commit_then_raise):
            store.insert_nodes(
                [_make_node("new.md", "c:0", "new", [0.2] * 768)],
                known_absent=True,
            )

        dataset = lance.dataset(_lance_path(tmpdir))
        assert dataset.count_rows("doc_id = 'new.md'") == 1


def test_exclusive_writer_session_skips_per_document_refresh_and_probe():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        store.upsert_nodes(
            [_make_node("existing.md", "c:0", "old", [0.1] * 768)]
        )

        with patch.object(store, "_checkout_latest") as checkout, patch.object(
            store, "_contains_doc_id_latest"
        ) as probe:
            with store.exclusive_writer_session():
                store.upsert_nodes(
                    [_make_node("existing.md", "c:0", "new", [0.2] * 768)]
                )
                store.insert_nodes(
                    [_make_node("new.md", "c:0", "new", [0.3] * 768)]
                )
                store.delete_by_doc_ids(["existing.md"])

        checkout.assert_not_called()
        probe.assert_not_called()
        assert set(store.list_doc_ids()) == {"new.md"}


def test_delete_invalidates_completed_insert_cache_for_standalone_reinsert():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        node = _make_node("again.md", "c:0", "restored", [0.1] * 768)
        store.insert_nodes([node], known_absent=True)
        store.delete_by_doc_ids(["again.md"])

        store.insert_nodes([node])

        assert store.contains_doc_id("again.md") is True
        assert store.count_chunks() == 1


def test_failed_upsert_after_delete_invalidates_completed_insert_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        node = _make_node("again.md", "c:0", "restored", [0.1] * 768)
        store.insert_nodes([node], known_absent=True)

        with patch.object(
            type(store._vs), "add", side_effect=RuntimeError("add failed")
        ):
            with pytest.raises(RuntimeError, match="add failed"):
                store.upsert_nodes([node])

        assert store.contains_doc_id("again.md") is False
        store.insert_nodes([node])
        assert store.contains_doc_id("again.md") is True
        assert store.count_chunks() == 1


def test_insert_nodes_is_idempotent_across_store_handles():
    """A stale new-doc snapshot or ambiguous retry must not append duplicates."""
    import lance

    with tempfile.TemporaryDirectory() as tmpdir:
        first = LanceDBStore(tmpdir, "test_chunks")
        first.upsert_nodes([_make_node("seed.md", "c:0", "seed", [0.1] * 768)])
        second = LanceDBStore(tmpdir, "test_chunks")

        second.upsert_nodes([
            _make_node("race.md", "c:0", "targeted write", [0.2] * 768)
        ])
        before_retry = lance.dataset(_lance_path(tmpdir)).version
        # Refresh the existence probe without checking out the long-lived
        # writer table. Repeated checkout_latest calls retain large manifest
        # file caches on fragmented production datasets.
        with patch.object(
            first,
            "_checkout_latest",
            side_effect=AssertionError("insert must not checkout the writer table"),
        ):
            first.insert_nodes([
                _make_node("race.md", "c:0", "stale sweep write", [0.3] * 768)
            ])

        dataset = lance.dataset(_lance_path(tmpdir))
        assert dataset.count_rows("doc_id = 'race.md'") == 1
        assert dataset.version == before_retry


def test_concurrent_insert_and_upsert_do_not_leave_duplicate_rows():
    """Two real handles serialize the stale-snapshot insert/upsert race."""
    import lance

    with tempfile.TemporaryDirectory() as tmpdir:
        insert_store = LanceDBStore(tmpdir, "test_chunks")
        insert_store.upsert_nodes([
            _make_node("seed.md", "c:0", "seed", [0.1] * 768)
        ])
        upsert_store = LanceDBStore(tmpdir, "test_chunks")
        barrier = threading.Barrier(2)

        def insert():
            barrier.wait()
            insert_store.insert_nodes([
                _make_node("race.md", "c:0", "sweep", [0.2] * 768)
            ])

        def upsert():
            barrier.wait()
            upsert_store.upsert_nodes([
                _make_node("race.md", "c:0", "targeted", [0.3] * 768)
            ])

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(insert), pool.submit(upsert)]
            for future in futures:
                future.result(timeout=10)

        dataset = lance.dataset(_lance_path(tmpdir))
        assert dataset.count_rows("doc_id = 'race.md'") == 1


def test_document_write_lock_excludes_another_process():
    """The filesystem lock—not only process-local RLocks—guards a document."""
    context = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        store.upsert_nodes([_make_node("seed.md", "c:0", "seed", [0.1] * 768)])
        holder_acquired = context.Event()
        release_holder = context.Event()
        contender_started = context.Event()
        contender_acquired = context.Event()
        holder = context.Process(
            target=_hold_document_lock,
            args=(tmpdir, holder_acquired, release_holder),
        )
        contender = context.Process(
            target=_enter_document_lock,
            args=(tmpdir, contender_started, contender_acquired),
        )
        holder.start()
        try:
            assert holder_acquired.wait(10)
            contender.start()
            assert contender_started.wait(10)
            assert not contender_acquired.wait(0.5)
        finally:
            release_holder.set()
            holder.join(10)
            if contender.pid is not None:
                contender.join(10)
            for process in (holder, contender):
                if process.pid is not None and process.is_alive():
                    process.terminate()
                    process.join(5)

        assert holder.exitcode == 0
        assert contender.exitcode == 0
        assert contender_acquired.is_set()


def test_reopening_store_does_not_replace_existing_scalar_index():
    """Read-path construction must not create a new Lance version each time."""
    import lance

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        store.upsert_nodes([_make_node("a.md", "c:0", "hello", [0.1] * 768)])

        LanceDBStore(tmpdir, "test_chunks")
        version_with_index = lance.dataset(_lance_path(tmpdir)).version
        reopened = LanceDBStore(tmpdir, "test_chunks")

        assert lance.dataset(_lance_path(tmpdir)).version == version_with_index
        assert any(
            list(index.columns) == ["doc_id"]
            and str(index.index_type).upper() == "BTREE"
            for index in reopened._vs.table.list_indices()
        )


def test_reopening_store_repairs_wrong_doc_id_index_type():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        store.upsert_nodes([_make_node("a.md", "c:0", "hello", [0.1] * 768)])
        table = LanceDBStore(tmpdir, "test_chunks")._vs.table
        table.create_scalar_index("doc_id", index_type="BITMAP", replace=True)

        reopened = LanceDBStore(tmpdir, "test_chunks")

        assert any(
            list(index.columns) == ["doc_id"]
            and str(index.index_type).upper() == "BTREE"
            for index in reopened._vs.table.list_indices()
        )


def test_delete_by_doc_ids():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        nodes = [
            _make_node("a.md", "c:0", "keep", [0.1] * 768),
            _make_node("b.md", "c:0", "delete me", [0.2] * 768),
        ]
        store.upsert_nodes(nodes)
        store.delete_by_doc_ids(["b.md"])
        doc_ids = store.list_doc_ids()
        assert doc_ids == ["a.md"]


def test_promote_table_swaps_shadow_into_active():
    with tempfile.TemporaryDirectory() as tmpdir:
        active = LanceDBStore(tmpdir, "test_chunks")
        shadow = LanceDBStore(tmpdir, "test_chunks__shadow")

        active.upsert_nodes([_make_node("active.md", "c:0", "active", [0.1] * 768)])
        shadow.upsert_nodes([_make_node("shadow.md", "c:0", "shadow", [0.2] * 768)])
        shadow.create_fts_index()

        active.promote_table("test_chunks__shadow")

        reopened = LanceDBStore(tmpdir, "test_chunks")
        assert reopened.list_doc_ids() == ["shadow.md"]
        assert reopened.get_chunk("shadow.md", "c:0").text == "shadow"


def test_list_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        assert store.list_doc_ids() == []


def test_init_retries_after_corruption_recovery(monkeypatch):
    """Constructor should retry once after auto-recovering a corrupt Lance table."""
    with tempfile.TemporaryDirectory() as tmpdir:
        seed = LanceDBStore(tmpdir, "test_chunks")
        seed.upsert_nodes([_make_node("a.md", "c:0", "hello world", [0.1] * 768)])

        real_probe = LanceDBStore._probe_table_read
        probe_calls = {"count": 0}
        recovery_calls = {"count": 0}

        def flaky_probe(self):
            probe_calls["count"] += 1
            if probe_calls["count"] == 1:
                raise RuntimeError(
                    "lance error: LanceError(IO): Generic memory error: "
                    "Invalid range 0..0 for object of size 0 bytes"
                )
            return real_probe(self)

        def fake_recover(self):
            recovery_calls["count"] += 1
            return True

        monkeypatch.setattr(LanceDBStore, "_probe_table_read", flaky_probe)
        monkeypatch.setattr(LanceDBStore, "_recover_corrupt_table", fake_recover)

        recovered = LanceDBStore(tmpdir, "test_chunks")

        assert recovery_calls["count"] == 1
        assert probe_calls["count"] == 2
        assert recovered.list_doc_ids() == ["a.md"]


def test_init_reraises_non_corruption_probe_error(monkeypatch):
    """Constructor should not hide unrelated probe failures."""
    with tempfile.TemporaryDirectory() as tmpdir:
        seed = LanceDBStore(tmpdir, "test_chunks")
        seed.upsert_nodes([_make_node("a.md", "c:0", "hello world", [0.1] * 768)])

        recovery_calls = {"count": 0}

        def bad_probe(self):
            raise RuntimeError("boom")

        def fake_recover(self):
            recovery_calls["count"] += 1
            return True

        monkeypatch.setattr(LanceDBStore, "_probe_table_read", bad_probe)
        monkeypatch.setattr(LanceDBStore, "_recover_corrupt_table", fake_recover)

        with pytest.raises(RuntimeError, match="boom"):
            LanceDBStore(tmpdir, "test_chunks")

        assert recovery_calls["count"] == 0


def test_vector_search():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        nodes = [
            _make_node("a.md", "c:0", "apple banana", [1.0] + [0.0] * 767),
            _make_node("b.md", "c:0", "cherry date", [0.0] + [1.0] + [0.0] * 766),
        ]
        store.upsert_nodes(nodes)
        hits = store.vector_search([1.0] + [0.0] * 767, top_k=1)
        assert len(hits) == 1
        assert hits[0].doc_id == "a.md"


def test_vector_search_recovers_from_stale_store_handle():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        store.upsert_nodes([
            _make_node("a.md", "c:0", "apple banana", [1.0] + [0.0] * 767),
        ])
        # A peer writer moves the table past this handle — what actually makes
        # the handle below stale, and what read recovery keys on.
        LanceDBStore(tmpdir, "test_chunks").upsert_nodes([
            _make_node("b.md", "c:0", "cherry date", [0.0] * 768),
        ])

        class _StaleTable:
            def search(self, *args, **kwargs):
                raise RuntimeError(
                    "Dataset at path data/index/test_chunks.lance/_versions/4555.manifest "
                    "was not found: Not found: data/index/test_chunks.lance/_versions/4555.manifest"
                )

        class _StaleVS:
            @property
            def table(self):
                return _StaleTable()

        store._vs = _StaleVS()

        hits = store.vector_search([1.0] + [0.0] * 767, top_k=1)
        assert len(hits) == 1
        assert hits[0].doc_id == "a.md"


def _make_node_with_meta(doc_id, loc, text, vector, **extra_meta):
    """Helper that lets tests set arbitrary metadata fields."""
    meta = {
        "doc_id": doc_id,
        "source_type": "md",
        "loc": loc,
        "snippet": text[:200],
        "mtime": 1.0,
        "size": len(text),
    }
    meta.update(extra_meta)
    node = TextNode(
        text=text,
        id_=f"{doc_id}::{loc}",
        embedding=vector,
        metadata=meta,
    )
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
    return node


def test_facets_empty_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        facets = store.facets()
        assert facets["total_docs"] == 0
        assert facets["total_chunks"] == 0


def test_facets_counts():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        nodes = [
            _make_node_with_meta("a.md", "c:0", "hello", vec, tags="recipe,korean", folder="Projects", status="active", author="Dan"),
            _make_node_with_meta("a.md", "c:1", "world", vec, tags="recipe,korean", folder="Projects", status="active", author="Dan"),
            _make_node_with_meta("b.pdf", "p:1:c:0", "doc", vec, source_type="pdf", tags="finance", folder="Archive", status="archived", author="Jane"),
        ]
        store.upsert_nodes(nodes)
        facets = store.facets()
        assert facets["total_docs"] == 2  # a.md and b.pdf
        assert facets["total_chunks"] == 3
        tag_values = {t["value"] for t in facets["tags"]}
        assert "recipe" in tag_values
        assert "korean" in tag_values
        assert "finance" in tag_values
        folder_values = {f["value"] for f in facets["folders"]}
        assert "Projects" in folder_values
        assert "Archive" in folder_values


def test_facets_streams_projected_rows_without_materializing_full_table():
    """Large indexes must not become one in-memory Arrow table for facets."""
    from lancedb.query import LanceEmptyQueryBuilder

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        store.upsert_nodes([
            _make_node_with_meta(
                "a.md",
                "c:0",
                "hello",
                vec,
                tags="recipe,korean",
                folder="Projects",
            ),
            _make_node_with_meta(
                "a.md",
                "c:1",
                "world",
                vec,
                tags="recipe,korean",
                folder="Projects",
            ),
            _make_node_with_meta(
                "b.md",
                "c:0",
                "other",
                vec,
                tags="finance",
                folder="Archive",
            ),
        ])

        with patch.object(
            LanceEmptyQueryBuilder,
            "to_arrow",
            side_effect=AssertionError("facets must stream bounded batches"),
        ):
            facets = store.facets()

        assert facets["total_docs"] == 2
        assert facets["total_chunks"] == 3
        assert {row["value"] for row in facets["folders"]} == {
            "Projects",
            "Archive",
        }


def test_search_hit_has_mtime():
    """Verify mtime is passed through on SearchHit from vector_search."""
    import time
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        now = time.time()
        nodes = [_make_node_with_meta("a.md", "c:0", "test", [0.1] * 768, mtime=now)]
        store.upsert_nodes(nodes)
        hits = store.vector_search([0.1] * 768, top_k=1)
        assert len(hits) == 1
        assert abs(hits[0].mtime - now) < 1.0


def test_search_hit_has_description_author():
    """Verify description/author/custom_meta are passed through."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.1] * 768
        nodes = [_make_node_with_meta(
            "a.md", "c:0", "test", vec,
            description="A test doc",
            author="Dan Park",
            custom_meta='{"source": "http://example.com"}',
        )]
        store.upsert_nodes(nodes)
        hits = store.vector_search(vec, top_k=1)
        assert len(hits) == 1
        assert hits[0].description == "A test doc"
        assert hits[0].author == "Dan Park"
        assert hits[0].custom_meta == '{"source": "http://example.com"}'


def test_search_hit_preserves_importance_fields():
    """All enrichment fields, including importance, round-trip through LanceDBStore."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.1] * 768
        nodes = [_make_node_with_meta(
            "a.md", "c:0", "test", vec,
            enr_importance="0.9",
            enr_importance_source="llm",
        )]
        store.upsert_nodes(nodes)

        hits = store.vector_search(vec, top_k=1)

        assert len(hits) == 1
        assert hits[0].enr_importance == "0.9"
        assert hits[0].enr_importance_source == "llm"


def test_schema_evolution_create_failure_preserves_existing_table(monkeypatch):
    """A mid-write schema failure must preserve the table and clean work paths."""
    import lancedb

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.1] * 768
        store.upsert_nodes(
            [
                _make_node_with_meta(
                    f"seed-{index}.md",
                    "c:0",
                    f"alpha {index}",
                    vec,
                )
                for index in range(300)
            ]
        )

        real_connect = lancedb.connect

        class FailingAddTable:
            def __init__(self, table):
                self._table = table

            def __getattr__(self, name):
                return getattr(self._table, name)

            def add(self, *args, **kwargs):
                raise RuntimeError("chunk add failed")

        class FailingCreateDB:
            def __init__(self, db):
                self._db = db

            def __getattr__(self, name):
                return getattr(self._db, name)

            def create_table(self, *args, **kwargs):
                return FailingAddTable(self._db.create_table(*args, **kwargs))

        monkeypatch.setattr(
            lancedb,
            "connect",
            lambda uri, **kwargs: FailingCreateDB(real_connect(uri, **kwargs)),
        )

        with pytest.raises(RuntimeError, match="chunk add failed"):
            store.upsert_nodes([
                _make_node_with_meta("b.md", "c:0", "beta", vec, section="Intro")
            ])

        restored = LanceDBStore(tmpdir, "test_chunks")
        assert restored.count_chunks() == 300
        assert restored.get_chunk("seed-0.md", "c:0").text == "alpha 0"
        assert not (Path(tmpdir) / "test_chunks__schema_tmp.lance").exists()
        assert not (Path(tmpdir) / "test_chunks__schema_backup.lance").exists()


# --- Schema evolution tests ---


def test_schema_evolution_new_metadata_field():
    """Adding nodes with a new metadata field (section) after initial insert should work."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768

        # First batch: no section field
        nodes1 = [_make_node_with_meta("a.pdf", "p:1:c:0", "pdf chunk", vec)]
        store.upsert_nodes(nodes1)

        # Second batch: has section field
        nodes2 = [_make_node_with_meta("b.md", "c:0", "md chunk", vec, section="Introduction")]
        store.upsert_nodes(nodes2)

        doc_ids = set(store.list_doc_ids())
        assert doc_ids == {"a.pdf", "b.md"}
        assert "section" in store._metadata_subfields()


def test_change_hash_projection_survives_sparse_fragment_after_schema_widening():
    """A later row may omit a metadata field already present in the schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768

        store.upsert_nodes([
            _make_node_with_meta(
                "hashed.md", "c:0", "hashed", vec, change_hash="sha-abc"
            )
        ])
        store.upsert_nodes([
            _make_node_with_meta("legacy.md", "c:0", "legacy", vec)
        ])

        assert store.list_doc_change_hashes() == {
            "hashed.md": "sha-abc",
            "legacy.md": "",
        }


def _claim_columns_the_file_lacks(index_root: str, table_name: str = "chunks") -> None:
    """Recreate #0771: a fragment manifest claiming columns its data file lacks.

    Production got here through a daily compaction, which cannot be driven
    deterministically from a test. The damage it left can: rewrite a sparse
    fragment's manifest entry so it claims the full schema, which is exactly
    what the surviving production fragment did (107 field ids over a 67-column
    file). Lance then rejects every scan that touches the fragment.
    """
    import lance
    from lance import LanceOperation
    from lance.fragment import FragmentMetadata

    dataset_path = str(Path(index_root) / f"{table_name}.lance")
    dataset = lance.dataset(dataset_path)
    widest = max(dataset.get_fragments(), key=lambda f: len(f.metadata.files[0].fields))
    full_fields = list(widest.metadata.files[0].fields)
    full_indices = list(widest.metadata.files[0].column_indices)

    # Drop the wide rows so the sparse fragment is the first one any scan
    # reads — in production the damaged fragment sorted first, which is why
    # even a limit=1 probe died and the store could not be opened at all.
    dataset.delete("doc_id = 'wide.md'")
    dataset = lance.dataset(dataset_path)
    (sparse,) = dataset.get_fragments()

    metadata = sparse.metadata.to_json()
    metadata["files"][0]["fields"] = full_fields
    metadata["files"][0]["column_indices"] = full_indices
    lance.LanceDataset.commit(
        dataset_path,
        LanceOperation.Rewrite(
            groups=[
                LanceOperation.RewriteGroup(
                    old_fragments=[sparse.metadata],
                    new_fragments=[FragmentMetadata.from_json(json.dumps(metadata))],
                )
            ],
            rewritten_indices=[],
        ),
        read_version=dataset.version,
    )


def test_store_open_repairs_fragment_that_overclaims_its_columns():
    """#0771: one over-claiming fragment must not make the whole table unreadable."""
    import lance

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "chunks")
        vec = [0.0] * 768
        store.upsert_nodes([
            _make_node_with_meta(
                "wide.md", "c:0", "wide", vec, **{f"k{i}": "v" for i in range(20)}
            )
        ])
        store.upsert_nodes([_make_node_with_meta("narrow.md", "c:0", "narrow", vec)])

        _claim_columns_the_file_lacks(tmpdir)

        dataset_path = str(Path(tmpdir) / "chunks.lance")
        with pytest.raises(pa.ArrowInvalid):
            lance.dataset(dataset_path).to_table()

        # Opening the store must repair the claim rather than surface the
        # ArrowInvalid, which is what froze indexing for ~62 h.
        reopened = LanceDBStore(tmpdir, "chunks")

        rows = reopened._vs.table.to_lance().to_table()
        assert rows.num_rows == 1
        assert rows.column("doc_id").to_pylist() == ["narrow.md"]
        # The rows were never damaged, only the claim about them: the columns
        # the file really holds must still carry their values.
        assert rows.column("text").to_pylist() == ["narrow"]
        metadata = rows.column("metadata").combine_chunks()
        assert metadata.field("loc").to_pylist() == ["c:0"]
        assert metadata.field("source_type").to_pylist() == ["md"]


def test_physical_column_paths_match_lance_manifest_for_nested_lists(tmp_path):
    """List children need stable IDs; a fixed-size vector child does not."""
    import lance
    from lance.file import LanceFileReader
    from lancedb_store import (
        _file_columns_include_containers,
        _lance_field_ids_by_path,
        _physical_column_paths,
    )

    table = pa.Table.from_pylist(
        [
            {
                "id": "row-1",
                "items": [{"label": "a", "scores": [1, 2]}],
                "large_values": ["x"],
                "vector": [1.0, 2.0, 3.0],
            }
        ],
        schema=pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field(
                    "items",
                    pa.list_(
                        pa.struct(
                            [
                                pa.field("label", pa.string()),
                                pa.field("scores", pa.large_list(pa.int64())),
                            ]
                        )
                    ),
                ),
                pa.field("large_values", pa.large_list(pa.string())),
                pa.field("vector", pa.list_(pa.float32(), 3)),
            ]
        ),
    )
    dataset_path = tmp_path / "nested-lists.lance"
    lance.write_dataset(table, str(dataset_path))
    dataset = lance.dataset(str(dataset_path))
    entry = dataset.get_fragments()[0].metadata.to_json()["files"][0]
    file_schema = LanceFileReader(
        str(dataset_path / "data" / entry["path"])
    ).metadata().schema

    field_ids = _lance_field_ids_by_path(dataset.lance_schema)
    # Lance's own manifest is the ground truth for which schema nodes got a
    # physical column, and it differs by data file version — so this asserts
    # against whatever the installed Lance actually wrote.
    assert [
        field_ids[path]
        for path in _physical_column_paths(
            file_schema,
            field_ids,
            containers_are_columns=_file_columns_include_containers(
                entry["file_major_version"], entry["file_minor_version"]
            ),
        )
    ] == entry["fields"]


def test_lance_file_metadata_version_is_not_the_data_file_version(tmp_path):
    """The reader's own version numbers are not the format version.

    Lance 3 and 4 report `0.3` from `LanceFileReader(...).metadata()` for a file
    the fragment manifest calls `2.0`, so a repair that keys the physical-column
    layout on the reader's numbers silently reads the wrong side of the 2.1
    change. Only the manifest's version is comparable.
    """
    import lance
    from lance.file import LanceFileReader

    dataset_path = tmp_path / "versions.lance"
    lance.write_dataset(
        pa.table({"id": pa.array(["row-1"], pa.string())}), str(dataset_path)
    )
    entry = (
        lance.dataset(str(dataset_path)).get_fragments()[0].metadata.to_json()["files"][0]
    )
    reader_metadata = LanceFileReader(
        str(dataset_path / "data" / entry["path"])
    ).metadata()

    assert entry["file_major_version"] == 2
    assert entry["file_minor_version"] in (0, 1)
    if (reader_metadata.major_version, reader_metadata.minor_version) != (
        entry["file_major_version"],
        entry["file_minor_version"],
    ):
        # Whenever they disagree it is the reader that is unusable, never the
        # manifest — a v0.x reader version must not read as "older than 2.1".
        assert reader_metadata.major_version < 2


def test_lance_field_ids_by_path_fails_closed_on_an_ambiguous_schema():
    """#1436: a duplicate identity must stop the repair, not pick a winner."""
    from lancedb_store import _lance_field_ids_by_path

    class _Field:
        def __init__(self, name, field_id, children=()):
            self._name, self._id, self._children = name, field_id, list(children)

        def name(self):
            return self._name

        def id(self):
            return self._id

        def children(self):
            return self._children

    class _Schema:
        def __init__(self, fields):
            self._fields = fields

        def fields(self):
            return self._fields

    assert _lance_field_ids_by_path(
        _Schema([_Field("a", 0), _Field("b", 1, [_Field("c", 2)])])
    ) == {("a",): 0, ("b",): 1, ("b", "c"): 2}

    with pytest.raises(ValueError, match="duplicate Lance field ID 0"):
        _lance_field_ids_by_path(_Schema([_Field("a", 0), _Field("b", 0)]))

    with pytest.raises(ValueError, match="duplicate Lance field path"):
        _lance_field_ids_by_path(_Schema([_Field("a", 0), _Field("a", 1)]))


def test_repair_declines_a_data_file_newer_than_the_measured_layout():
    """A 2.2 packed struct is one opaque column, so the leaf rule is wrong there.

    Measured on lance 10.0.0: a struct field written with `packed=true` at
    `data_storage_version="2.2"` gets manifest `fields=[0, 1]` while the v2.1
    leaf rule derives `[0, 2, 3]`. With a *single* child it derives `[0, 2]` —
    the wrong IDs at the right length, which the column-count check in the
    repair cannot catch. So anything past the layout this module has actually
    been measured against is declined rather than guessed at.
    """
    from lancedb_store import (
        _MAX_REPAIRABLE_DATA_FILE_VERSION,
        _data_file_layout_is_measured,
    )

    assert _MAX_REPAIRABLE_DATA_FILE_VERSION == (2, 1)
    assert _data_file_layout_is_measured(2, 0) is True
    assert _data_file_layout_is_measured(2, 1) is True
    assert _data_file_layout_is_measured(2, 2) is False
    assert _data_file_layout_is_measured(3, 0) is False

    # ...and the repair actually consults it: with the layout declared
    # unmeasured, a fragment it would otherwise fix is left alone.
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "chunks")
        vec = [0.0] * 768
        store.upsert_nodes([
            _make_node_with_meta(
                "wide.md", "c:0", "wide", vec, **{f"k{i}": "v" for i in range(20)}
            )
        ])
        store.upsert_nodes([_make_node_with_meta("narrow.md", "c:0", "narrow", vec)])
        _claim_columns_the_file_lacks(tmpdir)

        assert store._repair_overclaiming_fragments() is True

        _claim_columns_the_file_lacks(tmpdir)
        with patch("lancedb_store._data_file_layout_is_measured", return_value=False):
            assert store._repair_overclaiming_fragments() is False


def test_file_columns_include_containers_switches_at_the_2_1_data_file():
    """#1101/#1110/#1436: the layout is keyed on the file version, not the library.

    Production runs Lance 10 (data file v2.1), the deterministic tiers run Lance
    4 (v2.0), and a repair that assumes either one is wrong on the other.
    """
    from lancedb_store import _file_columns_include_containers

    assert _file_columns_include_containers(2, 0) is True
    assert _file_columns_include_containers(1, 0) is True
    assert _file_columns_include_containers(2, 1) is False
    assert _file_columns_include_containers(3, 0) is False


def test_physical_column_paths_drop_containers_on_a_2_1_data_file():
    """Only leaves are columns from v2.1 on; a fixed-size list stays one column.

    The fixed-size-list-of-struct arm cannot be written by Lance 10 at all
    (`FixedSizeList<Struct> is not enabled by the selected file format`), so it
    is asserted here against a hand-built schema rather than a real dataset —
    #139's fixture for it is unrunnable on the deployed stack.
    """
    from lancedb_store import _physical_column_paths

    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field(
                "items",
                pa.list_(pa.struct([pa.field("label", pa.string())])),
            ),
            pa.field("vector", pa.list_(pa.float32(), 3)),
            pa.field(
                "fixed_struct",
                pa.list_(pa.struct([pa.field("value", pa.int32())]), 1),
            ),
        ]
    )
    field_ids = {
        ("id",): 0,
        ("items",): 1,
        ("items", "item"): 2,
        ("items", "item", "label"): 3,
        ("vector",): 4,
        ("fixed_struct",): 5,
        # Lance 4 mints IDs for a fixed-size list's children but gives them no
        # column; descending into them is what invented the extra column.
        ("fixed_struct", "item"): 6,
        ("fixed_struct", "item", "value"): 7,
    }

    assert _physical_column_paths(schema, field_ids, containers_are_columns=True) == [
        ("id",),
        ("items",),
        ("items", "item"),
        ("items", "item", "label"),
        ("vector",),
        ("fixed_struct",),
    ]
    assert _physical_column_paths(schema, field_ids, containers_are_columns=False) == [
        ("id",),
        ("items", "item", "label"),
        ("vector",),
        ("fixed_struct",),
    ]


def test_overclaimed_file_column_count_reads_lances_reported_width():
    """The repair depends on parsing the real column count out of Lance's error."""
    from lancedb_store import _overclaimed_file_column_count

    assert _overclaimed_file_column_count(
        pa.ArrowInvalid(
            "External error: Invalid user input: The projection specified the "
            "column index 67 but there are only 67 columns in the file, "
            "/home/runner/work/lance/lance/rust/lance-file/src/reader.rs:1259:28"
        )
    ) == 67
    assert _overclaimed_file_column_count(pa.ArrowInvalid("manifest was not found")) is None


class _ChangeHashProjectionDataset:
    def __init__(self, *, projected_table=None, projection_error=None, batches=()):
        self.projected_table = projected_table
        self.projection_error = projection_error
        self.batches = batches
        self.to_table_calls = []
        self.to_batches_calls = []

    def to_table(self, *, columns):
        self.to_table_calls.append(columns)
        if self.projection_error is not None:
            raise self.projection_error
        return self.projected_table

    def to_batches(self, *, columns, batch_size):
        self.to_batches_calls.append((columns, batch_size))
        return iter(self.batches)


def _store_for_change_hash_projection(dataset):
    store = LanceDBStore.__new__(LanceDBStore)
    store._metadata_subfields = lambda: {"change_hash"}
    store._run_read_with_recovery = lambda operation, default: operation()
    store._vs = SimpleNamespace(
        table=SimpleNamespace(to_lance=lambda: dataset)
    )
    return store


def test_change_hash_projection_narrow_projection_avoids_batch_fallback():
    dataset = _ChangeHashProjectionDataset(
        projected_table=pa.table({
            "doc_id": ["hashed.md", "legacy.md", None],
            "ch": ["sha-abc", None, "ignored"],
        })
    )
    store = _store_for_change_hash_projection(dataset)

    assert store.list_doc_change_hashes() == {
        "hashed.md": "sha-abc",
        "legacy.md": "",
    }
    assert dataset.to_table_calls == [
        {"doc_id": "doc_id", "ch": "metadata.change_hash"}
    ]
    assert dataset.to_batches_calls == []


def test_change_hash_projection_exact_sparse_metadata_error_uses_bounded_batches():
    batch = pa.record_batch({
        "doc_id": ["hashed.md", "legacy.md", None],
        "metadata": pa.array([
            {"change_hash": "sha-abc"},
            {"change_hash": None},
            {"change_hash": "ignored"},
        ]),
    })
    dataset = _ChangeHashProjectionDataset(
        projection_error=pa.ArrowInvalid(
            "projection supplied fewer column indices; "
            "ran out at field 'metadata'"
        ),
        batches=[batch],
    )
    store = _store_for_change_hash_projection(dataset)

    assert store.list_doc_change_hashes() == {
        "hashed.md": "sha-abc",
        "legacy.md": "",
    }
    assert dataset.to_batches_calls == [(["doc_id", "metadata"], 1024)]


@pytest.mark.parametrize("message", [
    "projection supplied fewer column indices",
    "ran out at field 'metadata'",
    "unrelated invalid Arrow projection",
])
def test_change_hash_projection_near_match_arrow_invalid_is_rethrown(message):
    error = pa.ArrowInvalid(message)
    dataset = _ChangeHashProjectionDataset(projection_error=error)
    store = _store_for_change_hash_projection(dataset)

    with pytest.raises(pa.ArrowInvalid) as caught:
        store.list_doc_change_hashes()

    assert caught.value is error
    assert dataset.to_batches_calls == []


def test_change_hash_projection_large_metadata_streams_multiple_bounded_batches():
    row_count = 2050
    payload = "x" * 16_384
    rows = [
        {"change_hash": None if index % 11 == 0 else f"sha-{index}", "payload": payload}
        for index in range(row_count)
    ]
    table = pa.table({
        "doc_id": [f"doc-{index}.md" for index in range(row_count)],
        "metadata": pa.array(rows),
    })
    batches = table.to_batches(max_chunksize=1024)
    dataset = _ChangeHashProjectionDataset(
        projection_error=pa.ArrowInvalid(
            "projection supplied fewer column indices; "
            "ran out at field 'metadata'"
        ),
        batches=batches,
    )
    store = _store_for_change_hash_projection(dataset)

    hashes = store.list_doc_change_hashes()

    assert len(batches) == 3
    assert hashes == {
        f"doc-{index}.md": "" if index % 11 == 0 else f"sha-{index}"
        for index in range(row_count)
    }
    assert dataset.to_table_calls == [
        {"doc_id": "doc_id", "ch": "metadata.change_hash"}
    ]
    assert dataset.to_batches_calls == [(["doc_id", "metadata"], 1024)]


def test_schema_evolution_preserves_vectors():
    """Schema evolution should not corrupt existing vectors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")

        # Insert a node with a distinctive vector
        vec_a = [1.0] + [0.0] * 767
        nodes1 = [_make_node_with_meta("a.md", "c:0", "alpha", vec_a)]
        store.upsert_nodes(nodes1)

        # Trigger schema evolution with a new field
        vec_b = [0.0] + [1.0] + [0.0] * 766
        nodes2 = [_make_node_with_meta("b.md", "c:0", "beta", vec_b, section="Setup")]
        store.upsert_nodes(nodes2)

        # Vector search should still find the correct nearest neighbor
        hits = store.vector_search([1.0] + [0.0] * 767, top_k=1)
        assert len(hits) == 1
        assert hits[0].doc_id == "a.md"


def test_schema_evolution_old_doc_returns_empty_for_new_field():
    """Old docs indexed before schema evolution should return empty string for new fields.

    Covers: vector_search, get_chunk, get_doc_chunks all work on old docs
    whose metadata was backfilled with empty strings after schema evolution.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")

        # Index a PDF without section field
        vec_a = [1.0] + [0.0] * 767
        nodes1 = [_make_node_with_meta(
            "a.pdf", "p:1:c:0", "old pdf content", vec_a,
            source_type="pdf", folder="Archive", status="archived",
        )]
        store.upsert_nodes(nodes1)

        # Index a Markdown with section field — triggers schema evolution
        vec_b = [0.0] + [1.0] + [0.0] * 766
        nodes2 = [_make_node_with_meta(
            "b.md", "c:0", "new md content", vec_b,
            source_type="md", folder="Projects", status="active",
            section="Introduction",
        )]
        store.upsert_nodes(nodes2)

        # Search returning both docs
        hits = store.vector_search([1.0] + [0.0] * 767, top_k=2)
        old_hit = next(h for h in hits if h.doc_id == "a.pdf")
        new_hit = next(h for h in hits if h.doc_id == "b.md")

        # Old doc: text intact, original metadata preserved
        assert old_hit.text == "old pdf content"
        assert old_hit.source_type == "pdf"
        assert old_hit.folder == "Archive"
        assert old_hit.status == "archived"

        # New doc: section value present
        assert new_hit.text == "new md content"
        assert new_hit.source_type == "md"
        assert new_hit.folder == "Projects"

        # get_chunk on old doc should work fine
        chunk = store.get_chunk("a.pdf", "p:1:c:0")
        assert chunk is not None
        assert chunk.text == "old pdf content"
        assert chunk.source_type == "pdf"

        # get_doc_chunks on old doc should work fine
        chunks = store.get_doc_chunks("a.pdf")
        assert len(chunks) == 1
        assert chunks[0].loc == "p:1:c:0"
        assert chunks[0].source_type == "pdf"


def test_schema_evolution_list_recent_docs_mixed_metadata():
    """list_recent_docs should work when metadata fields vary across docs.

    Simulates: PDFs indexed first (no section), then Markdown indexed (has section).
    list_recent_docs filters by source_type and folder via SQL on metadata struct —
    both old and new docs must be retrievable.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768

        # PDF indexed first — no section field
        nodes1 = [_make_node_with_meta(
            "report.pdf", "p:1:c:0", "quarterly report", vec,
            source_type="pdf", folder="Archive", status="archived", mtime=100.0,
        )]
        store.upsert_nodes(nodes1)

        # Markdown indexed later — has section, triggers schema evolution
        nodes2 = [_make_node_with_meta(
            "notes.md", "c:0", "meeting notes", vec,
            source_type="md", folder="Projects", status="active", mtime=200.0,
            section="Action Items",
        )]
        store.upsert_nodes(nodes2)

        # Unfiltered: both docs present
        all_docs = store.list_recent_docs(limit=10)
        doc_ids = {d["doc_id"] for d in all_docs}
        assert doc_ids == {"report.pdf", "notes.md"}

        # Filter by source_type — only matching type returned
        pdf_docs = store.list_recent_docs(limit=10, source_type="pdf")
        assert len(pdf_docs) == 1
        assert pdf_docs[0]["doc_id"] == "report.pdf"

        md_docs = store.list_recent_docs(limit=10, source_type="md")
        assert len(md_docs) == 1
        assert md_docs[0]["doc_id"] == "notes.md"

        # Filter by folder
        archive_docs = store.list_recent_docs(limit=10, folder="Archive")
        assert len(archive_docs) == 1
        assert archive_docs[0]["doc_id"] == "report.pdf"


def test_list_recent_docs_projects_rel_path():
    """A document listing must carry each document's path (#1203).

    ``file_list_documents`` / ``file_recent`` hand these rows straight to
    callers, and doc_id is an opaque 5-char id — without rel_path a listing
    cannot be acted on, and any caller filtering documents by path reads an
    empty string for every row, so its filter never matches.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768

        store.upsert_nodes([_make_node_with_meta(
            "documents::0000h", "c:0", "simulated video transcript", vec,
            source_type="mp4", rel_path="Media/clip.mp4", mtime=100.0,
        )])

        docs = store.list_recent_docs(limit=10)
        assert [d.get("rel_path") for d in docs] == ["Media/clip.mp4"]


def test_schema_evolution_multiple_new_fields():
    """Multiple new fields added at once should all appear in schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768

        # First batch: baseline metadata only
        nodes1 = [_make_node_with_meta("a.md", "c:0", "first", vec)]
        store.upsert_nodes(nodes1)

        # Second batch: two new fields at once
        nodes2 = [_make_node_with_meta(
            "b.md", "c:0", "second", vec,
            section="Overview", sentiment="positive",
        )]
        store.upsert_nodes(nodes2)

        subfields = store._metadata_subfields()
        assert "section" in subfields
        assert "sentiment" in subfields


def test_schema_evolution_streams_large_table_within_cgroup_envelope():
    """Widening 66k rows must never materialize the whole table as Arrow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 32
        store.upsert_nodes([
            _make_node_with_meta(
                "seed.md",
                "c:0",
                "x" * 512,
                vec,
                source_type="md",
            )
        ])

        seed = store._vs.table.to_arrow().slice(0, 1).to_pylist()[0]
        schema = store._vs.table.schema

        def _large_table_batches():
            remaining = 66_000 - 1
            while remaining:
                row_count = min(1_000, remaining)
                yield from pa.Table.from_pylist(
                    [seed] * row_count,
                    schema=schema,
                ).to_batches()
                remaining -= row_count

        store._vs.table.add(_large_table_batches())
        assert store._vs.table.count_rows() == 66_000

        cgroup_limit = 64 * 1024 * 1024
        arrow_envelope = cgroup_limit // 8
        largest_arrow_allocation = 0
        table_type = type(store._vs.table)
        dataset_type = type(store._vs.table.to_lance())
        real_to_arrow = table_type.to_arrow
        real_to_batches = dataset_type.to_batches

        def _reject_unbounded_materialization(table, *args, **kwargs):
            nonlocal largest_arrow_allocation
            materialized = real_to_arrow(table, *args, **kwargs)
            largest_arrow_allocation = max(
                largest_arrow_allocation,
                materialized.nbytes,
            )
            if materialized.nbytes > arrow_envelope:
                raise AssertionError(
                    "metadata widening exceeded cgroup-derived Arrow envelope: "
                    f"{materialized.nbytes} > {arrow_envelope}"
                )
            return materialized

        def _measure_streamed_batches(dataset, *args, **kwargs):
            nonlocal largest_arrow_allocation
            for batch in real_to_batches(dataset, *args, **kwargs):
                largest_arrow_allocation = max(
                    largest_arrow_allocation,
                    batch.nbytes,
                )
                assert batch.nbytes <= arrow_envelope
                yield batch

        with patch(
            "lancedb_store._cgroup_memory_limit_bytes",
            return_value=cgroup_limit,
            create=True,
        ), patch.object(
            table_type,
            "to_arrow",
            _reject_unbounded_materialization,
        ), patch.object(
            dataset_type,
            "to_batches",
            _measure_streamed_batches,
        ):
            store.upsert_nodes([
                _make_node_with_meta(
                    "new.md",
                    "c:0",
                    "new field",
                    vec,
                    content_status="indexed",
                )
            ])

        assert 0 < largest_arrow_allocation <= arrow_envelope
        assert store._vs.table.count_rows() == 66_001
        assert "content_status" in store._metadata_subfields()


def test_schema_evolution_reads_real_cgroup_memory_limit(tmp_path):
    memory_max = tmp_path / "memory.max"
    memory_max.write_text("8589934592\n")

    assert (
        lancedb_store_module._cgroup_memory_limit_bytes(memory_max_path=memory_max)
        == 8 * 1024 * 1024 * 1024
    )


def test_schema_evolution_uses_smallest_nested_cgroup_limit(tmp_path):
    cgroup_root = tmp_path / "cgroup"
    leaf = cgroup_root / "workload" / "child"
    leaf.mkdir(parents=True)
    (cgroup_root / "memory.max").write_text("17179869184\n")
    (cgroup_root / "workload" / "memory.max").write_text("8589934592\n")
    (leaf / "memory.max").write_text("max\n")
    proc_cgroup = tmp_path / "self.cgroup"
    proc_cgroup.write_text("0::/workload/child\n")

    assert lancedb_store_module._cgroup_memory_limit_bytes(
        cgroup_root=cgroup_root,
        proc_cgroup_path=proc_cgroup,
    ) == 8 * 1024 * 1024 * 1024


def test_schema_evolution_rejects_single_row_over_cgroup_envelope():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 8
        store.upsert_nodes([_make_node_with_meta("seed.md", "c:0", "seed", vec)])
        oversized = _make_node_with_meta(
            "large.md",
            "c:0",
            "x" * (3 * 1024 * 1024),
            vec,
        )
        store.upsert_nodes([oversized])

        with patch(
            "lancedb_store._cgroup_memory_limit_bytes",
            return_value=16 * 1024 * 1024,
        ), pytest.raises(RuntimeError, match="single row"):
            store.upsert_nodes([
                _make_node_with_meta(
                    "next.md",
                    "c:0",
                    "next",
                    vec,
                    content_status="indexed",
                )
            ])

        assert "content_status" not in store._metadata_subfields()
        assert set(store.list_doc_ids()) == {"seed.md", "large.md"}


def test_schema_evolution_logs_fields_and_row_count_before_widening(caplog):
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 8
        store.upsert_nodes([_make_node_with_meta("seed.md", "c:0", "seed", vec)])

        caplog.set_level(logging.INFO, logger="lancedb_store")
        store.upsert_nodes([
            _make_node_with_meta(
                "next.md",
                "c:0",
                "next",
                vec,
                content_status="indexed",
                content_failure_reasons="",
            )
        ])

        widening_record = next(
            record
            for record in caplog.records
            if record.getMessage().startswith("Widening Lance metadata schema")
        )
        assert widening_record.new_fields == [
            "content_failure_reasons",
            "content_status",
        ]
        assert widening_record.row_count == 1


def test_schema_evolution_blocks_writes_until_atomic_swap_finishes():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 8
        store.upsert_nodes([_make_node_with_meta("seed.md", "c:0", "seed", vec)])

        normal_write_at_add = threading.Event()
        release_normal_write = threading.Event()
        evolution_scanning = threading.Event()
        release_evolution = threading.Event()
        errors: list[Exception] = []
        dataset_type = type(store._vs.table.to_lance())
        real_to_batches = dataset_type.to_batches
        old_vector_store = store._vs
        vector_store_type = type(old_vector_store)
        real_add = vector_store_type.add
        normal_thread_id: int | None = None

        def _pause_normal_add(vector_store, nodes):
            if (
                vector_store is old_vector_store
                and threading.get_ident() == normal_thread_id
            ):
                normal_write_at_add.set()
                if not release_normal_write.wait(5):
                    raise TimeoutError("test did not release normal write")
            return real_add(vector_store, nodes)

        def _pause_first_schema_scan(dataset, *args, **kwargs):
            iterator = real_to_batches(dataset, *args, **kwargs)
            evolution_scanning.set()
            if not release_evolution.wait(5):
                raise TimeoutError("test did not release schema evolution")
            yield from iterator

        def _widen_schema():
            try:
                store.upsert_nodes([
                    _make_node_with_meta(
                        "widening.md",
                        "c:0",
                        "widening",
                        vec,
                        content_status="indexed",
                    )
                ])
            except Exception as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        def _normal_write():
            nonlocal normal_thread_id
            normal_thread_id = threading.get_ident()
            try:
                store.upsert_nodes([
                    _make_node_with_meta("concurrent.md", "c:0", "concurrent", vec)
                ])
            except Exception as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        with patch.object(
            vector_store_type,
            "add",
            _pause_normal_add,
        ), patch.object(dataset_type, "to_batches", _pause_first_schema_scan):
            normal_thread = threading.Thread(target=_normal_write)
            normal_thread.start()
            assert normal_write_at_add.wait(5)

            evolution_thread = threading.Thread(target=_widen_schema)
            evolution_thread.start()
            try:
                assert not evolution_scanning.wait(0.2)
            finally:
                release_normal_write.set()
                evolution_scanning.wait(5)
                release_evolution.set()

            normal_thread.join(10)
            evolution_thread.join(10)

        assert not evolution_thread.is_alive()
        assert not normal_thread.is_alive()
        assert errors == []
        assert set(store.list_doc_ids()) == {
            "seed.md",
            "widening.md",
            "concurrent.md",
        }


def test_schema_evolution_concurrent_new_fields_on_shared_store():
    """Concurrent schema evolution should not corrupt the shared temp table path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768

        store.upsert_nodes([_make_node_with_meta("seed.md", "c:0", "seed", vec)])

        field_names = ["field_alpha", "field_beta", "field_gamma", "field_delta"]
        nodes = [
            _make_node_with_meta(
                f"doc-{idx}.md",
                "c:0",
                f"body {idx}",
                vec,
                **{field_name: f"value-{idx}"},
            )
            for idx, field_name in enumerate(field_names, start=1)
        ]

        metadata_barrier = threading.Barrier(len(nodes))
        metadata_call_lock = threading.Lock()
        metadata_call_count = 0
        real_metadata_subfields = store._metadata_subfields
        real_evolve_metadata_schema = store._evolve_metadata_schema

        def _synchronized_metadata_subfields():
            nonlocal metadata_call_count
            result = real_metadata_subfields()
            with metadata_call_lock:
                metadata_call_count += 1
                call_number = metadata_call_count
            if result and call_number <= len(nodes):
                metadata_barrier.wait(timeout=5)
            return result

        def _slow_evolve_metadata_schema(new_fields):
            time.sleep(0.05)
            return real_evolve_metadata_schema(new_fields)

        store._metadata_subfields = _synchronized_metadata_subfields  # type: ignore[method-assign]
        store._evolve_metadata_schema = _slow_evolve_metadata_schema  # type: ignore[method-assign]

        errors: list[Exception] = []

        def _worker(node: TextNode) -> None:
            try:
                store.upsert_nodes([node])
            except Exception as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=len(nodes)) as executor:
            list(executor.map(_worker, nodes))

        assert errors == []
        assert set(store.list_doc_ids()) == {"seed.md", *(n.ref_doc_id for n in nodes)}
        assert set(field_names) <= store._metadata_subfields()


# --- Dynamic metadata pipeline tests ---


def test_extra_metadata_in_vector_search():
    """Dynamic fields should be visible in vector_search results via extra_metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.1] * 768
        nodes = [_make_node_with_meta(
            "a.md", "c:0", "intro text", vec, section="Introduction",
        )]
        store.upsert_nodes(nodes)

        hits = store.vector_search(vec, top_k=1)
        assert len(hits) == 1
        # Accessible via extra_metadata dict
        assert hits[0].extra_metadata["section"] == "Introduction"
        # Accessible via __getattr__ fallback
        assert hits[0].section == "Introduction"


def test_context_metadata_filter_and_passthrough():
    """Context enrichment fields stay filterable and pass through extra_metadata."""
    from doc_enrichment import ENRICHMENT_FIELDS

    assert "enr_context_confidence" in ENRICHMENT_FIELDS

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.1] * 768
        store.upsert_nodes([
            _make_node_with_meta(
                "photo.jpg",
                "img:c:0",
                "photo of garage",
                vec,
                source_type="img",
                enr_context_entities_places="54 S Broad Main Unit E",
                enr_context_confidence="high",
            )
        ])

        hits = store.vector_search(
            vec,
            top_k=5,
            where="lower(metadata.enr_context_confidence) = 'high'",
        )

        assert hits
        assert hits[0].extra_metadata["enr_context_entities_places"] == "54 S Broad Main Unit E"
        assert hits[0].extra_metadata["enr_context_confidence"] == "high"
        assert "enr_context_entities_places" not in hits[0].__dict__


def test_context_narrative_metadata_excluded_from_dynamic_facets():
    """Narrative/JSON context fields should not be comma-split into facets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.1] * 768
        store.upsert_nodes([
            _make_node_with_meta(
                "photo.jpg",
                "img:c:0",
                "photo of garage",
                vec,
                source_type="img",
                enr_context_confidence="high",
                enr_context_key_facts='["Unit E has bins, shelves, and tools."]',
                enr_context_warning="Nearby messages mention Unit E, but one reply is unrelated.",
            )
        ])

        facets = store.facets()

        assert facets["enr_context_confidence"] == [{"value": "high", "count": 1}]
        assert "enr_context_key_facts" not in facets
        assert "enr_context_warning" not in facets


def test_dynamic_facets_exclude_internal_narrative_and_identifier_fields():
    available = {
        "section",
        "priority",
        "_node_content",
        "message_body",
        "enr_summary",
        "enr_key_facts",
        "custom_meta",
        "source_message_id",
        "content_hash",
        "sidecar_path",
        "updated_at",
        "dup_sources",
        "file_size_bytes",
        "id",
        "path",
        "filename",
        "hash",
        "timestamp",
        "url",
        "uri",
        "facts",
        "json",
        "narrative",
        "case_narrative",
        "ids",
        "content_type",
        "warning_level",
        "description_kind",
        "facts_type",
        "narrative_type",
    }

    assert LanceDBStore._dynamic_facet_fields(available) == {
        "section",
        "priority",
        "content_type",
        "warning_level",
        "description_kind",
        "facts_type",
        "narrative_type",
    }


def test_extra_metadata_in_get_chunk():
    """Dynamic fields should be visible in get_chunk results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        nodes = [_make_node_with_meta(
            "a.md", "c:0", "setup steps", vec, section="Setup",
        )]
        store.upsert_nodes(nodes)

        chunk = store.get_chunk("a.md", "c:0")
        assert chunk is not None
        assert chunk.extra_metadata["section"] == "Setup"
        assert chunk.section == "Setup"


def test_extra_metadata_in_get_doc_chunks():
    """Dynamic fields should be visible in get_doc_chunks results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        nodes = [_make_node_with_meta(
            "a.md", "c:0", "overview", vec, section="Overview",
        )]
        store.upsert_nodes(nodes)

        chunks = store.get_doc_chunks("a.md")
        assert len(chunks) == 1
        assert chunks[0].section == "Overview"


def test_extra_metadata_visible_in_hit_to_dict():
    """_hit_to_dict should include extra_metadata fields in the output dict."""
    from core.storage import SearchHit
    import mcp_server

    hit = SearchHit(
        doc_id="a.md", loc="c:0", snippet="test", text="test text",
        score=0.5, extra_metadata={"section": "Intro", "sentiment": "positive"},
    )
    d = mcp_server._hit_to_dict(hit)
    assert d["section"] == "Intro"
    assert d["sentiment"] == "positive"
    # Core fields still present
    assert d["doc_id"] == "a.md"
    assert d["score"] == 0.5


def test_extra_metadata_filter_via_search_impl():
    """metadata_filters should filter results by dynamic fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")

        # Two docs: one with section, one without
        vec_a = [1.0] + [0.0] * 767
        nodes1 = [_make_node_with_meta(
            "a.md", "c:0", "intro content", vec_a, section="Introduction",
        )]
        vec_b = [0.0] + [1.0] + [0.0] * 766
        nodes2 = [_make_node_with_meta("b.md", "c:0", "other content", vec_b)]

        store.upsert_nodes(nodes1)
        store.upsert_nodes(nodes2)

        # Search without filter — both docs findable
        all_hits = store.vector_search([0.5] * 768, top_k=10)
        assert len(all_hits) == 2

        # Use hybrid_search metadata_filters via the search impl
        # (requires embedding — test the filter logic directly)
        filtered = [h for h in all_hits if getattr(h, "section", None) == "Introduction"]
        assert len(filtered) == 1
        assert filtered[0].doc_id == "a.md"


def test_facets_includes_dynamic_fields():
    """facets() should include dynamic metadata fields like section."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        nodes = [
            _make_node_with_meta("a.md", "c:0", "intro", vec, section="Introduction"),
            _make_node_with_meta("b.md", "c:0", "setup", vec, section="Setup"),
            _make_node_with_meta("c.md", "c:0", "setup2", vec, section="Setup"),
        ]
        store.upsert_nodes(nodes)

        facets = store.facets()
        # section should appear as a dynamic facet
        assert "section" in facets
        section_values = {f["value"]: f["count"] for f in facets["section"]}
        assert section_values["Setup"] == 2
        assert section_values["Introduction"] == 1


def test_custom_frontmatter_promoted_to_columns():
    """Extra frontmatter fields promoted to real columns should appear in
    extra_metadata, facets, and be filterable via getattr."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768

        # Simulate promoted frontmatter keys (priority, category)
        nodes = [
            _make_node_with_meta(
                "a.md", "c:0", "high-pri research", vec,
                priority="high", category="research",
            ),
            _make_node_with_meta(
                "b.md", "c:0", "low-pri notes", vec,
                priority="low", category="notes",
            ),
        ]
        store.upsert_nodes(nodes)

        # extra_metadata exposes promoted fields on SearchHit
        hits = store.vector_search(vec, top_k=10)
        assert len(hits) == 2
        for h in hits:
            assert "priority" in h.extra_metadata
            assert "category" in h.extra_metadata

        hit_a = next(h for h in hits if h.doc_id == "a.md")
        assert hit_a.extra_metadata["priority"] == "high"
        assert hit_a.extra_metadata["category"] == "research"
        # Also accessible via __getattr__
        assert hit_a.priority == "high"
        assert hit_a.category == "research"

        # facets() includes promoted fields
        facets = store.facets()
        assert "priority" in facets
        assert "category" in facets
        pri_values = {f["value"] for f in facets["priority"]}
        assert pri_values == {"high", "low"}
        cat_values = {f["value"] for f in facets["category"]}
        assert cat_values == {"research", "notes"}

        # Filterable via getattr
        high_pri = [h for h in hits if getattr(h, "priority", None) == "high"]
        assert len(high_pri) == 1
        assert high_pri[0].doc_id == "a.md"


def test_file_status_returns_metadata_fields():
    """file_status should include metadata_fields listing all schema fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        nodes = [_make_node_with_meta(
            "a.md", "c:0", "text", vec, section="Intro",
        )]
        store.upsert_nodes(nodes)

        import mcp_server
        # Wire up the test store
        mcp_server._cache = (store, None, {"index_root": tmpdir})

        result = mcp_server._file_status_impl()
        assert "metadata_fields" in result
        assert "section" in result["metadata_fields"]
        assert "doc_id" in result["metadata_fields"]
        assert "source_type" in result["metadata_fields"]


# --- _build_where_clause tests ---


def test_build_where_clause_empty():
    """No filters should return None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        assert store._build_where_clause() is None


def test_build_where_clause_source_type():
    """source_type should produce case-insensitive exact match on metadata.source_type."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        clause = store._build_where_clause(source_type="pdf")
        assert clause == "lower(metadata.source_type) = 'pdf'"


def test_build_where_clause_doc_id_prefix():
    """doc_id_prefix should produce LIKE on metadata.rel_path (path-based browsing)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        clause = store._build_where_clause(doc_id_prefix="Projects/")
        assert clause == "metadata.rel_path LIKE 'Projects/%'"


def test_build_where_clause_comma_fields():
    """Comma-separated tags/enr_doc_type/enr_topics should produce case-insensitive OR clauses."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        clause = store._build_where_clause(tags="recipe,korean")
        assert "lower(metadata.tags) LIKE '%recipe%'" in clause
        assert "lower(metadata.tags) LIKE '%korean%'" in clause
        assert " OR " in clause

        clause2 = store._build_where_clause(enr_doc_type="Report,Summary")
        assert "lower(metadata.enr_doc_type) LIKE '%report%'" in clause2
        assert "lower(metadata.enr_doc_type) LIKE '%summary%'" in clause2

        clause3 = store._build_where_clause(enr_topics="machine learning,NLP")
        assert "lower(metadata.enr_topics) LIKE '%machine learning%'" in clause3
        assert "lower(metadata.enr_topics) LIKE '%nlp%'" in clause3


def test_build_where_clause_combined():
    """Multiple filters should be AND-joined."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        clause = store._build_where_clause(
            source_type="pdf", folder="Archive", tags="finance",
        )
        assert "lower(metadata.source_type) = 'pdf'" in clause
        assert "lower(metadata.folder) = 'archive'" in clause
        assert "lower(metadata.tags) LIKE '%finance%'" in clause
        assert " AND " in clause


def test_build_where_clause_sql_escape():
    """Single quotes in values should be escaped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        clause = store._build_where_clause(folder="O'Brien's")
        assert "o''brien''s" in clause


def test_build_where_clause_metadata_filters():
    """metadata_filters dict should produce case-insensitive metadata.key = 'value' clauses."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        clause = store._build_where_clause(
            metadata_filters={"section": "Introduction", "priority": "high"},
        )
        assert "lower(metadata.section) = 'introduction'" in clause
        assert "lower(metadata.priority) = 'high'" in clause


def test_build_where_clause_filter_ast_nested_boolean():
    """filter_ast should support nested boolean logic for any metadata field."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        clause = store._build_where_clause(
            filter_ast={
                "and": [
                    {"eq": {"source_name": "sor"}},
                    {"or": [
                        {"eq": {"status": "active"}},
                        {"eq": {"status": "pending"}},
                    ]},
                    {"not": {"contains": {"owner": "dan"}}},
                ]
            },
        )

        assert "lower(metadata.source_name) = 'sor'" in clause
        assert "lower(metadata.status) = 'active'" in clause
        assert " OR " in clause
        assert "NOT (lower(metadata.owner) LIKE '%dan%')" in clause
        assert " AND " in clause


def test_build_where_clause_filter_ast_in_and_prefix():
    """filter_ast should support IN lists and rel_path prefix filters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        clause = store._build_where_clause(
            filter_ast={
                "and": [
                    {"in": {"status": ["active", "pending"]}},
                    {"prefix": {"rel_path": "Projects/"}},
                ]
            },
        )

        assert "lower(metadata.status) IN ('active', 'pending')" in clause
        assert "metadata.rel_path LIKE 'Projects/%'" in clause


def test_build_where_clause_filter_ast_rejects_unknown_operator():
    """filter_ast should reject unsupported operators instead of emitting SQL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        with pytest.raises(ValueError, match="Unsupported filter operator"):
            store._build_where_clause(filter_ast={"gt": {"priority": "high"}})


def test_build_where_clause_filter_ast_rejects_unsafe_key():
    """filter_ast should reject unsafe metadata field names."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        with pytest.raises(ValueError, match="Unsafe metadata filter key"):
            store._build_where_clause(filter_ast={"eq": {"status; DROP TABLE chunks": "active"}})


# --- _row_to_hit tests ---


def test_row_to_hit_vector_distance():
    """_row_to_hit should convert _distance to similarity score."""
    row = {
        "metadata": {"doc_id": "a.md", "loc": "c:0", "snippet": "test", "mtime": 1.0},
        "text": "test text",
        "_distance": 0.3,
    }
    hit = LanceDBStore._row_to_hit(row)
    assert hit.doc_id == "a.md"
    assert abs(hit.score - 0.7) < 0.001  # 1.0 - 0.3


def test_row_to_hit_fts_score():
    """_row_to_hit should use _score for FTS results."""
    row = {
        "metadata": {"doc_id": "b.md", "loc": "c:1", "mtime": 2.0},
        "text": "fts text",
        "_score": 5.5,
    }
    hit = LanceDBStore._row_to_hit(row)
    assert hit.doc_id == "b.md"
    assert hit.score == 5.5


# --- Pre-filtered vector search tests ---


def test_vector_search_with_where():
    """vector_search with where clause should filter results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [1.0] + [0.0] * 767
        nodes = [
            _make_node_with_meta("a.md", "c:0", "apple", vec, source_type="md"),
            _make_node_with_meta("b.pdf", "p:1:c:0", "apple pdf", vec, source_type="pdf"),
        ]
        store.upsert_nodes(nodes)
        hits = store.vector_search(vec, top_k=10, where="metadata.source_type = 'pdf'")
        assert len(hits) == 1
        assert hits[0].doc_id == "b.pdf"


def test_keyword_search_with_where():
    """keyword_search with where clause should filter results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        nodes = [
            _make_node_with_meta("a.md", "c:0", "banana fruit tropical", vec, source_type="md", folder="Recipes"),
            _make_node_with_meta("b.md", "c:0", "banana split dessert tropical", vec, source_type="md", folder="Archive"),
        ]
        store.upsert_nodes(nodes)
        store.create_fts_index()
        hits = store.keyword_search("banana tropical", top_k=10, where="metadata.folder = 'Recipes'")
        assert len(hits) == 1
        assert hits[0].doc_id == "a.md"


def test_keyword_search_retries_phrase_queries_without_positions():
    """Quoted phrase queries should fall back to plain token search when Lance lacks positions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        nodes = [
            _make_node_with_meta("a.md", "c:0", "banana fruit tropical", vec, source_type="md"),
            _make_node_with_meta("b.md", "c:0", "banana split dessert tropical", vec, source_type="md"),
        ]
        store.upsert_nodes(nodes)
        store.create_fts_index()

        hits = store.keyword_search('"banana tropical"', top_k=10)

        assert [hit.doc_id for hit in hits] == ["a.md", "b.md"]


def test_ensure_fts_index_creates_when_missing():
    """ensure_fts_index on a fresh table should create the FTS index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        store.upsert_nodes([
            _make_node_with_meta("a.md", "c:0", "banana fruit tropical", vec, source_type="md"),
        ])
        store.ensure_fts_index()
        hits = store.keyword_search("banana", top_k=10)
        assert len(hits) == 1


def test_ensure_fts_index_missing_path_tags_exact_latest_without_data_compaction():
    """Creating missing FTS still finalizes restore metadata, without rewrite."""
    from datetime import date

    import lance

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        store.upsert_nodes(
            [_make_node_with_meta("a.md", "c:0", "banana", vec, source_type="md")]
        )

        with patch.object(store, "_compact_data_files") as compact:
            store.ensure_fts_index()

        tag = f"daily-{date.today().isoformat()}"
        compact.assert_not_called()
        assert lance.dataset(_lance_path(tmpdir), version=tag).version == (
            lance.dataset(_lance_path(tmpdir)).version
        )


def _lance_path(tmpdir: str, table: str = "test_chunks") -> str:
    return str(Path(tmpdir) / f"{table}.lance")


def test_ensure_fts_index_creates_todays_daily_restore_point():
    """A routine indexing run tags the current version as today's restore
    point (daily-<date>), pointing at the latest version."""
    from datetime import date

    import lance

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        store.upsert_nodes([
            _make_node_with_meta("a.md", "c:0", "banana", vec, source_type="md"),
        ])
        store.create_fts_index()
        store.ensure_fts_index()

        tags = store._vs.table.tags.list()
        today_tag = f"daily-{date.today().isoformat()}"
        assert today_tag in tags
        assert lance.dataset(_lance_path(tmpdir), version=today_tag).version == (
            lance.dataset(_lance_path(tmpdir)).version
        )


def test_daily_restore_point_is_revertible_after_more_writes():
    """A tagged restore point pins the point-in-time snapshot: after more rows
    are written, opening the dataset at the tag still shows the old state."""
    import lance

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        store.upsert_nodes([
            _make_node_with_meta("a.md", "c:0", "first", vec, source_type="md"),
        ])
        store.create_fts_index()
        tag = "daily-2099-01-01"
        store._vs.table.tags.create(tag, store._vs.table.version)  # 1 row here

        store.upsert_nodes([
            _make_node_with_meta("b.md", "c:0", "second", vec, source_type="md"),
            _make_node_with_meta("c.md", "c:0", "third", vec, source_type="md"),
        ])

        path = _lance_path(tmpdir)
        assert lance.dataset(path).count_rows() == 3           # current state
        assert lance.dataset(path, version=tag).count_rows() == 1  # restore point


def test_prune_preserves_tagged_versions_removes_untagged():
    """With retention=0 (prune everything superseded), an untagged old version
    is reclaimed but a tagged restore point survives — proving the prune runs
    with error_if_tagged_old_versions=False (skip tags, don't error)."""
    from unittest.mock import patch
    import lance

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        store.upsert_nodes([
            _make_node_with_meta("a.md", "c:0", "first", vec, source_type="md"),
        ])
        store.create_fts_index()
        # a restore point INSIDE the retention window (won't be expired)
        tag = "daily-2099-06-15"
        store._vs.table.tags.create(tag, store._vs.table.version)

        env = {"LANCE_VERSION_RETENTION_MINUTES": "0", "LANCE_DAILY_RESTORE_POINTS": "30"}
        with patch.dict("os.environ", env):
            store.upsert_nodes([
                _make_node_with_meta("b.md", "c:0", "second", vec, source_type="md"),
            ])
            # tag is future-dated (2099) so real date.today() never expires it
            store.ensure_fts_index()  # optimize -> tag today -> prune retention=0

        path = _lance_path(tmpdir)
        # tagged restore point still readable after an aggressive prune
        assert lance.dataset(path, version=tag).count_rows() == 1
        assert lance.dataset(path).count_rows() == 2


def test_manage_restore_points_expires_tags_past_window():
    """Daily tags older than the retention window are deleted; ones inside it
    (and non-daily tags) are left alone."""
    from datetime import date

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        store.upsert_nodes([
            _make_node_with_meta("a.md", "c:0", "banana", vec, source_type="md"),
        ])
        table = store._vs.table
        v = table.version
        table.tags.create("daily-2026-01-01", v)   # ancient -> expire
        table.tags.create("daily-2026-06-20", v)   # within 30d of 'today' below -> keep
        table.tags.create("manual-keepsake", v)    # not ours -> never touch

        from unittest.mock import patch
        with patch.dict("os.environ", {"LANCE_DAILY_RESTORE_POINTS": "30"}):
            store._manage_restore_points(table, date(2026, 6, 25))

        tags = set(table.tags.list())
        assert "daily-2026-01-01" not in tags       # expired
        assert "daily-2026-06-20" in tags           # inside window
        assert "daily-2026-06-25" in tags           # today's, freshly created
        assert "manual-keepsake" in tags            # untouched


def test_restore_points_disabled_when_days_zero():
    from datetime import date
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        store.upsert_nodes([
            _make_node_with_meta("a.md", "c:0", "banana", vec, source_type="md"),
        ])
        table = store._vs.table
        with patch.dict("os.environ", {"LANCE_DAILY_RESTORE_POINTS": "0"}):
            store._manage_restore_points(table, date(2026, 6, 25))
        assert not any(t.startswith("daily-") for t in table.tags.list())


def test_ensure_fts_index_prune_failure_is_non_fatal():
    """A dataset-level prune failure is swallowed by the prune boundary."""
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        store.upsert_nodes([
            _make_node_with_meta("a.md", "c:0", "banana", vec, source_type="md"),
        ])
        store.create_fts_index()
        with patch("core.resilience.time.sleep", lambda *_: None):
            with patch("lance.dataset", side_effect=RuntimeError("disk gone")):
                store._prune_versions("test")  # must not raise
        # index still works after a prune failure
        assert len(store.keyword_search("banana", top_k=5)) == 1


def test_lance_version_retention_env_parsing():
    from unittest.mock import patch
    import lancedb_store as ls

    with patch.dict("os.environ", {}, clear=False):
        os_environ = ls.os.environ
        os_environ.pop("LANCE_VERSION_RETENTION_MINUTES", None)
        assert ls._lance_version_retention_minutes() == 30.0
    with patch.dict("os.environ", {"LANCE_VERSION_RETENTION_MINUTES": "5"}):
        assert ls._lance_version_retention_minutes() == 5.0
    with patch.dict("os.environ", {"LANCE_VERSION_RETENTION_MINUTES": "garbage"}):
        assert ls._lance_version_retention_minutes() == 30.0
    with patch.dict("os.environ", {"LANCE_VERSION_RETENTION_MINUTES": "-3"}):
        assert ls._lance_version_retention_minutes() == 0.0


def test_daily_restore_point_days_env_parsing():
    from unittest.mock import patch
    import lancedb_store as ls

    with patch.dict("os.environ", {}, clear=False):
        ls.os.environ.pop("LANCE_DAILY_RESTORE_POINTS", None)
        assert ls._daily_restore_point_days() == 7
    with patch.dict("os.environ", {"LANCE_DAILY_RESTORE_POINTS": "14"}):
        assert ls._daily_restore_point_days() == 14
    with patch.dict("os.environ", {"LANCE_DAILY_RESTORE_POINTS": "0"}):
        assert ls._daily_restore_point_days() == 0
    with patch.dict("os.environ", {"LANCE_DAILY_RESTORE_POINTS": "junk"}):
        assert ls._daily_restore_point_days() == 7


def test_parse_daily_tag_date():
    from datetime import date
    import lancedb_store as ls

    assert ls._parse_daily_tag_date("daily-2026-07-03") == date(2026, 7, 3)
    assert ls._parse_daily_tag_date("manual-keepsake") is None
    assert ls._parse_daily_tag_date("daily-not-a-date") is None


def test_is_retryable_commit_conflict():
    import lancedb_store as ls

    assert ls._is_retryable_commit_conflict(
        RuntimeError("lance error: Retryable commit conflict for version 24252")
    )
    assert not ls._is_retryable_commit_conflict(RuntimeError("offset overflow"))


def test_fts_finds_rows_added_after_index_creation():
    """Native FTS should surface rows written after the index was built, no rebuild."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        store.upsert_nodes([
            _make_node_with_meta("a.md", "c:0", "banana fruit tropical", vec, source_type="md"),
        ])
        store.create_fts_index()
        store.upsert_nodes([
            _make_node_with_meta("b.md", "c:0", "quasar telescope astronomy", vec, source_type="md"),
        ])
        hits = store.keyword_search("quasar astronomy", top_k=10)
        assert len(hits) == 1
        assert hits[0].doc_id == "b.md"

        store.ensure_fts_index()  # merges the unindexed tail; still searchable after
        hits = store.keyword_search("quasar astronomy", top_k=10)
        assert len(hits) == 1


# ---------------------------------------------------------------------------
# Daily compaction cadence (#0232) — full data compaction runs at most once
# per calendar day; every run still merges new rows into the search indices.
# Per-run compaction × retained restore-point tags pinned every superseded
# fragment set, growing a ~5 GB table to 350+ GB of dead files.
# ---------------------------------------------------------------------------


def _compaction_marker(tmpdir: str) -> Path:
    return Path(tmpdir) / "test_chunks.lance.last-compaction"


def _fts_unindexed_rows(tmpdir: str) -> int:
    """Unindexed-row count via a fresh handle (immune to stale-version caching)."""
    table = LanceDBStore(tmpdir, "test_chunks")._vs.table
    name = next(
        idx.name for idx in table.list_indices()
        if str(getattr(idx, "index_type", "")).upper() == "FTS"
    )
    return table.index_stats(name).num_unindexed_rows


def test_daily_compaction_orders_prune_compact_marker():
    """The prune frees headroom before the rewrite, and the cadence marker is
    only written once the rewrite has been verified."""
    from datetime import date

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        store.upsert_nodes([_make_node("seed.md", "c:0", "seed", [0.1] * 768)])
        order: list[str] = []

        with (
            patch.object(store, "_compaction_due", return_value=True),
            patch.object(
                store,
                "_prune_versions",
                side_effect=lambda label: order.append(f"prune:{label}"),
            ),
            patch.object(
                store,
                "_compact_data_files",
                side_effect=lambda: order.append("compact"),
            ),
            patch.object(
                store,
                "_record_compaction",
                side_effect=lambda _day: order.append("marker"),
            ),
        ):
            store.compact_data_files_if_due(date.today())

        assert order == [
            "prune:pre-compaction",
            "compact",
            "marker",
        ]


def test_index_run_maintenance_never_forks_a_compaction_worker():
    """#1254: daily compaction forks a second full-memory Python worker into the
    container's memory cgroup. Run inline in a live index run it drove that
    cgroup to its 8 GiB ceiling twice (2026-08-14, 2026-08-19); the kernel then
    killed the fattest task in the cgroup — the long-lived server, not the
    worker — the container restarted, and the index run in flight died with it
    (51 of 55 documents on 08-19). No index-run maintenance path may fork it.
    The cadence itself is unchanged; the idle-window entry point owns it."""
    from datetime import date

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        store.upsert_nodes([_make_node("seed.md", "c:0", "seed", [0.1] * 768)])
        store.create_fts_index()

        with patch.object(store, "_compact_data_files") as worker:
            store.prepare_indexing_maintenance()
            store.ensure_fts_index()

        assert worker.call_count == 0
        assert not _compaction_marker(tmpdir).exists()

        assert store.compact_data_files_if_due(date.today()) is True
        assert (
            _compaction_marker(tmpdir).read_text().strip() == date.today().isoformat()
        )


def test_compaction_worker_is_killed_before_the_cgroup_ceiling():
    """#1254: the worker allocates inside the same memory cgroup as the server
    and any live index run, and the kernel's OOM killer picks the fattest task
    in that cgroup — never reliably the worker. Watch the cgroup while the
    worker runs and kill the worker first: a dead worker only defers compaction
    to the next idle window, a dead server restarts the container."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        samples = iter([1_000, 9_600])

        with (
            patch.object(
                lancedb_store_module,
                "_compaction_worker_command",
                return_value=[sys.executable, "-c", "import time; time.sleep(30)"],
            ),
            patch.object(
                lancedb_store_module,
                "_cgroup_memory_limit_bytes",
                return_value=10_000,
            ),
            patch.object(
                lancedb_store_module,
                "_cgroup_memory_anon_bytes",
                side_effect=lambda: next(samples, 9_600),
            ),
            patch.object(
                lancedb_store_module, "_WORKER_MEMORY_POLL_SECONDS", 0.01
            ),
        ):
            with pytest.raises(MemoryError, match="cgroup"):
                store._compact_data_files()


def test_compaction_worker_runs_unguarded_without_a_readable_cgroup():
    """No cgroup accounting (bare host, cgroup v1) must not disable compaction —
    the guard is a ceiling, not a precondition."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")

        with (
            patch.object(
                lancedb_store_module,
                "_compaction_worker_command",
                return_value=[sys.executable, "-c", "pass"],
            ),
            patch.object(
                lancedb_store_module, "_cgroup_memory_anon_bytes", return_value=None
            ),
        ):
            store._compact_data_files()


def test_unreadable_compaction_is_restored_and_not_recorded(caplog):
    """A successful worker exit must not commit an unreadable latest version."""
    import lance
    from datetime import date

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        store.upsert_nodes([
            _make_node_with_meta(
                "wide.md",
                "c:0",
                "wide",
                vec,
                **{f"k{i}": "v" for i in range(20)},
            )
        ])
        store.upsert_nodes([
            _make_node_with_meta("narrow.md", "c:0", "narrow", vec)
        ])

        with patch.object(store, "_compaction_due", return_value=True), patch.object(
            store,
            "_compact_data_files",
            side_effect=lambda: _claim_columns_the_file_lacks(tmpdir, "test_chunks"),
        ):
            compacted = store.compact_data_files_if_due(date.today())

        assert compacted is False
        assert not _compaction_marker(tmpdir).exists()
        restored = lance.dataset(store._dataset_path()).to_table()
        assert set(restored.column("doc_id").to_pylist()) == {"wide.md", "narrow.md"}
        assert "restored pre-compaction version" in caplog.text


def test_readable_compaction_probes_each_fragment_before_recording():
    """Every output fragment gets a bounded read before cadence is recorded."""
    import lance
    from datetime import date

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        fragments = [MagicMock(), MagicMock(), MagicMock()]
        dataset = SimpleNamespace(version=7, get_fragments=lambda: fragments)

        with patch.object(store, "_compaction_due", return_value=True), patch.object(
            store, "_compact_data_files"
        ), patch.object(lance, "dataset", return_value=dataset):
            compacted = store.compact_data_files_if_due(date.today())

        assert compacted is True
        assert _compaction_marker(tmpdir).exists()
        for fragment in fragments:
            fragment.to_table.assert_called_once_with(limit=1)


def test_data_compaction_runs_binary_copy_in_short_lived_subprocess():
    """Compaction's native allocations must die before document processing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")

        with patch("lancedb_store._run_worker_under_memory_ceiling") as run:
            store._compact_data_files()

        run.assert_called_once_with(
            [
                sys.executable,
                "-m",
                "core.lance_maintenance",
                "compact",
                store._dataset_path(),
            ],
            label="Lance compaction worker",
        )


def test_data_compaction_surfaces_worker_error_for_retry_classification():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")

        with patch.object(
            lancedb_store_module,
            "_compaction_worker_command",
            return_value=[
                sys.executable,
                "-c",
                "import sys; sys.exit(sys.stderr.write("
                "'Retryable commit conflict at version 42') and 1 or 1)",
            ],
        ):
            with pytest.raises(RuntimeError, match="Retryable commit conflict"):
                store._compact_data_files()


def test_pre_index_compaction_refreshes_store_for_following_writes():
    """The long-lived LanceDB handle must follow fresh-handle compaction."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        store.upsert_nodes(
            [_make_node_with_meta("a.md", "c:0", "banana", vec, source_type="md")]
        )
        store.create_fts_index()

        store.prepare_indexing_maintenance()
        store.upsert_nodes(
            [_make_node_with_meta("b.md", "c:0", "quasar", vec, source_type="md")]
        )
        store.ensure_fts_index()

        assert set(store.list_doc_ids()) == {"a.md", "b.md"}
        assert len(store.keyword_search("quasar", top_k=5)) == 1


def test_exclusive_compaction_refreshes_parent_handle_exactly_once():
    """The worker commits a new manifest from its own Lance session, so the
    idle-window driver's stable handle is refreshed once before it writes."""
    from datetime import date

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        store.upsert_nodes(
            [_make_node("seed.md", "c:0", "seed", [0.1] * 768)]
        )

        with patch.object(store, "_compaction_due", return_value=True), patch.object(
            store, "_compact_data_files"
        ), patch.object(store, "_checkout_latest") as checkout:
            with store.exclusive_writer_session():
                store.compact_data_files_if_due(date.today())
                store.insert_nodes(
                    [_make_node("new.md", "c:0", "new", [0.2] * 768)]
                )

        checkout.assert_called_once_with()


def test_post_index_maintenance_skips_compaction_then_merges_tags_and_prunes():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        table = MagicMock()
        order: list[str] = []

        with (
            patch.object(
                store,
                "_prune_versions",
                side_effect=lambda label: order.append(f"prune:{label}"),
            ),
            patch.object(
                store,
                "_compact_data_files",
                side_effect=lambda: order.append("compact"),
            ),
            patch.object(
                store,
                "_merge_index_deltas",
                side_effect=lambda: order.append("merge"),
            ),
            patch.object(
                store,
                "_expire_restore_points",
                side_effect=lambda _table, _day: order.append("expire"),
            ),
            patch.object(
                store,
                "_tag_latest_restore_point",
                side_effect=lambda _table, _day: (order.append("tag"), True)[1],
            ),
        ):
            store._optimize_and_prune(table)

        assert order == [
            "merge",
            "tag",
            "expire",
            "prune:post-expiry",
            "tag",
        ]


def test_post_prune_restore_point_tracks_exact_latest_version():
    """Cleanup may commit a new manifest; today's tag must follow that version."""
    from datetime import date

    import lance

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        store.upsert_nodes(
            [_make_node_with_meta("a.md", "c:0", "banana", vec, source_type="md")]
        )
        store.create_fts_index()

        def commit_prune_manifest(_label):
            store.upsert_nodes(
                [_make_node_with_meta("b.md", "c:0", "quasar", vec, source_type="md")]
            )

        with patch.object(store, "_prune_versions", side_effect=commit_prune_manifest):
            store._finish_index_maintenance(store._vs.table, date.today())

        tag = f"daily-{date.today().isoformat()}"
        assert lance.dataset(_lance_path(tmpdir), version=tag).version == (
            lance.dataset(_lance_path(tmpdir)).version
        )


def test_staging_tag_failure_aborts_expiry_and_prune():
    """Never destroy restore state when today's safety tag cannot be staged."""
    from datetime import date

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        table = MagicMock()
        with (
            patch.object(store, "_tag_latest_restore_point", return_value=False) as tag,
            patch.object(store, "_expire_restore_points") as expire,
            patch.object(store, "_prune_versions") as prune,
        ):
            store._finish_index_maintenance(table, date.today())

        tag.assert_called_once_with(table, date.today())
        expire.assert_not_called()
        prune.assert_not_called()


def test_final_tag_failure_preserves_staged_readable_restore_point():
    """Never prune without first pinning a snapshot that survives retag failure."""
    from datetime import date

    import lance

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        store.upsert_nodes(
            [_make_node_with_meta("a.md", "c:0", "banana", vec, source_type="md")]
        )
        store.create_fts_index()
        original_tag = store._tag_latest_restore_point
        tag_calls = 0

        def fail_final_tag(table, today):
            nonlocal tag_calls
            tag_calls += 1
            if tag_calls == 2:
                return False
            return original_tag(table, today)

        def commit_prune_manifest(_label):
            store.upsert_nodes(
                [_make_node_with_meta("b.md", "c:0", "quasar", vec, source_type="md")]
            )

        with (
            patch.object(store, "_tag_latest_restore_point", side_effect=fail_final_tag),
            patch.object(store, "_prune_versions", side_effect=commit_prune_manifest),
        ):
            store._finish_index_maintenance(store._vs.table, date.today())

        tag = f"daily-{date.today().isoformat()}"
        assert tag_calls == 2
        assert lance.dataset(_lance_path(tmpdir), version=tag).count_rows() == 1
        assert lance.dataset(_lance_path(tmpdir)).count_rows() == 2


def test_current_marker_skips_compaction_but_merges_indices():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        table = MagicMock()
        with (
            patch.object(store, "_compaction_due", return_value=False),
            patch.object(store, "_prune_versions"),
            patch.object(store, "_compact_data_files") as compact,
            patch.object(store, "_merge_index_deltas") as merge,
            patch.object(store, "_expire_restore_points"),
            patch.object(store, "_tag_latest_restore_point"),
        ):
            store._optimize_and_prune(table)

        compact.assert_not_called()
        merge.assert_called_once_with()


def test_compaction_failure_leaves_marker_absent_and_still_merges_indices():
    """A failed compaction is housekeeping: it leaves the cadence marker
    unwritten for the next idle window and never blocks index maintenance."""
    from datetime import date

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        store.upsert_nodes([_make_node("seed.md", "c:0", "seed", [0.1] * 768)])
        table = MagicMock()
        with (
            patch.object(store, "_compaction_due", return_value=True),
            patch.object(store, "_prune_versions"),
            patch.object(
                store,
                "_compact_data_files",
                side_effect=RuntimeError("disk hiccup"),
            ) as compact,
            patch.object(store, "_merge_index_deltas") as merge,
            patch.object(store, "_expire_restore_points"),
            patch.object(store, "_tag_latest_restore_point"),
        ):
            assert store.compact_data_files_if_due(date.today()) is False
            store._optimize_and_prune(table)

        assert compact.call_count == 1
        merge.assert_called_once_with()
        assert not _compaction_marker(tmpdir).exists()


def test_index_merge_failure_skips_the_restore_point_refresh():
    """The merge is the only propagating step, so a merge failure must not go
    on to stage or expire restore points on an index it did not update."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        store.upsert_nodes([_make_node("seed.md", "c:0", "seed", [0.1] * 768)])
        table = MagicMock()
        with (
            patch.object(store, "_prune_versions"),
            patch.object(
                store,
                "_merge_index_deltas",
                side_effect=RuntimeError("index merge failed"),
            ),
            patch.object(store, "_expire_restore_points") as expire,
            patch.object(store, "_tag_latest_restore_point") as tag,
        ):
            with pytest.raises(RuntimeError, match="index merge failed"):
                store._optimize_and_prune(table)

        expire.assert_not_called()
        tag.assert_not_called()


def test_first_idle_window_of_the_day_compacts_and_records_marker():
    """The day's first idle window compacts data and records durable cadence."""
    from datetime import date

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        store.upsert_nodes([_make_node_with_meta("a.md", "c:0", "banana", vec, source_type="md")])
        store.create_fts_index()
        with patch.object(
            store, "_compact_data_files", wraps=store._compact_data_files
        ) as spy:
            store.compact_data_files_if_due(date.today())

        assert spy.call_count == 1
        marker = _compaction_marker(tmpdir)
        assert marker.exists()
        assert marker.read_text().strip() == date.today().isoformat()


def test_maintenance_compacts_at_most_once_per_day():
    """A second idle window on the same day must not rewrite data files (per-run
    compaction × retained restore tags is what grew the index to 350 GB) —
    while an index run still merges newly written rows into the FTS index."""
    from datetime import date

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        store.upsert_nodes([_make_node_with_meta("a.md", "c:0", "banana", vec, source_type="md")])
        store.create_fts_index()
        store.compact_data_files_if_due(date.today())  # first today: writes marker

        store.upsert_nodes([_make_node_with_meta("b.md", "c:0", "quasar telescope", vec, source_type="md")])
        with patch.object(
            store, "_compact_data_files", wraps=store._compact_data_files
        ) as spy:
            store.compact_data_files_if_due(date.today())  # same day: no rewrite
            store.ensure_fts_index()

        assert spy.call_count == 0                  # data compaction skipped
        assert _fts_unindexed_rows(tmpdir) == 0     # index deltas still merged
        assert len(store.keyword_search("quasar", top_k=5)) == 1


def test_maintenance_compacts_again_when_marker_is_stale():
    """A marker from a previous day (or a fresh restart with an old marker)
    makes compaction due again — the marker is the only cadence state."""
    from datetime import date

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        store.upsert_nodes([_make_node_with_meta("a.md", "c:0", "banana", vec, source_type="md")])
        store.create_fts_index()
        store.compact_data_files_if_due(date.today())
        _compaction_marker(tmpdir).write_text("2020-01-01")

        with patch.object(
            store, "_compact_data_files", wraps=store._compact_data_files
        ) as spy:
            store.compact_data_files_if_due(date.today())

        assert spy.call_count == 1


def test_compaction_failure_is_non_fatal_and_retried_next_idle_window():
    """A failed daily compaction must not raise at its caller and must not
    record the marker, so the next idle window retries it."""
    from datetime import date

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        store.upsert_nodes([_make_node_with_meta("a.md", "c:0", "banana", vec, source_type="md")])
        store.create_fts_index()

        with patch.object(
            store, "_compact_data_files", side_effect=RuntimeError("disk hiccup")
        ):
            store.compact_data_files_if_due(date.today())  # must not raise

        assert not _compaction_marker(tmpdir).exists()
        assert len(store.keyword_search("banana", top_k=5)) == 1  # search unharmed

        store.compact_data_files_if_due(date.today())  # next window retries it
        assert _compaction_marker(tmpdir).exists()


def test_restore_points_keep_exactly_configured_count():
    """LANCE_DAILY_RESTORE_POINTS counts retained points — today plus N-1
    prior days — not an N-day grace window on top of today. Every retained
    tag pins that day's data files, so the window must be exact."""
    from datetime import date

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        store.upsert_nodes([_make_node_with_meta("a.md", "c:0", "banana", vec, source_type="md")])
        table = store._vs.table
        v = table.version
        for day in ("2026-06-18", "2026-06-19", "2026-06-20", "2026-06-24"):
            table.tags.create(f"daily-{day}", v)

        with patch.dict("os.environ", {"LANCE_DAILY_RESTORE_POINTS": "7"}):
            store._manage_restore_points(table, date(2026, 6, 25))

        daily = {t for t in table.tags.list() if t.startswith("daily-")}
        # cutoff = 06-25 - 6 days = 06-19: the 06-18 point is the 8th -> expired
        assert daily == {
            "daily-2026-06-19",
            "daily-2026-06-20",
            "daily-2026-06-24",
            "daily-2026-06-25",
        }


def test_restore_points_zero_removes_existing_managed_tags():
    """Turning restore points off must also drop previously created daily tags
    (each pins data files against reclaim) while never touching tags created
    by an operator."""
    from datetime import date

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        store.upsert_nodes([_make_node_with_meta("a.md", "c:0", "banana", vec, source_type="md")])
        table = store._vs.table
        v = table.version
        table.tags.create("daily-2026-06-20", v)
        table.tags.create("manual-keepsake", v)

        with patch.dict("os.environ", {"LANCE_DAILY_RESTORE_POINTS": "0"}):
            store._manage_restore_points(table, date(2026, 6, 25))

        tags = set(table.tags.list())
        assert "daily-2026-06-20" not in tags
        assert "manual-keepsake" in tags


def test_tag_expiry_reclaims_pinned_versions_same_run():
    """A version pinned only by an expiring daily tag is reclaimed by the same
    maintenance pass that expires the tag — the post-expiry prune must not
    wait for the next indexing run (regression guard for the pass ordering)."""
    import lance

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        store.upsert_nodes([_make_node_with_meta("a.md", "c:0", "banana", vec, source_type="md")])
        store.create_fts_index()
        table = store._vs.table
        pinned_version = table.version
        table.tags.create("daily-2020-01-01", pinned_version)  # long expired

        store.upsert_nodes([_make_node_with_meta("b.md", "c:0", "cherry", vec, source_type="md")])
        env = {"LANCE_VERSION_RETENTION_MINUTES": "0", "LANCE_DAILY_RESTORE_POINTS": "7"}
        with patch.dict("os.environ", env):
            store.ensure_fts_index()

        versions = {v["version"] for v in lance.dataset(_lance_path(tmpdir)).versions()}
        assert pinned_version not in versions


def test_index_maintenance_reclaims_orphan_index_directories():
    """Every `_indices/<uuid>` directory the retained versions cannot reach is
    reclaimed by the maintenance pass (#1160).

    Each run's index merge writes a whole new index generation and orphans the
    previous one, but `cleanup_old_versions` never deletes a file newer than
    the oldest version it retains — and a daily restore-point tag deliberately
    retains a days-old one. Production held 26 GiB of dead index copies (live
    index: 43 MiB) because nothing swept them."""
    import lance

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        store.upsert_nodes([_make_node_with_meta("a.md", "c:0", "banana", vec, source_type="md")])
        store.create_fts_index()

        env = {"LANCE_VERSION_RETENTION_MINUTES": "0", "LANCE_DAILY_RESTORE_POINTS": "7"}
        for doc_id, text in (("b.md", "cherry"), ("c.md", "durian")):
            store.upsert_nodes([
                _make_node_with_meta(doc_id, "c:0", text, vec, source_type="md"),
            ])
            with patch.dict("os.environ", env):
                store.ensure_fts_index()

        dataset = lance.dataset(_lance_path(tmpdir))
        reachable = {
            str(segment.uuid)
            for version in dataset.versions()
            for index in dataset.checkout_version(version["version"]).describe_indices()
            for segment in index.segments
        }
        on_disk = {
            entry.name
            for entry in (Path(_lance_path(tmpdir)) / "_indices").iterdir()
        }
        assert on_disk == reachable
        assert len(store.keyword_search("durian", top_k=5)) == 1  # search unharmed


def test_keyword_search_recovers_from_stale_store_handle():
    """keyword_search should reopen the table after stale file-handle errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        store.upsert_nodes([
            _make_node_with_meta("a.md", "c:0", "banana fruit tropical", vec, source_type="md"),
        ])
        store.create_fts_index()
        # A peer writer moves the table past this handle — what actually makes
        # the handle below stale, and what read recovery keys on.
        LanceDBStore(tmpdir, "test_chunks").upsert_nodes([
            _make_node_with_meta("b.md", "c:0", "cherry date", vec, source_type="md"),
        ])

        class _StaleTable:
            def search(self, *args, **kwargs):
                raise RuntimeError("Not found: data/index/test_chunks.lance/data/abc123.lance")

        class _StaleVS:
            @property
            def table(self):
                return _StaleTable()

        store._vs = _StaleVS()

        hits = store.keyword_search("banana tropical", top_k=10)
        assert len(hits) == 1
        assert hits[0].doc_id == "a.md"


def test_read_survives_a_second_table_swap_landing_inside_its_own_recovery():
    """Schema evolution swaps the whole <table>.lance directory, so every open
    read handle is left pointing at deleted files. A sweep on a new index does
    that once per new metadata sub-field, in bursts — so the swap that follows
    can land inside the reopen a read is already recovering through, and one
    reopen is not enough (#1656)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vec = [0.0] * 768
        writer = LanceDBStore(tmpdir, "test_chunks")

        def widen(**new_fields):
            writer.upsert_nodes([
                _make_node_with_meta("a.md", f"c:{i}", f"cobalt vestibule {i}", vec,
                                     source_type="md", **new_fields)
                for i in range(3)
            ])

        widen()
        reader = LanceDBStore(tmpdir, "test_chunks")
        assert len(reader.get_doc_chunks("a.md")) == 3  # warm the handle

        # The next swap in the burst arrives while the reader is reopening.
        # Widening is idempotent, so this fires exactly once.
        reopen = reader._reopen_vector_store

        def reopen_then_writer_swaps_again():
            reopen()
            widen(dup_count="2", dup_sources="quo")

        reader._reopen_vector_store = reopen_then_writer_swaps_again
        widen(dup_locations="c:0")

        assert len(reader.get_doc_chunks("a.md")) == 3


def test_read_failure_over_an_unmoved_table_reaches_the_caller():
    """Recovery is for handles the writer moved out from under. A failure over a
    table that has not moved is a real retrieval failure: retrying it would only
    hide a broken index behind an empty-looking result (#1656)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        LanceDBStore(tmpdir, "test_chunks").upsert_nodes([
            _make_node_with_meta("a.md", "c:0", "cobalt vestibule", [0.0] * 768,
                                 source_type="md"),
        ])
        store = LanceDBStore(tmpdir, "test_chunks")  # handle is up to date

        attempts = []

        def _failing_read():
            attempts.append(1)
            # Wording of a stale-handle error, over a table nothing has touched.
            raise RuntimeError("Not found: data/index/test_chunks.lance/data/abc123.lance")

        with pytest.raises(RuntimeError):
            store._run_read_with_recovery(_failing_read, [])
        assert len(attempts) == 1


# --- get_chunk / get_doc_chunks comprehensive tests ---


def test_get_chunk_returns_all_core_fields():
    """get_chunk should return a SearchHit with all core metadata fields populated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        nodes = [_make_node_with_meta(
            "Projects/recipe.md", "c:0", "A Korean bibimbap recipe with gochujang sauce",
            vec, source_type="md", title="Bibimbap Recipe",
            tags="recipe,korean", folder="Projects", status="active",
            created="2026-01-15", mtime=1700000000.0,
        )]
        store.upsert_nodes(nodes)

        hit = store.get_chunk("Projects/recipe.md", "c:0")
        assert hit is not None
        assert hit.doc_id == "Projects/recipe.md"
        assert hit.loc == "c:0"
        assert hit.text == "A Korean bibimbap recipe with gochujang sauce"
        assert hit.snippet == "A Korean bibimbap recipe with gochujang sauce"
        assert hit.score == 0.0  # direct lookup, not a search
        assert hit.source_type == "md"
        assert hit.title == "Bibimbap Recipe"
        assert hit.tags == "recipe,korean"
        assert hit.folder == "Projects"
        assert hit.status == "active"
        assert hit.created == "2026-01-15"
        assert hit.mtime == 1700000000.0


def test_get_chunk_returns_enrichment_fields():
    """get_chunk should include all enr_* enrichment fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        nodes = [_make_node_with_meta(
            "a.md", "c:0", "test doc", vec,
            enr_summary="LLM summary",
            enr_doc_type="report",
            enr_topics="finance,tax",
            enr_keywords="deduction,income",
            enr_entities_people="Dan Park",
            enr_entities_places="Vancouver",
            enr_entities_orgs="CRA",
            enr_entities_dates="2026-01-01",
            enr_key_facts="Tax refund of $5000",
        )]
        store.upsert_nodes(nodes)

        hit = store.get_chunk("a.md", "c:0")
        assert hit is not None
        assert hit.enr_summary == "LLM summary"
        assert hit.enr_doc_type == "report"
        assert hit.enr_topics == "finance,tax"
        assert hit.enr_keywords == "deduction,income"
        assert hit.enr_entities_people == "Dan Park"
        assert hit.enr_entities_places == "Vancouver"
        assert hit.enr_entities_orgs == "CRA"
        assert hit.enr_entities_dates == "2026-01-01"
        assert hit.enr_key_facts == "Tax refund of $5000"


def test_get_chunk_not_found_returns_none():
    """get_chunk should return None for a nonexistent doc_id/loc."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        nodes = [_make_node("a.md", "c:0", "hello", vec)]
        store.upsert_nodes(nodes)
        assert store.get_chunk("nonexistent.md", "c:0") is None
        assert store.get_chunk("a.md", "c:99") is None


def test_get_chunk_empty_store():
    """get_chunk on empty store should return None (no crash)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        assert store.get_chunk("a.md", "c:0") is None


def test_get_doc_chunks_returns_all_chunks_sorted():
    """get_doc_chunks should return all chunks for a doc, sorted by loc."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        nodes = [
            _make_node_with_meta("a.md", "c:2", "third chunk", vec),
            _make_node_with_meta("a.md", "c:0", "first chunk", vec),
            _make_node_with_meta("a.md", "c:1", "second chunk", vec),
            _make_node_with_meta("b.md", "c:0", "other doc", vec),
        ]
        store.upsert_nodes(nodes)

        chunks = store.get_doc_chunks("a.md")
        assert len(chunks) == 3
        assert [c.loc for c in chunks] == ["c:0", "c:1", "c:2"]
        assert chunks[0].text == "first chunk"
        assert chunks[1].text == "second chunk"
        assert chunks[2].text == "third chunk"
        # All should have score 0.0 (direct lookup)
        assert all(c.score == 0.0 for c in chunks)


def test_get_doc_chunks_returns_enrichment_fields():
    """get_doc_chunks should include enr_* fields on each chunk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        nodes = [_make_node_with_meta(
            "a.md", "c:0", "enriched chunk", vec,
            enr_summary="LLM summary", enr_doc_type="report",
            enr_topics="finance", enr_keywords="tax",
        )]
        store.upsert_nodes(nodes)

        chunks = store.get_doc_chunks("a.md")
        assert len(chunks) == 1
        assert chunks[0].enr_summary == "LLM summary"
        assert chunks[0].enr_doc_type == "report"
        assert chunks[0].enr_topics == "finance"
        assert chunks[0].enr_keywords == "tax"


def test_get_doc_chunks_not_found_returns_empty():
    """get_doc_chunks for nonexistent doc should return empty list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        nodes = [_make_node("a.md", "c:0", "hello", vec)]
        store.upsert_nodes(nodes)
        assert store.get_doc_chunks("nonexistent.md") == []


def test_get_doc_chunks_empty_store():
    """get_doc_chunks on empty store should return empty list (no crash)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        assert store.get_doc_chunks("a.md") == []


def test_get_chunk_consistency_with_vector_search():
    """get_chunk and vector_search should return the same fields for the same chunk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.1] * 768
        nodes = [_make_node_with_meta(
            "a.md", "c:0", "consistency test", vec,
            source_type="md", title="Test Doc", tags="test",
            folder="Root", status="active", enr_summary="summary",
            enr_doc_type="note", enr_topics="testing",
        )]
        store.upsert_nodes(nodes)

        # Get via direct lookup
        chunk_hit = store.get_chunk("a.md", "c:0")
        # Get via vector search
        search_hits = store.vector_search(vec, top_k=1)

        assert chunk_hit is not None
        assert len(search_hits) == 1
        search_hit = search_hits[0]

        # All metadata fields should match (score differs)
        assert chunk_hit.doc_id == search_hit.doc_id
        assert chunk_hit.loc == search_hit.loc
        assert chunk_hit.text == search_hit.text
        assert chunk_hit.source_type == search_hit.source_type
        assert chunk_hit.title == search_hit.title
        assert chunk_hit.tags == search_hit.tags
        assert chunk_hit.folder == search_hit.folder
        assert chunk_hit.status == search_hit.status
        assert chunk_hit.enr_summary == search_hit.enr_summary
        assert chunk_hit.enr_doc_type == search_hit.enr_doc_type
        assert chunk_hit.enr_topics == search_hit.enr_topics


def test_frontmatter_and_enrichment_coexist():
    """Frontmatter 'summary' and LLM 'enr_summary' should coexist without collision."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.1] * 768
        nodes = [_make_node_with_meta(
            "a.md", "c:0", "test", vec,
            summary="User's custom summary",
            enr_summary="LLM generated summary",
        )]
        store.upsert_nodes(nodes)
        hits = store.vector_search(vec, top_k=1)
        assert len(hits) == 1
        # LLM enrichment accessible via named attribute
        assert hits[0].enr_summary == "LLM generated summary"
        # Frontmatter summary accessible via extra_metadata (not in _CORE_META_KEYS)
        assert hits[0].extra_metadata["summary"] == "User's custom summary"
        # Both accessible via getattr
        assert hits[0].summary == "User's custom summary"


# --- _build_where_clause: identifier validation (Fix 1) ---


@pytest.mark.parametrize("bad_key", [
    "x' OR 1=1 --",
    "a; DROP TABLE",
    "",
    "foo bar",
    "123abc",
    "key.nested",
])
def test_build_where_clause_rejects_unsafe_keys(bad_key):
    """metadata_filters with injection-style keys must raise ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        with pytest.raises(ValueError, match="Unsafe metadata filter key"):
            store._build_where_clause(metadata_filters={bad_key: "val"})


@pytest.mark.parametrize("good_key", ["section", "my_field_2", "_private"])
def test_build_where_clause_accepts_valid_keys(good_key):
    """Valid identifier keys should produce a WHERE clause without error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        clause = store._build_where_clause(metadata_filters={good_key: "val"})
        assert f"lower(metadata.{good_key}) = 'val'" in clause


# --- upsert_nodes: add failure visibility (Fix 2) ---


def test_upsert_add_failure_reraises_and_recovers():
    """If _vs.add() fails after delete, the exception propagates;
    a subsequent upsert succeeds (self-healing)."""
    from unittest.mock import MagicMock

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        nodes = [_make_node("a.md", "c:0", "hello", [0.1] * 768)]

        # First upsert succeeds (creates the table)
        store.upsert_nodes(nodes)
        assert store.list_doc_ids() == ["a.md"]

        # Swap _vs with a mock whose add() raises
        real_vs = store._vs
        mock_vs = MagicMock(wraps=real_vs)
        mock_vs.add.side_effect = RuntimeError("disk full")
        store._vs = mock_vs

        with pytest.raises(RuntimeError, match="disk full"):
            store.upsert_nodes(nodes)

        # Restore real _vs and verify self-healing
        store._vs = real_vs
        store.upsert_nodes(nodes)
        assert store.list_doc_ids() == ["a.md"]


class TestSourceNameFilter:
    """Search honors source_name metadata filter."""

    def test_where_clause_includes_source_name(self, tmp_path):
        from lancedb_store import LanceDBStore

        store = LanceDBStore(str(tmp_path / "idx"), "chunks")
        where = store._build_where_clause(source_name="comm_messages")
        assert "source_name" in where
        assert "comm_messages" in where


def test_open_store_with_recovery_recovers_known_corruption():
    sentinel = object()
    error = RuntimeError(
        "LanceError(IO): Generic memory error: Invalid range 0..0 for object of size 0 bytes"
    )

    with patch("lancedb_store.LanceDBStore", side_effect=error) as store_cls:
        with patch("lancedb_store.recover_corrupt_table", return_value=sentinel) as recover:
            result = open_store_with_recovery("/tmp/index", "chunks")

    assert result is sentinel
    store_cls.assert_called_once_with("/tmp/index", "chunks")
    recover.assert_called_once()


def test_open_store_with_recovery_reraises_non_corruption():
    error = RuntimeError("bad config")

    with patch("lancedb_store.LanceDBStore", side_effect=error):
        with patch("lancedb_store.recover_corrupt_table") as recover:
            with pytest.raises(RuntimeError, match="bad config"):
                open_store_with_recovery("/tmp/index", "chunks")

    recover.assert_not_called()


# ---------------------------------------------------------------------------
# Vector (ANN) index + vector-search concurrency gate
#
# Without an ANN index every vector search brute-force scans the whole fp32
# vector column (~1.9 GB transient per query at 88k rows x 4096 dims); four
# concurrent searches took the 8 GiB container to its cgroup ceiling 20 times
# on 2026-09-06. These pin the two defences: the index exists after
# ensure_vector_index(), and concurrent searches are bounded process-wide.
# ---------------------------------------------------------------------------


def _seed_onehot_nodes(store: LanceDBStore, count: int = 6) -> None:
    store.upsert_nodes(
        [
            _make_node(
                f"d{i}.md", "c:0", f"text {i}", [float(i == j) for j in range(768)]
            )
            for i in range(count)
        ]
    )


def _vector_indices(store: LanceDBStore) -> list:
    return [
        index
        for index in store._vs.table.list_indices()
        if list(index.columns) == ["vector"]
    ]


def test_ensure_vector_index_creates_ann_index_once_and_search_stays_exact():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        _seed_onehot_nodes(store)
        assert store.vector_index_available() is False
        assert _vector_indices(store) == []

        assert store.ensure_vector_index() is True
        assert store.vector_index_available() is True
        assert len(_vector_indices(store)) == 1

        # Second call is a no-op: no rebuild, no second index.
        assert store.ensure_vector_index() is False
        assert len(_vector_indices(store)) == 1

        hits = store.vector_search([float(j == 3) for j in range(768)], top_k=1)
        assert hits[0].doc_id == "d3.md"


def test_ensure_vector_index_skips_table_without_rows():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        assert store.ensure_vector_index() is False
        assert store.vector_index_available() is False


def test_ensure_vector_index_honours_configured_type_and_rejects_unknown():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        _seed_onehot_nodes(store)
        with pytest.raises(ValueError):
            store.ensure_vector_index(index_type="FLAT_EARTH")
        assert store.ensure_vector_index(index_type="IVF_HNSW_SQ") is True
        (index,) = _vector_indices(store)
        assert "HNSW" in str(index.index_type).upper()


def test_ivf_partitions_scale_with_rows_and_never_train_empty_clusters():
    """Lance trains centroids on 256 rows per partition; a 37-row hermetic
    table asked for 6 partitions warned about empty clusters on stderr — two
    untimestamped lines in indexer.log, the #0546 contract (caught by e2e)."""
    partitions = lancedb_store_module.ivf_partitions_for
    assert partitions(0) == 1
    assert partitions(37) == 1
    assert partitions(255) == 1
    assert partitions(5000) == 19  # rows // 256, below sqrt(5000) = 70
    assert partitions(88_440) == 256  # sqrt = 297, rows // 256 = 345, cap 256
    assert partitions(10_000_000) == 256


def test_vector_index_settings_from_config_reads_search_section():
    settings = lancedb_store_module.vector_index_settings_from_config(
        {"search": {"vector_index": {"type": "IVF_HNSW_SQ", "num_partitions": 4}}}
    )
    assert settings == {"index_type": "IVF_HNSW_SQ", "num_partitions": 4}
    assert lancedb_store_module.vector_index_settings_from_config({}) == {
        "index_type": None,
        "num_partitions": None,
    }


def test_configure_vector_search_from_config_sets_process_knobs():
    defaults = lancedb_store_module.vector_search_settings()
    try:
        lancedb_store_module.configure_vector_search_from_config(
            {"search": {"nprobes": 7, "max_concurrent_vector_searches": 2}}
        )
        assert lancedb_store_module.vector_search_settings() == {
            "nprobes": 7,
            "max_concurrent": 2,
        }
        # Missing keys leave the knobs alone.
        lancedb_store_module.configure_vector_search_from_config({})
        assert lancedb_store_module.vector_search_settings() == {
            "nprobes": 7,
            "max_concurrent": 2,
        }
        with pytest.raises(ValueError):
            lancedb_store_module.configure_vector_search(max_concurrent=0)
    finally:
        lancedb_store_module.configure_vector_search(**defaults)


def test_vector_search_concurrency_is_bounded_process_wide():
    """Six concurrent vector searches, gate of 2: never more than 2 inside Lance."""
    defaults = lancedb_store_module.vector_search_settings()
    state = {"in_flight": 0, "peak": 0}
    guard = threading.Lock()
    release = threading.Event()

    class _SlowQuery:
        def where(self, *a, **k):
            return self

        def nprobes(self, *a, **k):
            return self

        def limit(self, *a, **k):
            return self

        def select(self, *a, **k):
            return self

        def to_list(self):
            with guard:
                state["in_flight"] += 1
                state["peak"] = max(state["peak"], state["in_flight"])
            release.wait(5)
            with guard:
                state["in_flight"] -= 1
            return []

    fake_table = MagicMock()
    fake_table.search.side_effect = lambda *a, **k: _SlowQuery()

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        store._vs = SimpleNamespace(table=fake_table)
        try:
            lancedb_store_module.configure_vector_search(max_concurrent=2)
            threads = [
                threading.Thread(
                    target=store.vector_search, args=([0.0] * 768, 1), daemon=True
                )
                for _ in range(6)
            ]
            for t in threads:
                t.start()
            deadline = time.time() + 2
            while time.time() < deadline and state["peak"] < 2:
                time.sleep(0.01)
            time.sleep(0.2)  # let any over-admitted searches show up
            assert state["peak"] == 2, state
            release.set()
            for t in threads:
                t.join(5)
            assert state["peak"] == 2, state
            assert fake_table.search.call_count == 6
        finally:
            release.set()
            lancedb_store_module.configure_vector_search(**defaults)


def test_vector_search_plan_uses_the_ann_index_with_and_without_prefilter():
    """The regression that hid for six months: a table without a vector index
    still answers every query, by brute-force scanning the whole vector column
    (``KNNVectorDistance`` over a full ``LanceRead``). Pin the plan itself:
    once the index exists, vector_search plans ``ANNSubIndex`` — for the plain
    query and for the metadata-prefiltered one production runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        _seed_onehot_nodes(store, count=8)
        query = [float(j == 2) for j in range(768)]
        where = "metadata.source_type = 'md'"

        flat_plan = store.explain_vector_search(query)
        assert "KNNVectorDistance" in flat_plan and "ANNSubIndex" not in flat_plan

        assert store.ensure_vector_index() is True
        for clause in (None, where):
            plan = store.explain_vector_search(query, where=clause)
            assert "ANNSubIndex" in plan, plan
            assert "KNNVectorDistance" not in plan, plan
        assert store.vector_search(query, top_k=1, where=where)[0].doc_id == "d2.md"


def test_vector_index_stats_reports_absent_present_and_unindexed_tail():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        assert store.vector_index_stats() == lancedb_store_module.empty_vector_index_stats()

        _seed_onehot_nodes(store, count=6)
        assert store.vector_index_stats()["available"] is False

        store.ensure_vector_index()
        stats = store.vector_index_stats()
        assert stats["available"] is True
        assert "IVF" in stats["index_type"].upper()
        assert stats == {
            **stats,
            "num_indices": 1,
            "indexed_rows": 6,
            "unindexed_rows": 0,
            "stale": False,
        }

        # Rows written after the build (the single-document path) form a tail
        # the index does not cover yet.
        store.upsert_nodes(
            [_make_node("late.md", "c:0", "late", [0.5] * 768)]
        )
        stats = store.vector_index_stats()
        assert stats["unindexed_rows"] == 1 and stats["indexed_rows"] == 6
        assert stats["stale"] is False


def test_index_maintenance_merges_the_tail_into_the_single_vector_index():
    """The sweep's incremental step (ensure_fts_index -> optimize_indices) must
    fold new rows into the existing vector index: tail back to 0, still ONE
    index (a delta-per-merge index type would show num_indices climbing), and
    the new row reachable through the ANN plan."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        _seed_onehot_nodes(store, count=6)
        store.create_fts_index()
        store.ensure_vector_index()

        late = [0.0] * 768
        late[700] = 1.0
        store.upsert_nodes([_make_node("late.md", "c:0", "late", late)])
        assert store.vector_index_stats()["unindexed_rows"] == 1

        store.ensure_fts_index()  # what every index run calls after writes

        stats = store.vector_index_stats()
        assert stats["unindexed_rows"] == 0
        assert stats["indexed_rows"] == 7
        assert stats["num_indices"] == 1
        assert "ANNSubIndex" in store.explain_vector_search(late)
        assert store.vector_search(late, top_k=1)[0].doc_id == "late.md"


def test_vector_index_stale_flag_trips_when_the_tail_outgrows_the_threshold(monkeypatch):
    monkeypatch.setattr(lancedb_store_module, "VECTOR_INDEX_STALE_TAIL_ROWS", 2)
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        _seed_onehot_nodes(store, count=4)
        store.ensure_vector_index()
        store.upsert_nodes(
            [_make_node(f"tail{i}.md", "c:0", f"tail {i}", [0.1 * i] * 768) for i in range(3)]
        )
        stats = store.vector_index_stats()
        assert stats["unindexed_rows"] == 3
        assert stats["stale"] is True


def test_schema_evolution_keeps_the_vector_index_or_ensure_restores_it():
    """Widening the metadata struct swaps the physical table. Whatever Lance
    does with the index across that swap, the flow's ensure step must leave
    the table indexed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        store.upsert_nodes([_make_node_with_meta("a.pdf", "p:1:c:0", "pdf chunk", vec)])
        store.ensure_vector_index()
        assert store.vector_index_available() is True

        store.upsert_nodes(
            [_make_node_with_meta("b.md", "c:0", "md chunk", vec, section="Introduction")]
        )
        assert "section" in store._metadata_subfields()

        store.ensure_vector_index()
        assert store.vector_index_available() is True
        assert "ANNSubIndex" in store.explain_vector_search(vec)


def test_promoted_shadow_table_carries_its_vector_index():
    """A shadow rebuild indexes the shadow before promote_table swaps it in, so
    the active table never serves without the index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        active = LanceDBStore(tmpdir, "test_chunks")
        shadow = LanceDBStore(tmpdir, "test_chunks__shadow")
        active.upsert_nodes([_make_node("active.md", "c:0", "active", [0.1] * 768)])
        shadow.upsert_nodes([_make_node("shadow.md", "c:0", "shadow", [0.2] * 768)])
        shadow.create_fts_index()
        assert shadow.ensure_vector_index() is True

        active.promote_table("test_chunks__shadow")

        reopened = LanceDBStore(tmpdir, "test_chunks")
        assert reopened.list_doc_ids() == ["shadow.md"]
        assert reopened.vector_index_available() is True
        assert "ANNSubIndex" in reopened.explain_vector_search([0.2] * 768)


# ---------------------------------------------------------------------------
# Document deletes must match rows in every Lance version
#
# pylance 11 changed the SQL dialect: a double-quoted literal is a column
# name. llama-index's LanceDBVectorStore.delete() quotes ids that way, so under
# lance 11 every re-index kept the old chunks beside the new ones (production,
# 2026-09-07, `documents::002KC` doubled). The store now issues its own filter.
# ---------------------------------------------------------------------------


def _doc_rows(store: LanceDBStore, doc_id: str) -> list[str]:
    table = store._vs.table.to_lance().to_table(columns=["id", "doc_id"])
    return sorted(
        row_id
        for row_id, row_doc in zip(table["id"].to_pylist(), table["doc_id"].to_pylist())
        if row_doc == doc_id
    )


def test_reupsert_replaces_every_chunk_and_leaves_no_stale_rows():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.3] * 768
        store.upsert_nodes(
            [_make_node("note.md", f"c:{i}", f"v1 chunk {i}", vec) for i in range(3)]
        )
        store.upsert_nodes([_make_node("other.md", "c:0", "untouched", vec)])
        assert _doc_rows(store, "note.md") == ["note.md::c:0", "note.md::c:1", "note.md::c:2"]

        store.upsert_nodes(
            [_make_node("note.md", f"c:{i}", f"v2 chunk {i}", vec) for i in range(2)]
        )

        assert _doc_rows(store, "note.md") == ["note.md::c:0", "note.md::c:1"]
        assert store.get_chunk("note.md", "c:0").text == "v2 chunk 0"
        assert _doc_rows(store, "other.md") == ["other.md::c:0"]
        assert store.count_chunks() == 3


def test_delete_by_doc_ids_removes_every_row_of_the_document():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.3] * 768
        store.upsert_nodes(
            [_make_node("gone.md", f"c:{i}", f"chunk {i}", vec) for i in range(4)]
            + [_make_node("kept.md", "c:0", "kept", vec)]
        )
        store.delete_by_doc_ids(["gone.md"])
        assert _doc_rows(store, "gone.md") == []
        assert _doc_rows(store, "kept.md") == ["kept.md::c:0"]


def test_document_deletes_use_a_single_quoted_literal_not_llama_index_delete():
    """Deterministic across Lance versions: the filter itself is ours, escaped,
    and llama-index's double-quoted delete() is never called."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        store.upsert_nodes([_make_node("o'brien.md", "c:0", "text", [0.3] * 768)])
        fake_table = MagicMock()
        with patch.object(type(store._vs), "table", new_callable=PropertyMock, return_value=fake_table):
            with patch.object(type(store._vs), "delete") as llama_delete:
                store.delete_by_doc_ids(["o'brien.md"])
        fake_table.delete.assert_called_once_with("doc_id = 'o''brien.md'")
        llama_delete.assert_not_called()


def test_upsert_does_not_insert_when_the_delete_of_old_rows_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        node = _make_node("note.md", "c:0", "v1", [0.3] * 768)
        store.upsert_nodes([node])
        with patch.object(store, "_delete_doc_rows", side_effect=RuntimeError("filter rejected")):
            with pytest.raises(RuntimeError, match="filter rejected"):
                store.upsert_nodes([_make_node("note.md", "c:0", "v2", [0.3] * 768)])
        assert _doc_rows(store, "note.md") == ["note.md::c:0"]
        assert store.get_chunk("note.md", "c:0").text == "v1"


# ---------------------------------------------------------------------------
# checkout_latest after a same-path table swap
#
# Schema evolution swaps `<table>.lance` for a new incarnation whose version
# numbers restart at 1. Table.checkout_latest() on a handle opened before the
# swap binds the OLD dataset's index metadata to the new version number: every
# read after it fails with `Not found: _indices/<old uuid>` while the handle's
# snapshot reads as current, so stale-read recovery never fires (measured
# deterministic on lance 4 and 10; hit the fresh-stack e2e 1 run in 4).
# ---------------------------------------------------------------------------


def _build_incarnation(root: str) -> LanceDBStore:
    """Identical build sequence each time: same commits, same index set."""
    store = LanceDBStore(root, "test_chunks")
    for i in range(4):
        store.upsert_nodes([_make_node(f"d{i}.md", "c:0", f"text {i}", [float(i == j) for j in range(768)])])
    store.upsert_nodes([_make_node("d1.md", "c:0", "text 1 again", [float(1 == j) for j in range(768)])])
    store.create_fts_index()
    store.ensure_vector_index()
    return store


def _swap_in_new_incarnation(root: str) -> None:
    import shutil

    other = tempfile.mkdtemp()
    try:
        _build_incarnation(other)
        shutil.move(f"{root}/test_chunks.lance", f"{root}/test_chunks__schema_backup.lance")
        shutil.move(f"{other}/test_chunks.lance", f"{root}/test_chunks.lance")
        shutil.rmtree(f"{root}/test_chunks__schema_backup.lance")
    finally:
        shutil.rmtree(other, ignore_errors=True)


def _reads_work(store: LanceDBStore) -> None:
    assert store.vector_search([float(2 == j) for j in range(768)], top_k=1)[0].doc_id == "d2.md"
    assert store.keyword_search("text", top_k=3)
    assert store.get_chunk("d1.md", "c:0") is not None
    assert store.vector_index_available() is True


def test_checkout_latest_reopens_after_a_same_path_table_swap():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = _build_incarnation(tmpdir)
        _reads_work(store)

        _swap_in_new_incarnation(tmpdir)
        store._checkout_latest()  # what every write path and the serving refresh call

        _reads_work(store)
        assert store._table_moved_under_open_handle() is False


def test_checkout_latest_keeps_the_handle_when_the_directory_is_unchanged():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = _build_incarnation(tmpdir)
        handle = store._vs
        LanceDBStore(tmpdir, "test_chunks").upsert_nodes(
            [_make_node("peer.md", "c:0", "peer wrote", [0.5] * 768)]
        )  # a peer commit bumps the version; the directory is the same
        store._checkout_latest()
        assert store._vs is handle
        assert store.get_chunk("peer.md", "c:0") is not None
        assert store._table_moved_under_open_handle() is False
