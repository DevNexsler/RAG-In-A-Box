"""A metadata-only write must not take indexed search offline."""
from concurrent.futures import ThreadPoolExecutor
import os
import signal
import threading
import subprocess
import sys

import pytest

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

from lancedb_store import LanceDBStore


VECTOR = [1.0] + [0.0] * 31


def node(doc_id, **metadata):
    value = TextNode(
        id_=f"{doc_id}::c:0", text=f"searchable migration sentinel {doc_id}",
        embedding=VECTOR, metadata={"doc_id": doc_id, **metadata},
    )
    value.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
    return value


def assert_searchable(store):
    assert "ANNSubIndex" in store.explain_vector_search(VECTOR)
    assert store.vector_search(VECTOR, top_k=1)
    assert store.keyword_search("sentinel", top_k=1)


def test_schema_write_keeps_concurrent_and_fresh_readers_indexed(tmp_path):
    writer = LanceDBStore(str(tmp_path), "chunks")
    writer.insert_nodes([node("original")])
    writer.ensure_vector_index()
    writer.create_fts_index()
    reader = LanceDBStore(str(tmp_path), "chunks")
    ready, done = threading.Event(), threading.Event()

    def search_during_writes():
        observations = 0
        while not done.is_set():
            assert_searchable(reader)
            observations += 1
            ready.set()
            done.wait(.002)
        assert_searchable(reader)
        return observations

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(search_during_writes)
        try:
            assert ready.wait(10), "reader did not start"
            for number in range(3):
                writer.insert_nodes([node(f"new-{number}", **{f"new_field_{number}": "value"})])
                assert_searchable(LanceDBStore(str(tmp_path), "chunks"))
        finally:
            done.set()
        assert future.result(timeout=10) >= 2
    assert writer.count_chunks() == 4
    assert writer.get_doc_chunks("new-2")[0].extra_metadata["new_field_2"] == "value"


def test_failed_index_worker_keeps_original_searchable_and_retryable(tmp_path, monkeypatch):
    writer = LanceDBStore(str(tmp_path), "chunks")
    writer.insert_nodes([node("original")])
    writer.ensure_vector_index()
    writer.create_fts_index()
    reader = LanceDBStore(str(tmp_path), "chunks")
    original_popen = subprocess.Popen

    def fail_rebuild(command, **kwargs):
        if "core.schema_index_rebuild" in command:
            command = [sys.executable, "-c", "raise SystemExit('injected index disk failure')"]
        return original_popen(command, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(subprocess, "Popen", fail_rebuild)
        with pytest.raises(RuntimeError, match="injected index disk failure"):
            writer.insert_nodes([node("pending", new_field="value")])
    assert writer.count_chunks() == 1
    assert writer.get_doc_chunks("pending") == []
    assert_searchable(reader)
    assert_searchable(LanceDBStore(str(tmp_path), "chunks"))
    assert not list(tmp_path.glob("*__schema*"))
    writer.insert_nodes([node("pending", new_field="value")])
    assert writer.count_chunks() == 2
    assert_searchable(writer)


@pytest.mark.parametrize("cancel_at", ["communicate", "spawn", "watcher_start"])
def test_cancelled_index_worker_is_reaped_before_schema_cleanup(tmp_path, monkeypatch, cancel_at):
    writer = LanceDBStore(str(tmp_path), "chunks")
    writer.insert_nodes([node("original")])
    writer.ensure_vector_index()
    writer.create_fts_index()
    original_popen = subprocess.Popen
    original_thread_start = threading.Thread.start
    workers = []

    def start_watcher(thread):
        if cancel_at == "watcher_start" and thread.name == "worker-memory-ceiling":
            raise KeyboardInterrupt("cancel watcher startup")
        return original_thread_start(thread)

    def cancel_rebuild(command, **kwargs):
        if "core.schema_index_rebuild" not in command:
            return original_popen(command, **kwargs)
        worker = original_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        workers.append(worker)
        if cancel_at == "spawn":
            os.kill(os.getpid(), signal.SIGINT)
            return worker
        if cancel_at == "watcher_start":
            return worker
        communicate = worker.communicate

        def interrupted(*args, **kwargs):
            worker.communicate = communicate
            raise KeyboardInterrupt("cancel schema rebuild")

        worker.communicate = interrupted
        return worker

    try:
        with monkeypatch.context() as fault:
            fault.setattr(subprocess, "Popen", cancel_rebuild)
            fault.setattr(threading.Thread, "start", start_watcher)
            with pytest.raises(SystemExit if cancel_at == "spawn" else KeyboardInterrupt):
                writer.insert_nodes([node("pending", new_field="value")])
        assert workers and all(worker.poll() is not None for worker in workers)
        assert writer.count_chunks() == 1
        assert_searchable(LanceDBStore(str(tmp_path), "chunks"))
        assert not list(tmp_path.glob("*__schema*"))
    finally:
        # A red regression must not leave its deliberately leaked child alive.
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
            worker.communicate(timeout=10)


@pytest.mark.parametrize("index_type", ["IVF_FLAT", "IVF_HNSW_SQ", "IVF_PQ"])
def test_replacement_preserves_index_settings_and_phrase_search(tmp_path, index_type):
    import lance
    import lancedb
    import numpy as np

    from core.schema_index_rebuild import rebuild_indexes

    rows = [
        {"doc_id": str(i), "text": "Alpha beta gamma", "vector": vector.tolist()}
        for i, vector in enumerate(np.random.default_rng(7).random((256, 32)).astype("float32"))
    ]
    database = lancedb.connect(str(tmp_path))
    original = database.create_table("original", rows)
    database.create_table("replacement", rows)
    original.create_index(index_type=index_type, num_partitions=2, num_sub_vectors=4, metric="cosine")
    original.create_fts_index("text", use_tantivy=False, with_position=True, stem=False)
    original.create_scalar_index("doc_id", index_type="BTREE")
    rebuild_indexes(str(tmp_path / "original.lance"), str(tmp_path / "replacement.lance"))
    replacement = database.open_table("replacement")
    stats = lance.dataset(str(tmp_path / "replacement.lance")).stats.index_stats("vector_idx")
    assert stats["index_type"] == index_type
    assert stats["indices"][0]["metric_type"] == "cosine"
    assert stats["indices"][0]["num_partitions"] == 2
    if index_type == "IVF_PQ":
        assert stats["indices"][0]["sub_index"]["num_sub_vectors"] == 4
    assert replacement.search('"Alpha beta"', query_type="fts").limit(1).to_list()
    assert not replacement.search('"Alpha gamma"', query_type="fts").limit(1).to_list()
    assert replacement.search().where("doc_id = '3'").to_list()[0]["doc_id"] == "3"
