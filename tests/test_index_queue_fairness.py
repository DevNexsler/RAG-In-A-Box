"""Queue fairness during a long historical drain (#1625).

The full sweep takes the table writer lock for its whole run. Everything else
that could serve a targeted request asks for that lock non-blocking, so for as
long as the sweep lasts the durable queue cannot move: on 2026-08-26 a run that
had been going 17h had 2,201 documents `Completed()` and 2,129 chunk writes to
show for itself while 19 newer source requests sat at `attempts=0`, two of them
holding content from that morning. Both facts were true at once because they
describe different work — history draining under the lock, and current work that
never got it.

The reproduction below is that state; the rest cover the sweep serving the queue
at checkpoints between document batches, and the safety properties that made the
post-run drain trustworthy in the first place (bounded work, retained retries,
an intact sweep runtime).
"""

import threading
from unittest.mock import patch

import pytest

import flow_index_vault as fiv
from core.index_request_queue import IndexRequestQueue
from core.index_write_lock import index_write_lock

# The 00:32Z attachments that were still pending at 05:28 on 2026-08-26.
_NEWER_TARGET = "email-attachments/noreply/2026-08-26T00-32-28Z__msg678132__mm0.bin"


def _historical_docs(count: int) -> list[dict]:
    """A sweep's worth of old documents — the work that holds the lock."""
    return [
        {
            "doc_id": f"documents::hist{i:03d}",
            "rel_path": f"archive/2023/{i:03d}.txt",
            "abs_path": f"/data/documents/archive/2023/{i:03d}.txt",
            "mtime": 1.0,
            "size": 10,
            "ext": "txt",
            "source_type": "txt",
            "source_name": "documents",
        }
        for i in range(count)
    ]


@pytest.fixture
def sweep_runtime(tmp_path):
    """_RUNTIME as a running full sweep leaves it: table session and registry."""
    fiv._RUNTIME.clear()
    fiv._RUNTIME.update(
        {
            "store": object(),
            "doc_id_store": object(),
            "index_root": tmp_path,
            "config": {"index_root": str(tmp_path)},
        }
    )
    yield fiv._RUNTIME
    fiv._RUNTIME.clear()


@pytest.fixture
def config(tmp_path):
    return {
        "index_root": str(tmp_path),
        "lancedb": {"table": "chunks"},
        "index_queue": {},
    }


def _indexed(target: str) -> dict:
    return {"status": "indexed", "doc_id": "documents::00new", "rel_path": target}


# ---------------------------------------------------------------------------
# Reproduction: successful task outcomes and starved newer work, together
# ---------------------------------------------------------------------------


def test_long_historical_drain_starves_newer_requests_without_a_checkpoint(
    tmp_path, monkeypatch, config, sweep_runtime
):
    """#1625: 40 documents complete while a request queued behind the sweep's
    writer session stays untouched, and the scheduled drain can only report
    `writer_busy`. Success counts and a stale tail are not in tension — this is
    what produces both."""
    monkeypatch.setattr(fiv, "load_config", lambda _path: config)
    queue = IndexRequestQueue(tmp_path)
    queue.enqueue("chunks", "documents", _NEWER_TARGET)
    completed: list[str] = []
    holding, release = threading.Event(), threading.Event()

    def hold_the_sweeps_writer_session():
        with index_write_lock(tmp_path, "chunks"):
            holding.set()
            release.wait(10)

    holder = threading.Thread(target=hold_the_sweeps_writer_session)
    holder.start()
    try:
        assert holding.wait(5)
        with patch.object(
            fiv, "process_doc_task", lambda doc: completed.append(doc["doc_id"])
        ):
            failed = fiv._process_docs(_historical_docs(40), concurrency=1)
        drain = fiv.drain_index_queue()
    finally:
        release.set()
        holder.join(5)

    assert failed == []
    assert len(completed) == 40, "the historical drain itself succeeds throughout"
    assert drain == {"status": "writer_busy", "drained": 0}
    [starved] = queue.pending("chunks", limit=10)
    assert starved.target == _NEWER_TARGET
    assert starved.attempts == 0, "the request was never even attempted"


# ---------------------------------------------------------------------------
# The fix: the lock holder serves the queue mid-run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("concurrency", [1, 4])
def test_sweep_serves_queued_requests_before_it_finishes(
    tmp_path, config, sweep_runtime, concurrency
):
    """Newer work is served while the sweep is still processing history, on the
    serial and the concurrent path alike."""
    queue = IndexRequestQueue(tmp_path)
    queue.enqueue("chunks", "documents", _NEWER_TARGET)
    docs = _historical_docs(40)
    completed: list[str] = []
    served_after: list[int] = []

    def serve(passed_config, request, store, doc_id_store):
        assert store is sweep_runtime["store"]
        assert doc_id_store is sweep_runtime["doc_id_store"]
        served_after.append(len(completed))
        return _indexed(request.target)

    with patch.object(
        fiv, "process_doc_task", lambda doc: completed.append(doc["doc_id"])
    ), patch.object(fiv, "_index_document_unlocked", side_effect=serve):
        failed = fiv._process_docs(
            docs,
            concurrency=concurrency,
            checkpoint=lambda: fiv._service_index_queue(config, "chunks"),
            checkpoint_every=8,
        )

    assert failed == []
    assert len(completed) == len(docs)
    assert served_after, "the sweep never served the queued request"
    assert served_after[0] < len(docs), (
        "the request was served only after the whole sweep — that is the defect"
    )
    assert queue.pending("chunks", limit=10) == []


def test_sweep_checkpoint_cadence_is_every_configured_batch(
    tmp_path, config, sweep_runtime
):
    """The wait a newer request can inherit is bounded by the batch, not by the
    run: with a checkpoint every 8 documents it is served at document 8."""
    queue = IndexRequestQueue(tmp_path)
    queue.enqueue("chunks", "documents", _NEWER_TARGET)
    completed: list[str] = []
    served_after: list[int] = []

    def serve(passed_config, request, store, doc_id_store):
        served_after.append(len(completed))
        return _indexed(request.target)

    with patch.object(
        fiv, "process_doc_task", lambda doc: completed.append(doc["doc_id"])
    ), patch.object(fiv, "_index_document_unlocked", side_effect=serve):
        fiv._process_docs(
            _historical_docs(40),
            concurrency=1,
            checkpoint=lambda: fiv._service_index_queue(config, "chunks"),
            checkpoint_every=8,
        )

    assert served_after == [8]


def test_sweep_without_a_checkpoint_keeps_the_single_batch_shape(
    tmp_path, sweep_runtime
):
    """A run with no checkpoint (a shadow rebuild) processes exactly as before."""
    completed: list[str] = []

    with patch.object(
        fiv, "process_doc_task", lambda doc: completed.append(doc["doc_id"])
    ):
        failed = fiv._process_docs(_historical_docs(9), concurrency=4)

    assert failed == []
    assert len(completed) == 9


# ---------------------------------------------------------------------------
# Safety: the sweep keeps advancing, and the queue keeps its retry semantics
# ---------------------------------------------------------------------------


def test_sweep_queue_service_is_bounded_per_checkpoint(tmp_path, sweep_runtime):
    """A queue backlog cannot turn one checkpoint into a second sweep."""
    queue = IndexRequestQueue(tmp_path)
    for i in range(10):
        queue.enqueue("chunks", "documents", f"newer{i:02d}.bin")
    config = {"index_root": str(tmp_path), "index_queue": {"sweep_service_limit": 3}}

    with patch.object(
        fiv,
        "_index_document_unlocked",
        side_effect=lambda cfg, request, store, registry: _indexed(request.target),
    ):
        drained = fiv._service_index_queue(config, "chunks")

    assert drained == 3
    assert len(queue.pending("chunks", limit=20)) == 7


def test_sweep_queue_service_retains_a_failed_request_for_retry(
    tmp_path, config, sweep_runtime
):
    """Transient-failure handling is the shared drain's, unchanged: the request
    stays pending with its attempt counted, not dropped and not completed."""
    queue = IndexRequestQueue(tmp_path)
    queue.enqueue("chunks", "documents", "flaky.bin")

    with patch.object(
        fiv,
        "_index_document_unlocked",
        side_effect=RuntimeError("embed circuit open"),
    ):
        assert fiv._service_index_queue(config, "chunks") == 1

    [retained] = queue.pending("chunks", limit=10)
    assert retained.attempts == 1
    assert "embed circuit open" in retained.last_error


def test_sweep_queue_service_restores_the_sweep_runtime(
    tmp_path, config, sweep_runtime
):
    """_index_document_unlocked rebuilds _RUNTIME for its one document. The
    sweep's runtime has to come back, or every document after the checkpoint
    would be indexed against the single-document one."""
    queue = IndexRequestQueue(tmp_path)
    queue.enqueue("chunks", "documents", _NEWER_TARGET)
    sweep_store = sweep_runtime["store"]

    def rebuild_runtime_for_one_document(passed_config, request, store, registry):
        fiv._RUNTIME.clear()
        fiv._RUNTIME.update({"store": object(), "config": passed_config})
        return _indexed(request.target)

    with patch.object(
        fiv, "_index_document_unlocked", side_effect=rebuild_runtime_for_one_document
    ):
        assert fiv._service_index_queue(config, "chunks") == 1

    assert fiv._RUNTIME["store"] is sweep_store
    assert fiv._RUNTIME["index_root"] == tmp_path


def test_sweep_queue_service_failure_never_sinks_the_run(
    tmp_path, config, sweep_runtime
):
    """A run that is hours old must survive an unreadable queue."""
    with patch.object(
        fiv, "IndexRequestQueue", side_effect=OSError("index-requests.sqlite3 gone")
    ):
        assert fiv._service_index_queue(config, "chunks") == 0

    assert fiv._RUNTIME["store"] is sweep_runtime["store"]


def test_sweep_queue_service_is_inert_without_a_writer_runtime(tmp_path, config):
    """No table session, nothing to serve with — and nothing to blow up."""
    fiv._RUNTIME.clear()
    try:
        assert fiv._service_index_queue(config, "chunks") == 0
    finally:
        fiv._RUNTIME.clear()


def test_sweep_queue_service_is_cheap_when_the_queue_is_empty(
    tmp_path, config, sweep_runtime
):
    """The checkpoint runs on every batch of a multi-hour run; an empty queue
    must not reach the drain at all."""
    IndexRequestQueue(tmp_path)

    with patch.object(fiv, "_drain_index_requests") as drain:
        assert fiv._service_index_queue(config, "chunks") == 0

    drain.assert_not_called()
