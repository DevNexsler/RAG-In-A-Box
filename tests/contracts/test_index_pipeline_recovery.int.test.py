"""Public queue drain: persist grounded metadata before ACK; recover process death."""

import json
import os
import subprocess
import sys
import threading

import yaml

from core.index_request_queue import IndexRequestQueue
from lancedb_store import LanceDBStore


def test_queue_drain_crash_preserves_grounded_storage_and_callback(tmp_path, http_peer):
    documents, index = tmp_path / "documents", tmp_path / "index"
    documents.mkdir()
    index.mkdir()
    target = "invoice@abc12@.md"
    source = "Invoice 1234. Paid with Visa ****5678. Payment due Friday."
    (documents / target).write_text(source)
    model_reply = {
        "summary": "Paid by Visa ending in 1234.", "doc_type": ["invoice"],
        "entities_people": [], "entities_places": [], "entities_orgs": [], "entities_dates": [],
        "topics": ["billing"], "keywords": ["invoice"], "key_facts": ["Payment due Friday."],
        "suggested_tags": ["billing"], "suggested_folder": "Finance", "importance": 0.5,
    }
    entered, release = threading.Event(), threading.Event()

    def respond(request):
        if request["path"].endswith("/embeddings"):
            return 200, {"data": [{"index": i, "embedding": [0.1] * 768}
                                  for i, _ in enumerate(request["body"]["input"])]}
        if request["path"].endswith("/chat/completions"):
            return 200, {"choices": [{"message": {"content": json.dumps(model_reply)}}]}
        assert request["path"] == "/document-indexed"
        entered.set()
        assert release.wait(30), "parent failed to release callback peer"
        return 200, {"status": "updated"}

    http_peer.respond = respond
    config = {
        "documents_root": str(documents), "index_root": str(index),
        "embeddings": {"provider": "openrouter", "model": "contract", "api_key": "contract",
                       "base_url": http_peer.url},
        "enrichment": {"enabled": True, "provider": "openrouter", "model": "contract",
                       "api_key": "contract", "base_url": http_peer.url},
        "ocr": {"enabled": False}, "media": {"enabled": False}, "dedupe": {"enabled": False},
        "chunking": {"max_chars": 1800, "overlap": 200, "semantic": {"enabled": False}},
        "lancedb": {"table": "chunks"},
        "event_hooks": {"enabled": True, "hooks": [{"name": "cds", "type": "http",
            "events": ["document.indexed"], "url": http_peer.url + "/document-indexed",
            "accepted_statuses": ["updated", "duplicate"]}]},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    env = {**os.environ, "INDEX_ROOT": str(index), "DOCUMENTS_ROOT": str(documents),
           "OPENROUTER_API_KEY": "contract", "PREFECT_API_URL": "",
           "PREFECT_SERVER_ALLOW_EPHEMERAL_MODE": "true", "PREFECT_HOME": str(tmp_path / "prefect"),
           "PREFECT_SERVER_ANALYTICS_ENABLED": "false"}
    command = [sys.executable, "-c", """
import sys
from flow_index_vault import drain_index_queue
assert drain_index_queue(sys.argv[1]) == {"status": "drained", "drained": 1}
""", str(config_path)]
    queue = IndexRequestQueue(index)
    original = queue.enqueue("chunks", "documents", target)
    with (tmp_path / "worker.log").open("w+") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        try:
            reached_callback = entered.wait(60)
            log.seek(0)
            assert reached_callback, log.read()[-6000:]
            assert queue.pending("chunks", limit=10) == [original], "ACK preceded callback/store commit"
            process.kill()  # callback observed a committed write; queue ACK has not happened
            process.wait(timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            release.set()

    store = LanceDBStore(index, "chunks")
    [doc_id] = store.list_doc_ids()
    chunks = store.get_doc_chunks(doc_id)
    assert chunks and all(chunk.enr_summary == "Paid by Visa ending in 5678." for chunk in chunks)
    [callback] = [r["body"] for r in http_peer.requests if r["path"] == "/document-indexed"]
    assert callback["event"] == "document.indexed"
    assert callback["doc_id"] == doc_id
    assert callback["rel_path"] == target
    assert callback["metadata"]["enr_summary"] == chunks[0].enr_summary
    assert callback["text"] == source
    assert callback["chunks"]

    assert queue.pending("chunks", limit=10) == [original]
    queue.enqueue("chunks", "documents", target, force=True)
    assert queue.complete(original) is False, "old worker must not ACK newer forced work"
    result = subprocess.run(command, env=env, timeout=120, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-6000:]
    recovered = LanceDBStore(index, "chunks")
    assert recovered.list_doc_ids() == [doc_id]
    assert recovered.count_chunks() == len(chunks)
    assert all(chunk.enr_summary == chunks[0].enr_summary for chunk in recovered.get_doc_chunks(doc_id))
    assert queue.pending("chunks", limit=10) == []
