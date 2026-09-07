"""scripts/ensure_vector_index.py builds the ANN index under the sweep's writer lock."""

import json
import threading

import pytest
import yaml
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

import scripts.ensure_vector_index as script
from core.index_write_lock import index_write_lock
from lancedb_store import LanceDBStore


def _node(i: int) -> TextNode:
    node = TextNode(
        text=f"text {i}",
        id_=f"d{i}.md::c:0",
        embedding=[float(i == j) for j in range(8)],
        metadata={"doc_id": f"d{i}.md", "source_type": "md", "loc": "c:0", "mtime": 1.0},
    )
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=f"d{i}.md")
    return node


@pytest.fixture
def seeded(tmp_path):
    index_root = tmp_path / "index"
    (tmp_path / "docs").mkdir()
    LanceDBStore(index_root, "chunks").upsert_nodes([_node(i) for i in range(5)])
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "index_root": str(index_root),
                "documents_root": str(tmp_path / "docs"),
                "lancedb": {"table": "chunks"},
                "search": {"vector_index": {"type": "IVF_FLAT"}},
            }
        )
    )
    return index_root, config_path


def _run(capsys, argv) -> tuple[int, dict]:
    code = script.main(argv)
    return code, json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_script_builds_once_then_reports_present(seeded, capsys):
    index_root, config_path = seeded
    code, report = _run(capsys, ["--config", str(config_path), "--check"])
    assert code == 0 and report["vector_index_available"] is False and report["created"] is False

    code, report = _run(capsys, ["--config", str(config_path)])
    assert code == 0 and report["created"] is True and report["vector_index_available"] is True
    assert report["rows"] == 5

    code, report = _run(capsys, ["--config", str(config_path)])
    assert code == 0 and report["created"] is False and report["vector_index_available"] is True
    assert LanceDBStore(index_root, "chunks").vector_index_available() is True


def test_script_no_wait_yields_to_a_running_writer(seeded, capsys):
    """The writer lock is per-thread re-entrant, so the "running writer" must
    be another thread: that is what a peer process's flock looks like from
    inside the script."""
    index_root, config_path = seeded
    result: dict = {}

    def _script_in_other_thread():
        result["code"] = script.main(["--config", str(config_path), "--no-wait"])

    with index_write_lock(index_root, "chunks"):
        worker = threading.Thread(target=_script_in_other_thread)
        worker.start()
        worker.join(10)
    assert not worker.is_alive()
    code = result["code"]
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 3
    assert "already has a writer" in report["error"]
    assert report["created"] is False
    assert LanceDBStore(index_root, "chunks").vector_index_available() is False


def test_script_rebuild_replaces_the_index(seeded, capsys):
    import lance

    index_root, config_path = seeded
    _run(capsys, ["--config", str(config_path)])
    before = [i["uuid"] for i in lance.dataset(f"{index_root}/chunks.lance").list_indices() if "vector" in i["fields"]]
    code, report = _run(capsys, ["--config", str(config_path), "--rebuild"])
    assert code == 0 and report["created"] is True and report["vector_index_available"] is True
    assert "IVF" in str(report["index_type"]).upper()
    after = [i["uuid"] for i in lance.dataset(f"{index_root}/chunks.lance").list_indices() if "vector" in i["fields"]]
    assert before != after and len(after) == 1
