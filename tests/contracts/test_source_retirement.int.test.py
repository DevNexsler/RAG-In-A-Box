"""Real indexing runs preserve retirement evidence when a source was not scanned."""

import json
import os
import subprocess
import sys

import pytest
import yaml

from doc_id_store import DocIDStore
from lancedb_store import LanceDBStore


@pytest.mark.parametrize("mode", ["failed_scan", "scoped_scan", "healthy_scan"])
def test_retirement_requires_successful_scan_of_own_source(tmp_path, http_peer, mode):
    alpha, beta, index = (tmp_path / name for name in ("alpha", "beta", "index"))
    for path in (alpha, beta, index):
        path.mkdir()
    (alpha / "alpha.md").write_text("Alpha source invoice is due Friday.")
    (beta / "beta.md").write_text("Beta source appointment is on Tuesday.")
    http_peer.respond = lambda request: (200, {"data": [
        {"index": i, "embedding": [0.1] * 768}
        for i, _ in enumerate(request["body"]["input"])
    ]})
    config = {
        "index_root": str(index),
        "sources": [{"type": "filesystem", "name": name, "root": str(root),
                     "scan": {"include": ["**/*.md"], "exclude": []}}
                    for name, root in (("alpha", alpha), ("beta", beta))],
        "embeddings": {"provider": "openrouter", "model": "contract-model",
                       "api_key": "contract-only", "base_url": http_peer.url},
        "enrichment": {"enabled": False}, "ocr": {"enabled": False},
        "media": {"enabled": False}, "dedupe": {"enabled": False},
        "chunking": {"max_chars": 1800, "overlap": 200, "semantic": {"enabled": False}},
        "lancedb": {"table": "chunks"}, "pdf": {}, "logging": {"level": "INFO"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    env = {**os.environ, "PREFECT_API_URL": "", "PREFECT_SERVER_ALLOW_EPHEMERAL_MODE": "true",
           "PREFECT_HOME": str(tmp_path / "prefect"), "PREFECT_SERVER_ANALYTICS_ENABLED": "false",
           "OPENROUTER_API_KEY": "contract-only", "INDEX_ROOT": str(index)}

    def run(run_mode):
        result = subprocess.run([sys.executable, "-c", """
import os, sys
from pathlib import Path
from flow_index_vault import index_vault_flow
config, alpha, mode = sys.argv[1:]
if mode == "failed_scan":
    real_walk = os.walk
    def interrupted_walk(root, *args, **kwargs):
        yield from real_walk(root, *args, **kwargs)
        if Path(root) == Path(alpha):
            raise OSError("filesystem unavailable before scan completed")
    os.walk = interrupted_walk
index_vault_flow(config, source_name="beta" if mode == "scoped_scan" else None)
""", str(config_path), str(alpha), run_mode], env=env, timeout=120, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr[-7000:]

    run("healthy_scan")
    baseline = set(LanceDBStore(index, "chunks").list_doc_ids())
    assert len(baseline) == 2
    registry = DocIDStore(index / "doc_registry.db")
    for identifier in ("alpha::retired-active", "alpha::retired-terminal"):
        registry.register(identifier, identifier + ".md", source_name="alpha")
        registry.delete(identifier)
        assert registry.is_retired(identifier)
    registry.close()
    active = {"version": 2, "docs": {"alpha::retired-active": {
        "reasons": ["enrichment_failed"], "attempts": 1,
    }}}
    terminal = {"version": 1, "docs": {"alpha::retired-terminal": {
        "reasons": ["enrichment_failed"], "escalated_at": 1000,
    }}}
    (index / "degraded_docs.json").write_text(json.dumps(active))
    (index / "degraded_unresolved.json").write_text(json.dumps(terminal))
    (beta / "new.md").write_text("New healthy-source work arrived while alpha was unavailable.")
    run(mode)

    after = set(LanceDBStore(index, "chunks").list_doc_ids())
    assert baseline <= after, "one source's retirement must not delete another source's documents"
    assert len(after) == 3, "healthy source must keep making progress"
    for filename, before in (("degraded_docs.json", active), ("degraded_unresolved.json", terminal)):
        saved = json.loads((index / filename).read_text())["docs"]
        assert saved == ({} if mode == "healthy_scan" else before["docs"])
    if mode != "healthy_scan":
        run("healthy_scan")
        for filename in ("degraded_docs.json", "degraded_unresolved.json"):
            assert json.loads((index / filename).read_text())["docs"] == {}
