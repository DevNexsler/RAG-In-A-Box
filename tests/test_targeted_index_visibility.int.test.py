# Single-doc indexing serving-visibility regression (TICKET-6 / staging-gate e2e).
#
# Production bug: index_document_flow upserts into the Lance table but never
# writes index_metadata.json, so mcp_server._get_deps — which invalidates its
# cached LanceDB handle ONLY when that file's (mtime, size) signature changes —
# keeps serving a handle pinned to the pre-index table version. The freshly
# indexed doc stays invisible to file_search / file_get_chunk /
# file_list_documents until the next full sweep, breaking the documented
# "searchable within seconds" contract.
#
# This test exercises the REAL pieces end to end in-process: a real
# LanceDBStore on disk, the real index_document_flow (mock embeddings, no
# providers), and the real mcp_server._get_deps caching layer.

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import flow_index_vault as fiv
import mcp_server


def _config(root: Path, index_root: Path) -> dict:
    return {
        "index_root": str(index_root),
        "documents_root": str(root),
        "lancedb": {"table": "chunks"},
        "pdf": {"strategy": "text_then_ocr"},
        "search": {},
        "sources": [
            {
                "type": "filesystem",
                "name": "documents",
                "root": str(root),
                "scan": {
                    "include": ["**/*.txt"],
                    "exclude": [],
                    "no_rename": ["email-attachments/"],
                },
            }
        ],
    }


def _mock_embed() -> MagicMock:
    embed = MagicMock()
    embed.embed_texts.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
    embed.embed_query.side_effect = lambda q: [0.1] * 768
    return embed


def _run_flow(cfg: dict, target: str) -> dict:
    """Run the real index_document_flow with a REAL store and mock providers."""
    with patch("flow_index_vault.load_config", return_value=cfg), \
         patch("flow_index_vault.build_embed_provider", side_effect=lambda c: _mock_embed()), \
         patch("flow_index_vault.build_ocr_provider", return_value=None), \
         patch("flow_index_vault.build_media_provider", return_value=None), \
         patch("flow_index_vault.dispatch_event", create=True, return_value=[]):
        return fiv.index_document_flow(target=target, source_name="documents")


@pytest.fixture
def serving_cache():
    """Snapshot and restore mcp_server's module-level deps cache."""
    saved = (mcp_server._cache, mcp_server._cache_index_signature, mcp_server._cache_identity)
    mcp_server._cache = None
    mcp_server._cache_index_signature = None
    mcp_server._cache_identity = None
    yield
    (mcp_server._cache, mcp_server._cache_index_signature, mcp_server._cache_identity) = saved


def test_single_doc_index_is_visible_through_cached_serving_deps(tmp_path, serving_cache):
    root = tmp_path / "docs"
    index_root = tmp_path / "index"
    seed = root / "email-attachments" / "seed@00sed@.txt"
    seed.parent.mkdir(parents=True)
    seed.write_bytes(b"seed document about aardvark budgets and quarterly planning")

    cfg = _config(root, index_root)

    # 1. Simulate the state after a full sweep: seed doc indexed through the
    #    same pipeline, then the sweep's index_metadata.json write.
    result = _run_flow(cfg, "email-attachments/seed@00sed@.txt")
    assert result["status"] == "indexed", result
    fiv.write_index_metadata_task.fn(index_root, doc_count=1, chunk_count=1)

    # 2. Serving process boots: _get_deps builds and caches its own store
    #    handle (exactly what file_search & co. read through).
    with patch("mcp_server.load_config", return_value=cfg), \
         patch("mcp_server.build_embed_provider", side_effect=lambda c: _mock_embed()):
        store, _embed, _config_out = mcp_server._get_deps()
        assert "documents::00sed" in store.list_doc_ids()  # sanity: cache serves the sweep state

        # 3. A new attachment is deposited and indexed via the single-doc path
        #    (production: wakeup-bridge -> hooks -> POST /api/index/document,
        #    or the MCP file_index_document tool).
        fresh = root / "email-attachments" / "note@00vis@.txt"
        fresh.write_bytes(b"the xylophone glacier permit was approved yesterday")
        result = _run_flow(cfg, "email-attachments/note@00vis@.txt")
        assert result["status"] == "indexed", result
        # "00vis" is a producer-minted token the registry never issued, so the
        # deposit is adjudicated a fresh identity instead (#0390); the
        # visibility contract holds for whatever ID the flow assigned.
        fresh_ns = result["doc_id"]
        assert fresh_ns.startswith("documents::")
        assert fresh_ns != "documents::00vis"

        # 4. Documented contract: the doc must be visible to the serving reads
        #    immediately — through the SAME cached-deps path the tools use.
        store, _embed, _config_out = mcp_server._get_deps()
        assert fresh_ns in store.list_doc_ids(), (
            "single-doc indexed document invisible to the serving store handle: "
            "_get_deps served a stale cached LanceDB handle (index_metadata.json "
            "was never touched by index_document_flow)"
        )

        payload = mcp_server._file_search_impl("xylophone glacier permit", top_k=5)
        assert not payload.get("error"), payload
        assert any(r["doc_id"] == fresh_ns for r in payload["results"]), payload["results"]
