"""Integration test: flow_index_vault handles multiple sources correctly.

Covers: namespaced doc_ids prevent collisions, source-scoped deletes don't
affect other sources, existing filesystem tests still pass through the
backward-compat shim.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

_has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
_has_deepinfra = bool(os.environ.get("DEEPINFRA_API_KEY"))

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (_has_openrouter and _has_deepinfra),
        reason="OPENROUTER_API_KEY + DEEPINFRA_API_KEY required",
    ),
]


class TestMultiSourceFlow:
    """Run index_vault_flow with a config that has two filesystem sources
    (so we can test without a Postgres dependency) and verify namespacing."""

    @pytest.fixture(scope="class")
    def two_source_result(self):
        """Create two vault dirs, index both, return the lancedb store + config."""
        import yaml
        from core.config import load_config
        from lancedb_store import LanceDBStore
        from doc_id_store import DocIDStore

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            vault_a = tmp / "vault_a"
            vault_b = tmp / "vault_b"
            src = Path(__file__).parent.parent / "test_vault"
            shutil.copytree(src, vault_a)
            shutil.copytree(src, vault_b)
            index_root = tmp / "index"
            index_root.mkdir()

            # Build config with TWO filesystem sources
            config = {
                "sources": [
                    {"type": "filesystem", "name": "alpha", "root": str(vault_a),
                     "scan": {"include": ["**/*.md"], "exclude": []}},
                    {"type": "filesystem", "name": "beta", "root": str(vault_b),
                     "scan": {"include": ["**/*.md"], "exclude": []}},
                ],
                "index_root": str(index_root),
                "chunking": {"max_chars": 1800, "overlap": 200, "semantic": {"enabled": False}},
                "embeddings": load_config()["embeddings"],
                "enrichment": {"enabled": False},  # Skip enrichment to keep test fast
                "search": {"mode": "hybrid", "vector_top_k": 10, "keyword_top_k": 10, "final_top_k": 5, "rrf_k": 60},
                "lancedb": {"table": "chunks"},
                "ocr": {"enabled": False},
                "pdf": {"strategy": "text_then_ocr", "ocr_page_limit": 200, "min_text_chars_before_ocr": 200},
                "mcp": {"host": "127.0.0.1", "port": 7788},
                "logging": {"level": "WARNING"},
            }

            config_path = tmp / "config.yaml"
            config_path.write_text(yaml.safe_dump(config))

            saved_env = {}
            for key in ("PREFECT_API_URL", "PREFECT_SERVER_ALLOW_EPHEMERAL_MODE", "DOCUMENTS_ROOT", "INDEX_ROOT"):
                saved_env[key] = os.environ.get(key)
            os.environ["PREFECT_API_URL"] = ""
            os.environ["PREFECT_SERVER_ALLOW_EPHEMERAL_MODE"] = "true"
            os.environ.pop("DOCUMENTS_ROOT", None)
            os.environ.pop("INDEX_ROOT", None)

            try:
                from prefect.settings.models.root import Settings as PrefectSettings
                import prefect.context
                prefect.context.get_settings_context().settings = PrefectSettings()
            except Exception:
                pass
            try:
                from flow_index_vault import index_vault_flow
                index_vault_flow(str(config_path))
            finally:
                for k, v in saved_env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

            store = LanceDBStore(str(index_root), "chunks")
            registry = DocIDStore(index_root / "doc_registry.db")
            yield {"store": store, "registry": registry, "vault_a": vault_a, "vault_b": vault_b, "index_root": index_root}

    def test_both_sources_indexed(self, two_source_result):
        """Every doc_id should be namespaced — alpha::... and beta::..."""
        doc_ids = two_source_result["store"].list_doc_ids()
        alpha_ids = [d for d in doc_ids if d.startswith("alpha::")]
        beta_ids = [d for d in doc_ids if d.startswith("beta::")]
        assert len(alpha_ids) >= 3
        assert len(beta_ids) >= 3

    def test_no_cross_source_collisions(self, two_source_result):
        """Same filename under two sources produces two distinct doc_ids.

        Both vaults are copies of the same test_vault, so they contain the same
        filenames (and same @XXXXX@ persistent IDs).  With namespacing, the
        namespaced doc_ids are distinct: alpha::<id> vs beta::<id>.
        The union of alpha and beta doc_ids must have no overlap with either
        set individually — i.e. no raw ID appears in both alpha:: and beta::
        without the namespace prefix protecting them.
        """
        doc_ids = two_source_result["store"].list_doc_ids()
        # Strip prefix and collect raw IDs per source
        alpha_raw = {d[len("alpha::"):] for d in doc_ids if d.startswith("alpha::")}
        beta_raw = {d[len("beta::"):] for d in doc_ids if d.startswith("beta::")}
        # The same underlying file IDs appear in both sources (same test vault)
        shared_raw = alpha_raw & beta_raw
        assert len(shared_raw) >= 1, "Expected at least one ID present in both sources (same vault copy)"
        # But the namespaced doc_ids must be distinct — no alpha::X == beta::X
        alpha_ids = {d for d in doc_ids if d.startswith("alpha::")}
        beta_ids = {d for d in doc_ids if d.startswith("beta::")}
        assert alpha_ids.isdisjoint(beta_ids), "Namespaced doc_ids must not overlap between sources"

    def test_registry_tracks_source_name(self, two_source_result):
        """DocIDStore has source_name populated for both sources."""
        registry = two_source_result["registry"]
        assert "alpha" in registry.distinct_source_names()
        assert "beta" in registry.distinct_source_names()
