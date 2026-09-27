"""Versioned relevance judgments against configured real embeddings (live tier)."""
from core.config import load_config
from providers.embed import build_embed_provider
from scripts.rag_quality import run_quality


def test_real_embeddings_meet_versioned_retrieval_judgments(tmp_path):
    report = run_quality(tmp_path, embed_provider=build_embed_provider(load_config('config_test.yaml')))
    assert report['passed'], report
