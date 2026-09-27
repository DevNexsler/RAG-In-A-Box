"""Versioned judgments run through real storage and hybrid retrieval."""
from scripts.rag_quality import run_quality


def test_versioned_retrieval_and_grounding_corpus(tmp_path):
    report = run_quality(tmp_path)
    assert report['passed'], report
    assert report['corpus_version'] == 1
    assert len(report['retrieval']) >= 6
    assert len(report['grounding']) >= 4


def test_configured_embedding_adapter_indexes_and_queries_same_vector_space(tmp_path):
    from scripts.rag_quality import LexicalEmbeddings

    class AlternateProvider:
        def embed_query(self, text):
            vector = LexicalEmbeddings().embed_query(text)
            return vector[97:] + vector[:97]

        def embed_texts(self, texts):
            return [self.embed_query(text) for text in texts]

    report = run_quality(tmp_path, embed_provider=AlternateProvider())
    assert report['passed'], report
    assert report['mode'] == 'configured-embedding-provider'


def test_real_provider_query_failure_cannot_pass_using_keyword_fallback(tmp_path):
    from scripts.rag_quality import LexicalEmbeddings

    class BrokenQueryProvider:
        def embed_texts(self, texts):
            return LexicalEmbeddings().embed_texts(texts)

        def embed_query(self, text):
            raise RuntimeError('provider unavailable')

    report = run_quality(tmp_path, embed_provider=BrokenQueryProvider())
    assert not report['passed'], 'provider health must not hide behind lexical relevance'
