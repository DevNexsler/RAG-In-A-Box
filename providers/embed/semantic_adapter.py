"""Embedding adapter for semantic chunking.

Wraps any EmbedProvider as a LlamaIndex BaseEmbedding so it can be used with
SemanticSplitterNodeParser.  Reuses the same embedding provider already
configured for retrieval — no additional model or server needed.
"""

import logging
from typing import Any

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding

logger = logging.getLogger(__name__)


class SemanticEmbeddingAdapter(BaseEmbedding):
    """LlamaIndex BaseEmbedding backed by any EmbedProvider.

    Used exclusively by SemanticSplitterNodeParser for topic-boundary
    detection during chunking.  Delegates all embedding calls to the
    existing EmbedProvider instance (no instruction prefix — raw
    text embeddings for similarity comparison).
    """

    _provider: Any = PrivateAttr(default=None)

    def __init__(self, provider: Any, **kwargs: Any) -> None:
        super().__init__(model_name=getattr(provider, "model_name", "embedding"), **kwargs)
        self._provider = provider
        logger.info("SemanticEmbeddingAdapter: using %s for semantic chunking", self.model_name)

    @classmethod
    def class_name(cls) -> str:
        return "SemanticEmbeddingAdapter"

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._provider.embed_texts([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._provider.embed_texts(texts)

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)
