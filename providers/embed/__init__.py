"""Embedding providers: wraps Ollama, Baseten, OpenRouter, and LlamaIndex embedding models."""

from __future__ import annotations

import logging

from providers.embed.base import EmbedProvider

logger = logging.getLogger(__name__)

__all__ = ["EmbedProvider", "build_embed_provider"]


def build_embed_provider(config: dict) -> EmbedProvider:
    """Factory: build an EmbedProvider from the embeddings config section.

    Supported providers:
      - "ollama": Qwen3-Embedding via Ollama /v1/embeddings (local)
      - "baseten": Qwen3-Embedding via Baseten-hosted BEI (cloud)
      - "openrouter": Any embedding model via OpenRouter (cloud)
      - "gemini": GoogleGenAIEmbedding via LlamaIndex (cloud)
      - "local_ollama": Ollama via LlamaIndex
      - "local_sentence_transformers": HuggingFace via LlamaIndex

    Config example (openrouter — cloud):
        embeddings:
          provider: "openrouter"
          model: "qwen/qwen3-embedding-8b"
          batch_size: 64
    """
    emb_cfg = config.get("embeddings", {})
    provider = emb_cfg.get("provider", "gemini")

    if provider == "openrouter":
        from providers.embed.openrouter_embed import OpenRouterEmbedProvider
        return OpenRouterEmbedProvider(
            model=emb_cfg.get("model", "qwen/qwen3-embedding-8b"),
            api_key=emb_cfg.get("api_key"),
            query_instruction=emb_cfg.get(
                "query_instruction",
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
            ),
            batch_size=emb_cfg.get("batch_size", 64),
        )

    if provider == "baseten":
        from providers.embed.baseten_embed import BasetenEmbedProvider
        return BasetenEmbedProvider(
            model_id=emb_cfg.get("model_id", ""),
            model_name=emb_cfg.get("model", "model"),
            api_key=emb_cfg.get("api_key"),
            query_instruction=emb_cfg.get(
                "query_instruction",
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
            ),
            batch_size=emb_cfg.get("batch_size", 64),
        )

    if provider == "ollama":
        from providers.embed.ollama_embed import OllamaEmbedProvider
        return OllamaEmbedProvider(
            base_url=emb_cfg.get("base_url", "http://localhost:11434"),
            model_name=emb_cfg.get("model", "qwen3-embedding:4b-q8_0"),
            query_instruction=emb_cfg.get(
                "query_instruction",
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
            ),
            batch_size=emb_cfg.get("batch_size", 64),
        )

    # For gemini, local_ollama, local_sentence_transformers — use LlamaIndex wrapper
    from providers.embed.llamaindex_embed import LlamaIndexEmbedProvider
    return LlamaIndexEmbedProvider(
        provider=provider,
        model=emb_cfg.get("model", "gemini-embedding-001"),
    )
