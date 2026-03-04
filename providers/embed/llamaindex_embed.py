"""LlamaIndex-based embedding provider. Wraps GoogleGenAIEmbedding (or any LlamaIndex embed model)."""

import os
from typing import Any

from providers.embed.base import EmbedProvider


class LlamaIndexEmbedProvider(EmbedProvider):
    """Wrap any LlamaIndex BaseEmbedding. Default: GoogleGenAIEmbedding (Gemini)."""

    def __init__(self, provider: str = "gemini", model: str = "gemini-embedding-001") -> None:
        self.provider = provider
        self.model = model
        self._embed_model = self._build_model()

    def _build_model(self) -> Any:
        if self.provider == "gemini":
            try:
                from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
            except ImportError:
                raise RuntimeError(
                    "Install llama-index-embeddings-google-genai: "
                    "pip install llama-index-embeddings-google-genai"
                )
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is not set")
            return GoogleGenAIEmbedding(model_name=self.model, api_key=api_key)

        elif self.provider == "local_ollama":
            try:
                from llama_index.embeddings.ollama import OllamaEmbedding
            except ImportError:
                raise RuntimeError(
                    "Install llama-index-embeddings-ollama: "
                    "pip install llama-index-embeddings-ollama"
                )
            return OllamaEmbedding(model_name=self.model)

        elif self.provider == "local_sentence_transformers":
            try:
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            except ImportError:
                raise RuntimeError(
                    "Install llama-index-embeddings-huggingface: "
                    "pip install llama-index-embeddings-huggingface"
                )
            return HuggingFaceEmbedding(model_name=self.model)

        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts (for indexing). Uses LlamaIndex batch embedding."""
        if not texts:
            return []
        return [self._embed_model.get_text_embedding(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query (for search). Uses LlamaIndex query embedding."""
        return self._embed_model.get_query_embedding(query)
