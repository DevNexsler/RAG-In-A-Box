"""Embedding provider interface. Wraps LlamaIndex embedding models so callers stay decoupled."""

from abc import ABC, abstractmethod


class EmbedProvider(ABC):
    """Interface: embed(texts) -> list[list[float]]. Callers don't import LlamaIndex directly."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text. Same order as input."""
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string. Used for search."""
        ...
