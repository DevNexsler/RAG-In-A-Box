"""Local Deep Research integration helpers."""

from .rag_in_a_box_retriever import (
    RagInABoxClient,
    RagInABoxError,
    build_retriever,
)

__all__ = ["RagInABoxClient", "RagInABoxError", "build_retriever"]
