"""Shared types and config for the indexer pipeline and MCP."""

from core.config import load_config
from core.storage import StorageInterface, SearchHit

__all__ = ["load_config", "StorageInterface", "SearchHit"]
