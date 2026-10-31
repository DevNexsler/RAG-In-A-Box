"""Carry MCP cancellation into synchronous search worker threads."""

from contextvars import ContextVar
from threading import Event


rerank_cancel_event: ContextVar[Event | None] = ContextVar("rerank_cancel_event", default=None)
