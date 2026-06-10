"""LangChain retriever adapter for Local Deep Research.

Use this from an LDR Python process to search the canonical RAG-in-a-Box index
through its HTTP `/api/search` endpoint.
"""

from __future__ import annotations

from typing import Any

import requests


class RagInABoxError(RuntimeError):
    """Raised when RAG-in-a-Box search returns an error."""


class RagInABoxClient:
    """Small HTTP client for RAG-in-a-Box search."""

    def __init__(
        self,
        base_url: str = "http://localhost:7788",
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.session = session or requests.Session()

    def search(self, query: str, top_k: int = 10, **filters: Any) -> dict[str, Any]:
        """Run semantic search against RAG-in-a-Box."""
        payload = {"query": query, "top_k": top_k}
        payload.update({key: value for key, value in filters.items() if value is not None})

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = self.session.post(
            f"{self.base_url}/api/search",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )

        try:
            body = response.json()
        except ValueError as exc:
            raise RagInABoxError(
                f"RAG-in-a-Box returned non-JSON response ({response.status_code}): {response.text[:200]}"
            ) from exc

        if response.status_code >= 400 or body.get("error"):
            code = body.get("code", response.status_code)
            message = body.get("message", response.text[:200])
            raise RagInABoxError(f"{code}: {message}")

        return body


def _result_to_document(result: dict[str, Any], document_cls: type) -> Any:
    text = result.get("text") or result.get("snippet") or ""
    metadata = {
        "source": result.get("doc_id") or result.get("rel_path") or "rag-in-a-box",
        "title": result.get("title") or result.get("doc_id") or "RAG-in-a-Box result",
        "score": result.get("score"),
        "loc": result.get("loc"),
        "source_type": result.get("source_type"),
        "folder": result.get("folder"),
        "tags": result.get("tags", []),
        "rag_box": result,
    }
    return document_cls(page_content=text, metadata=metadata)


def build_retriever(
    base_url: str = "http://localhost:7788",
    *,
    api_key: str | None = None,
    top_k: int = 10,
    timeout: float = 30.0,
    **default_filters: Any,
) -> Any:
    """Build a LangChain BaseRetriever for LDR's `retrievers={...}` API.

    This imports LangChain lazily so RAG-in-a-Box can run without LangChain
    installed. The target LDR environment already provides these packages.
    """
    try:
        from langchain_core.documents import Document
        from langchain_core.retrievers import BaseRetriever
        from pydantic import ConfigDict, Field
    except ImportError as exc:
        raise RagInABoxError(
            "LangChain adapter requires langchain-core and pydantic in the LDR environment"
        ) from exc

    class RagInABoxRetriever(BaseRetriever):
        client: RagInABoxClient
        top_k: int = top_k
        default_filters: dict[str, Any] = Field(default_factory=lambda: dict(default_filters))

        model_config = ConfigDict(arbitrary_types_allowed=True)

        def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Any]:
            body = self.client.search(query, top_k=self.top_k, **self.default_filters)
            return [_result_to_document(item, Document) for item in body.get("results", [])]

    return RagInABoxRetriever(
        client=RagInABoxClient(base_url, api_key=api_key, timeout=timeout),
        top_k=top_k,
    )
