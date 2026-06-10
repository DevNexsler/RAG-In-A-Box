import pytest

from integrations.local_deep_research.rag_in_a_box_retriever import (
    RagInABoxClient,
    RagInABoxError,
)


class FakeResponse:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body or {}
        self.text = str(self._body)

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.response


def test_client_posts_search_with_bearer_auth():
    response = FakeResponse(
        body={
            "results": [{"doc_id": "notes/ml.md", "snippet": "ML notes"}],
            "diagnostics": {"degraded": False},
        }
    )
    session = FakeSession(response)
    client = RagInABoxClient(
        "http://rag-box:7788/",
        api_key="secret",
        session=session,
    )

    result = client.search("neural search", top_k=4, folder="notes")

    assert result["results"][0]["doc_id"] == "notes/ml.md"
    assert session.calls == [
        {
            "url": "http://rag-box:7788/api/search",
            "json": {"query": "neural search", "top_k": 4, "folder": "notes"},
            "headers": {"Authorization": "Bearer secret"},
            "timeout": 30.0,
        }
    ]


def test_client_raises_on_rag_error_response():
    response = FakeResponse(
        status_code=400,
        body={"error": True, "code": "empty_query", "message": "Query required"},
    )
    client = RagInABoxClient(
        "http://rag-box:7788",
        session=FakeSession(response),
    )

    with pytest.raises(RagInABoxError) as exc_info:
        client.search("")

    assert "empty_query" in str(exc_info.value)
