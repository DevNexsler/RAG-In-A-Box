# Local Deep Research Integration

Goal: use Local Deep Research (LDR) for agentic research while keeping RAG-in-a-Box as the canonical private knowledge base.

## Architecture

```
LDR research strategy
  -> LangChain retriever: rag_box
  -> RAG-in-a-Box HTTP POST /api/search
  -> LanceDB hybrid search (vector + BM25 + reranker + filters)
```

Do not build a second long-lived LDR document library for the same corpus. Use LDR's own library only for throwaway collections or sources you have not accepted into RAG-in-a-Box yet.

## RAG-in-a-Box

Run the existing service:

```bash
docker compose up -d doc-organizer
```

Search endpoint:

```bash
curl -X POST http://localhost:7788/api/search \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"deployment procedures","top_k":5}'
```

Response shape matches `file_search`:

```json
{
  "results": [
    {
      "doc_id": "ops/deployment.md",
      "loc": "c:0",
      "snippet": "...",
      "score": 0.42,
      "title": "Deployment"
    }
  ],
  "diagnostics": {"degraded": false}
}
```

## LDR Programmatic Use

Install or clone LDR separately, then make this repo importable from the same Python process:

```bash
export PYTHONPATH="/home/danpark/projects/RAG-in-a-Box:$PYTHONPATH"
```

Example:

```python
import os

from local_deep_research.api import quick_summary
from integrations.local_deep_research import build_retriever

rag_box = build_retriever(
    base_url="http://localhost:7788",
    api_key=os.environ.get("API_KEY"),
    top_k=8,
)

result = quick_summary(
    query="What do our docs say about production deploy rollback?",
    retrievers={"rag_box": rag_box},
    search_tool="rag_box",
)

print(result["summary"])
```

Hybrid private + web research:

```python
result = quick_summary(
    query="Compare our deploy rollback process to current SRE guidance",
    retrievers={"rag_box": rag_box},
    search_tool="auto",
    search_engines=["rag_box", "searxng", "arxiv"],
)
```

## Accepting LDR Sources Into RAG-in-a-Box

When LDR finds useful PDFs or pages, ingest them into RAG-in-a-Box instead of leaving them only in LDR:

```bash
curl -X POST http://localhost:7788/api/upload \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@paper.pdf" \
  -F "directory=research/inbox"
```

Then run `file_index_update` through MCP, or restart the indexer flow used by your deployment.
