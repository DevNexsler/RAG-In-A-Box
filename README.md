# RAG In A Box

Drop your documents into a folder, run the indexer, and get a production-grade RAG pipeline with an **MCP server** — any MCP-compatible AI assistant (Claude Code, OpenClaw, Claude Desktop, Cursor, etc.) can search your documents with a single config entry.

No infrastructure to manage. No GPU required. Works with **cloud APIs** out of the box or **fully self-hosted**.

## Use cases

- **Personal knowledge base** — Index your notes, PDFs, and documents. Ask your AI assistant questions and get answers grounded in your own files.
- **Company document search** — Drop legal contracts, reports, SOPs into a folder. Employees search via any MCP-compatible assistant with metadata filters (by department, doc type, date, tags).
- **Research assistant** — Index papers, datasets, and notes. Search by meaning, not just keywords. LLM enrichment auto-extracts entities, topics, and key facts.
- **Obsidian / Markdown vault** — Works with any markdown source (Obsidian, HackMD, Notion exports, GitBook). Extracts YAML frontmatter for rich filtering.
- **PDF-heavy workflows** — Scanned PDFs get OCR automatically. Page-aware chunking keeps context intact. Metadata (author, dates, page count) extracted from PDF properties.
- **Multi-agent tool** — Expose your document collection as 16 MCP tools. Multiple agents can search, browse, filter, and manage taxonomy concurrently.

## Why this over other RAG tools?

| Capability | RAG In A Box | Typical RAG |
|---|---|---|
| Search quality | 10-step hybrid pipeline (vector + BM25 + reranker + MMR) | Vector-only or basic hybrid |
| Document understanding | LLM enrichment extracts summary, entities, topics, importance | Raw chunks, no enrichment |
| Filtering | Pre-filter by tags, folder, doc type, topics, custom fields | Post-filter or none |
| Chunking | Heading-aware (MD) + page-aware (PDF) + semantic boundary detection | Fixed-size windows |
| Chunk context | Each chunk gets title, path, topics prepended for self-describing retrieval | Chunks lose document context |
| Metadata | YAML frontmatter auto-extracted, custom fields auto-promoted to filters | Manual schema setup |
| Taxonomy | Controlled vocabulary with semantic matching, managed via MCP tools | None |
| OCR | Built-in for scanned PDFs and images (cloud or local) | Separate pipeline needed |
| Deployment | Single container, cloud APIs, no GPU | Often needs GPU or complex infra |
| Integration | MCP server (16 tools) — works with Claude, Cursor, any MCP client | Custom API or SDK |
| Resilience | Per-query diagnostics, auto-recovery from DB corruption, structured errors | Silent failures |

## Stack

| Component | Provider |
|---|---|
| Embeddings | Qwen3-Embedding-8B via OpenRouter |
| LLM enrichment | GPT-4.1 Mini via OpenRouter |
| OCR | Gemini Vision (cloud) or DeepSeek OCR2 (local) |
| Reranker | Qwen3-Reranker-8B via DeepInfra |
| Vector + FTS | LanceDB + tantivy (BM25) |
| Orchestration | Prefect 3.x |

## Getting started


### 1. Install

```bash
git clone https://github.com/DevNexsler/RAG-In-A-Box.git
cd RAG-In-A-Box
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

Copy the example config and edit two paths:

```bash
cp config.yaml.example config.yaml   # cloud providers (default)
```

Open `config.yaml` and set:
- `documents_root` — path to your document collection
- `index_root` — where the index will be stored

> **Self-hosting?** Use `cp config.local.yaml.example config.yaml` instead. This config uses Ollama, DeepSeek OCR2, and llama-server — see [Local mode](#local-mode-optional) below.

### 3. Add API keys

Create a `.env` file in the project root:

```bash
GEMINI_API_KEY=...               # OCR — get one at https://aistudio.google.com/apikey
OPENROUTER_API_KEY=sk-or-...     # embeddings + enrichment — https://openrouter.ai/keys
DEEPINFRA_API_KEY=...            # reranker — https://deepinfra.com/dash/api_keys
```

### 4. Build the index

```bash
python run_index.py
```

This scans your documents, extracts text (Markdown, PDFs, images), generates embeddings, and writes everything to a LanceDB index. Prefect auto-starts a temporary server for flow/task logging — dashboard at `http://127.0.0.1:4200`.

### 5. Connect your AI assistant

The MCP server gives any compatible AI assistant access to your documents via tools like `file_search`, `file_status`, and `file_recent`. The assistant launches the server automatically — you just add a config entry.

#### Claude Code

Add to your project's `.mcp.json` (or `~/.claude.json` for global access):

```json
{
  "mcpServers": {
    "doc-organizer": {
      "command": "/path/to/Document-Organizer/.venv/bin/python",
      "args": ["/path/to/Document-Organizer/mcp_server.py"],
      "cwd": "/path/to/Document-Organizer"
    }
  }
}
```

#### OpenClaw

Add to your OpenClaw MCP config:

```json
{
  "mcpServers": {
    "doc-organizer": {
      "command": "/path/to/Document-Organizer/.venv/bin/python",
      "args": ["mcp_server.py"],
      "cwd": "/path/to/Document-Organizer"
    }
  }
}
```

#### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "doc-organizer": {
      "command": "/path/to/Document-Organizer/.venv/bin/python",
      "args": ["/path/to/Document-Organizer/mcp_server.py"],
      "cwd": "/path/to/Document-Organizer"
    }
  }
}
```

#### Any MCP-compatible client

The pattern is the same for Cursor, Windsurf, or any tool that supports MCP stdio servers. Point `command` at the venv Python and `args` at `mcp_server.py`. API keys are loaded from the `.env` file automatically — no need to pass them in the MCP config.

#### HTTP mode (remote / non-MCP clients)

```bash
python mcp_server.py --http
# Listens on 0.0.0.0:7788
```

#### VPS / Docker deployment

Run as a standalone HTTP server on any VPS or container platform. All ML inference uses cloud APIs — no GPU needed.

```bash
# Docker
docker build -t doc-organizer .
docker run -v /path/to/data:/data -p 7788:7788 \
  -e OPENROUTER_API_KEY=... \
  -e DEEPINFRA_API_KEY=... \
  -e API_KEY=your-secret-token \
  doc-organizer

# Or run directly
API_KEY=your-secret-token python server.py
```

**Environment variable overrides** (for container/VPS use):

| Variable | Description |
|----------|-------------|
| `DOCUMENTS_ROOT` | Override documents path (default: from config.yaml) |
| `INDEX_ROOT` | Override index path (default: from config.yaml) |
| `PORT` | Server port (default: 7788) |
| `API_KEY` | Bearer token for HTTP auth. No auth when unset. |

When `API_KEY` is set, all HTTP requests must include `Authorization: Bearer <API_KEY>`. See `config.vps.yaml.example` for VPS-specific config.

**Render.com:** One-click deploy with `render.yaml` — persistent disk at `/data`, auto-generated API key.

#### REST API (file management)

When running in HTTP mode (`--http` or `server.py`), a REST API is available alongside the MCP server for uploading, downloading, and listing documents. Auth uses the same `API_KEY` bearer token.

**Upload a file:**
```bash
curl -X POST http://localhost:7788/api/upload \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@report.pdf" \
  -F "directory=2-Area/Legal"
# -> {"uploaded": true, "doc_id": "2-Area/Legal/report.pdf", "size": 84521}
```

**Download a file:**
```bash
curl http://localhost:7788/api/documents/2-Area/Legal/report.pdf \
  -H "Authorization: Bearer $API_KEY" -o report.pdf
```

**List files in a directory:**
```bash
curl "http://localhost:7788/api/documents/?directory=2-Area&limit=50" \
  -H "Authorization: Bearer $API_KEY"
# -> {"directory": "2-Area", "files": [...], "total": 12, "offset": 0, "limit": 50}
```

| Endpoint | Method | Description |
|---|---|---|
| `/api/upload` | POST | Upload a file (multipart form: `file` + optional `directory`) |
| `/api/documents/{doc_id}` | GET | Download a file by path |
| `/api/documents/` | GET | List files (query params: `directory`, `limit`, `offset`) |

**Constraints:** Max upload 100 MB. Allowed types: `.md`, `.pdf`, `.png`, `.jpg`, `.jpeg`. Path traversal is blocked. After uploading, run `file_index_update` (via MCP) to index the new document.

### Local mode (optional)

To run everything on your own hardware instead of cloud APIs:

1. Copy `config.local.yaml.example` to `config.yaml`
2. Install and start **Ollama**: `brew install ollama && ollama serve`
   - `ollama pull qwen3-embedding:0.6b` (semantic chunking)
   - `ollama pull qwen3-embedding:4b-q8_0` (embeddings)
3. Start **DeepSeek OCR2** on port 8790 (for PDF/image OCR)

No cloud API keys needed in local mode (except reranker — DeepInfra is always cloud).

## Features

```
Document Collection                    AI Assistants
 +------------------+                   +-------------------+
 | Markdown (.md)   |                   | Claude Code       |
 | PDFs             |    +---------+    | OpenClaw          |
 | Images (.png/jpg)|───>| Indexer |    | Claude Desktop    |
 +------------------+    +----+----+    | Cursor / Windsurf |
                              |         +--------+----------+
                              v                  |
                   +----------+----------+       | MCP (stdio)
                   |     LanceDB Index   |       |
                   |  vectors + metadata |<------+
                   |  + full-text (BM25) |  file_search
                   +---------------------+  file_status
                                            file_recent ...
```

**Hybrid search** — Every query runs vector (semantic) and keyword (BM25) search in parallel, fuses results with Reciprocal Rank Fusion, applies length normalization, importance weighting, optional recency boost with time decay floor, cross-encoder reranking (60/40 blend with cosine fallback), MMR diversity filtering, and minimum score thresholding. Pre-filters (tags, folders, doc type, topics) apply at the database level before retrieval so every result matches.

**Multi-format extraction** — Indexes Markdown, PDFs, and images. PDFs use text extraction first, falling back to OCR for scanned pages. Images get OCR text plus visual descriptions. EXIF metadata (camera, GPS, dates) is extracted automatically.

**LLM enrichment** — Each document is analyzed by an LLM to extract structured metadata: summary, document type, entities (people, places, orgs, dates), topics, keywords, key facts, suggested tags, and suggested folder. All fields are searchable and filterable.

**Taxonomy system** — A controlled vocabulary of tags and folder paths stored in a separate LanceDB table with embedded descriptions. The LLM uses the taxonomy during enrichment to suggest consistent tags and filing locations. Seeded from existing tag/directory databases. Managed via 7 MCP CRUD tools (`file_taxonomy_*`).

**Smart chunking** — Markdown is split by headings, PDFs by pages. Large sections get semantic chunking (topic-boundary detection via sentence embeddings). Every chunk gets a contextual header prepended with its title, path, and topics — so each chunk is self-describing for better retrieval.

**Rich metadata & filtering** — YAML frontmatter (tags, status, author, dates, custom fields) is automatically extracted and promoted to filterable columns. Custom frontmatter keys are auto-promoted — no schema changes needed.

**MCP server** — Exposes 16 tools over the Model Context Protocol. Any MCP-compatible assistant can search, browse, filter your documents, and manage taxonomy entries. Works over stdio (launched automatically by the assistant) or HTTP.

**Incremental updates** — Only new and modified files are processed on re-index. Deleted files are cleaned up automatically. Failed documents are tracked and retried.

**Cloud or local** — Every component (OCR, embeddings, enrichment, reranker) has both cloud and local provider options. Default config uses cloud APIs with no servers to run. Switch to fully self-hosted with a single config file swap.

**Resilient by default** — Per-document error handling with retries, structured MCP error responses, search diagnostics on every query (`vector_search_active`, `reranker_applied`, `degraded`), SQL injection protection on filter keys, and automatic LanceDB corruption recovery (version rollback + rebuild).

### MCP tools

| Tool | Description |
|------|-------------|
| `file_search` | Hybrid semantic + keyword search with filters (tags, folder, source_type, doc_type, topics, etc.) |
| `file_get_chunk` | Get full text + metadata for one chunk by doc_id and loc |
| `file_get_doc_chunks` | Get all chunks for a document, sorted by position |
| `file_list_documents` | Browse all indexed documents with pagination and filters |
| `file_recent` | Recently modified/indexed docs (newest first) |
| `file_facets` | Distinct values + counts for all filterable fields |
| `file_folders` | Document folder/directory structure with file counts |
| `file_status` | Index stats, provider settings, health checks |
| `file_index_update` | Incrementally update the index without leaving the assistant |
| `file_taxonomy_list` | List taxonomy entries (tags, folders, doc_types) with filters |
| `file_taxonomy_get` | Get a single taxonomy entry by id |
| `file_taxonomy_search` | Semantic search on taxonomy descriptions |
| `file_taxonomy_add` | Add a new taxonomy entry |
| `file_taxonomy_update` | Update an existing taxonomy entry |
| `file_taxonomy_delete` | Delete a taxonomy entry |
| `file_taxonomy_import` | Import taxonomy from SQLite seed databases |

## Run tests

```bash
python -m pytest tests/ -m "not live" -x    # ~370 offline tests (no API keys)
python -m pytest tests/ -x                   # ~454 full suite (requires API keys)
```

## Project layout

```
core/                        Config, storage interface, taxonomy helpers
providers/embed/             Embedding providers (OpenRouter, Ollama, LlamaIndex)
providers/llm/               LLM providers (OpenRouter, Ollama)
providers/ocr/               OCR providers (Gemini Vision, DeepSeek OCR2)
taxonomy_store.py            Taxonomy LanceDB store (CRUD, vector search, FTS)
doc_enrichment.py            LLM metadata extraction (with taxonomy integration)
extractors.py                Text extraction (MD, PDF, images)
flow_index_vault.py          Prefect indexing flow
lancedb_store.py             LanceDB storage + search
search_hybrid.py             10-step hybrid search pipeline
mcp_server.py                MCP server (stdio + HTTP, 16 tools)
server.py                    VPS entrypoint — starts HTTP server on $PORT
run_index.py                 CLI entrypoint
scripts/seed_taxonomy.py     Import taxonomy from existing SQLite DBs
config.yaml.example          Cloud config template
config.local.yaml.example    Local/self-hosted config template
config.vps.yaml.example      VPS/container config template
Dockerfile                   Docker image (Python 3.13-slim, no GPU)
.dockerignore                Docker build exclusions
render.yaml                  Render.com deployment descriptor
tests/                       ~454 tests
docs/architecture.md         Search pipeline, schema, component details
docs/vps-architecture.md     VPS/cloud deployment architecture
```

## License

[PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/) — free for personal, research, educational, and nonprofit use. Commercial use requires a separate license.
