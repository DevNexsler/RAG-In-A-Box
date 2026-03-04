# Obsidian Vault Semantic Index

Index an Obsidian vault (Markdown, PDFs, images) into a vector store and expose it as an **MCP server** — any MCP-compatible AI assistant (Claude Code, OpenClaw, Claude Desktop, Cursor, etc.) can search your vault with a single config entry. Supports both **cloud APIs** (default — no local servers needed) and **local/self-hosted** providers.

Uses **Qwen3-Embedding-8B** (via OpenRouter) for embeddings, **GPT-4.1 Mini** (via OpenRouter) for document enrichment, **Gemini Vision** (cloud) or **DeepSeek OCR2** (local) for OCR, **Qwen3-Reranker-8B** (via Baseten) for cross-encoder reranking, **Prefect** for orchestration, **LanceDB** for storage + full-text search.

## Getting started

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/Document-Organizer.git
cd Document-Organizer
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
- `vault_root` — path to your Obsidian vault
- `index_root` — where the index will be stored

> **Self-hosting?** Use `cp config.local.yaml.example config.yaml` instead. This config uses Ollama, DeepSeek OCR2, and llama-server — see [Local mode](#local-mode-optional) below.

### 3. Add API keys

Create a `.env` file in the project root:

```bash
GEMINI_API_KEY=...               # OCR — get one at https://aistudio.google.com/apikey
OPENROUTER_API_KEY=sk-or-...     # embeddings + enrichment — https://openrouter.ai/keys
BASETEN_API_KEY=...              # reranker — https://app.baseten.co/settings/api_keys
```

### 4. Build the index

```bash
python run_index.py
```

This scans your vault, extracts text (Markdown, PDFs, images), generates embeddings, and writes everything to a LanceDB index. Prefect auto-starts a temporary server for flow/task logging — dashboard at `http://127.0.0.1:4200`.

### 5. Connect your AI assistant

The MCP server gives any compatible AI assistant access to your vault via tools like `vault_search`, `vault_status`, and `vault_recent`. The assistant launches the server automatically — you just add a config entry.

#### Claude Code

Add to your project's `.mcp.json` (or `~/.claude.json` for global access):

```json
{
  "mcpServers": {
    "vault-index": {
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
    "vault-index": {
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
    "vault-index": {
      "command": "/path/to/Document-Organizer/.venv/bin/python",
      "args": ["/path/to/Document-Organizer/mcp_server.py"],
      "cwd": "/path/to/Document-Organizer"
    }
  }
}
```

#### Any MCP-compatible client

The pattern is the same for Cursor, Windsurf, or any tool that supports MCP stdio servers. Point `command` at the venv Python and `args` at `mcp_server.py`. API keys are loaded from the `.env` file automatically — no need to pass them in the MCP config.

#### HTTP mode (testing / non-MCP clients)

```bash
python mcp_server.py --http
# Listens on 127.0.0.1:7788
```

### Local mode (optional)

To run everything on your own hardware instead of cloud APIs:

1. Copy `config.local.yaml.example` to `config.yaml`
2. Install and start **Ollama**: `brew install ollama && ollama serve`
   - `ollama pull qwen3-embedding:0.6b` (semantic chunking)
   - `ollama pull qwen3-embedding:4b-q8_0` (embeddings)
3. Start **DeepSeek OCR2** on port 8790 (for PDF/image OCR)
4. Place reranker GGUF model in `models/` (for llama_cpp reranker)

No cloud API keys needed in local mode.

## Features

```
Obsidian Vault                          AI Assistants
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
                   |  + full-text (BM25) |  vault_search
                   +---------------------+  vault_status
                                            vault_recent ...
```

**Hybrid search** — Every query runs vector (semantic) and keyword (BM25) search in parallel, fuses results with Reciprocal Rank Fusion, then reranks with a cross-encoder. Pre-filters (tags, folders, doc type, topics) apply at the database level before retrieval so every result matches.

**Multi-format extraction** — Indexes Markdown, PDFs, and images. PDFs use text extraction first, falling back to OCR for scanned pages. Images get OCR text plus visual descriptions. EXIF metadata (camera, GPS, dates) is extracted automatically.

**LLM enrichment** — Each document is analyzed by an LLM to extract structured metadata: summary, document type, entities (people, places, orgs, dates), topics, keywords, and key facts. All fields are searchable and filterable.

**Smart chunking** — Markdown is split by headings, PDFs by pages. Large sections get semantic chunking (topic-boundary detection via sentence embeddings). Every chunk gets a contextual header prepended with its title, path, and topics — so each chunk is self-describing for better retrieval.

**Rich metadata & filtering** — Obsidian YAML frontmatter (tags, status, author, dates, custom fields) is automatically extracted and promoted to filterable columns. Custom frontmatter keys are auto-promoted — no schema changes needed.

**MCP server** — Exposes 9 tools over the Model Context Protocol. Any MCP-compatible assistant can search, browse, and filter your vault. Works over stdio (launched automatically by the assistant) or HTTP.

**Incremental updates** — Only new and modified files are processed on re-index. Deleted files are cleaned up automatically. Failed documents are tracked and retried.

**Cloud or local** — Every component (OCR, embeddings, enrichment, reranker) has both cloud and local provider options. Default config uses cloud APIs with no servers to run. Switch to fully self-hosted with a single config file swap.

**Resilient by default** — Per-document error handling with retries, structured MCP error responses, search diagnostics on every query (`vector_search_active`, `reranker_applied`, `degraded`), and SQL injection protection on filter keys.

### MCP tools

| Tool | Description |
|------|-------------|
| `vault_search` | Hybrid semantic + keyword search with filters (tags, folder, source_type, doc_type, topics, etc.) |
| `vault_get_chunk` | Get full text + metadata for one chunk by doc_id and loc |
| `vault_get_doc_chunks` | Get all chunks for a document, sorted by position |
| `vault_list_documents` | Browse all indexed documents with pagination and filters |
| `vault_recent` | Recently modified/indexed docs (newest first) |
| `vault_facets` | Distinct values + counts for all filterable fields |
| `vault_folders` | Vault folder/directory structure with file counts |
| `vault_status` | Index stats, provider settings, health checks |
| `vault_index_update` | Incrementally update the index without leaving the assistant |

## Run tests

```bash
python -m pytest tests/ -m "not live" -x    # ~240 offline tests (no API keys)
python -m pytest tests/ -x                   # ~290 full suite (requires API keys)
```

## Project layout

```
core/                        Config + storage interface
providers/embed/             Embedding providers (OpenRouter, Ollama, Baseten, Gemini)
providers/llm/               LLM providers (OpenRouter, Ollama, Baseten)
providers/ocr/               OCR providers (Gemini Vision, DeepSeek OCR2)
doc_enrichment.py            LLM metadata extraction
extractors.py                Text extraction (MD, PDF, images)
flow_index_vault.py          Prefect indexing flow
lancedb_store.py             LanceDB storage + search
search_hybrid.py             4-stage hybrid search pipeline
mcp_server.py                MCP server (stdio + HTTP)
run_index.py                 CLI entrypoint
config.yaml.example          Cloud config template
config.local.yaml.example    Local/self-hosted config template
tests/                       ~290 tests
docs/architecture.md         Search pipeline, schema, component details
```
