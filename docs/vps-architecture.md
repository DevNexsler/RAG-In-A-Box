# VPS Architecture — Cloud-Native Document Organizer

## Goals

1. **No Obsidian dependency** — works with any markdown source
2. **VPS-ready** — runs on any Linux server with persistent disk
3. **Stateless compute, persistent data** — server can restart without data loss

---

## What's Already Done (VPS-Version branch)

These items from the original plan are **complete**:

- [x] `documents_root` config key (replaces `vault_root`, with fallback alias)
- [x] All MCP tools renamed from `vault_*` to `file_*`
- [x] Cloud providers (OpenRouter embeddings + enrichment, Baseten reranker) — no local GPU needed
- [x] Taxonomy system for consistent tagging (7 MCP CRUD tools)
- [x] MCP HTTP mode already works (`--http` flag)
- [x] 358 tests passing

---

## Current Architecture

```
Local Machine / VPS
  Documents folder (any path)
       |
  [Indexer] → LanceDB (local .lance/ files)
       |            + Taxonomy table
  [MCP Server] ← AI assistants (stdio or HTTP)
       |
  Cloud APIs:
    OpenRouter (embeddings + enrichment)
    Baseten (reranker)
    Gemini or DeepSeek OCR2 (OCR)
```

The core engine is already cloud-API-only (no local GPU). The remaining VPS work is just **deployment packaging** — how to run it remotely, accept files, and secure it.

---

## Remaining Work — 3 Phases

### Phase 1: Deployable Server (minimum viable VPS)

**Goal:** Run on a VPS, accept MCP connections over HTTP, serve from persistent disk.

| Task | What to do | Complexity |
|------|-----------|------------|
| **`server.py`** | Unified entrypoint: starts MCP HTTP server on `$PORT` (default 7788). Just wraps existing `mcp_server.py --http`. | Small — ~30 lines |
| **`Dockerfile`** | Python 3.13-slim, `pip install -r requirements.txt`, `VOLUME /data`, `CMD python server.py` | Small |
| **`config.vps.yaml.example`** | Same as cloud config but with `/data/documents` and `/data/index` paths | Small — copy + edit |
| **Auth middleware** | `API_KEY` env var checked on every HTTP request. Single `Bearer` token. Add to `mcp_server.py` HTTP handler. | Small — ~20 lines |
| **Env var overrides** | `DOCUMENTS_ROOT`, `INDEX_ROOT`, `PORT` override config.yaml values | Small — in `core/config.py` |

**Result:** `docker build && docker run -v /data:/data -e API_KEY=... -e OPENROUTER_API_KEY=...` and it works. MCP clients connect via HTTP/SSE.

### Phase 2: Document Ingestion API

**Goal:** Accept files from external sources (upload, git sync, web UI).

| Task | What to do | Complexity |
|------|-----------|------------|
| **`api_server.py`** | FastAPI app with REST routes. Mount alongside MCP server in `server.py`. | Medium |
| `POST /api/upload` | Multipart file upload → save to `documents_root` → trigger incremental index | Medium |
| `POST /api/sync` | Trigger `file_index_update` (re-scan documents_root) | Small — calls existing flow |
| `GET /api/search` | REST wrapper around `file_search` for non-MCP clients | Small |
| `GET /api/status` | REST wrapper around `file_status` | Small |
| `DELETE /api/documents/{doc_id}` | Remove file from documents_root + delete from index | Small |

**Note:** The REST API is optional. MCP HTTP already provides full functionality for AI assistants. The REST API is for web UIs and scripts that don't speak MCP.

### Phase 3: Platform Configs (as needed)

| Platform | Config file | Notes |
|----------|------------|-------|
| **Render.com** | `render.yaml` | Render Disk at `/data`, auto-deploy from git |
| **Docker Compose** | `docker-compose.yml` | Optional Prefect container if you want dashboards |
| **Systemd** | `doc-organizer.service` | For bare-metal VPS (Hetzner, DO, etc.) |
| **Fly.io** | `fly.toml` | Fly Volume for persistence |

Add these as needed per deployment target. Don't pre-build all of them.

---

## Design Principles

**Keep it simple:**
- One process serves everything (MCP + optional REST API)
- LanceDB is the only data store (no Postgres, no Redis, no external DBs)
- Cloud APIs handle all ML inference (no GPU, no model downloads on server)
- Persistent disk at `/data` — documents + index side by side

**Evolve incrementally:**
- Phase 1 is a weekend project — it's just Docker + auth
- Phase 2 adds ingestion only when you need non-filesystem sources
- Phase 3 is platform-specific glue, done only for platforms you actually deploy to

**Don't over-engineer:**
- No multi-tenant auth until you actually have multiple users
- No S3-backed LanceDB until you need multi-instance scaling
- No message queue for indexing until throughput demands it
- No Kubernetes — a single Docker container is fine for thousands of documents

---

## Storage

LanceDB stores everything as local `.lance/` files. This must survive container restarts.

| Platform | Approach |
|----------|---------|
| **Any VPS** | Regular disk at `/data` (ext4/ZFS). Back up with rsync/restic. |
| **Render.com** | Render Disk (persistent SSD at `/data`). Survives deploys. |
| **Fly.io** | Fly Volume (NVMe). |
| **Docker** | Bind mount or named volume at `/data`. |

```yaml
# config.vps.yaml
documents_root: "/data/documents"
index_root: "/data/index"
```

---

## Environment Variables (VPS)

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | Embeddings + enrichment |
| `BASETEN_API_KEY` | Yes | Reranker |
| `GEMINI_API_KEY` | No | OCR (only if using Gemini instead of DeepSeek OCR2) |
| `API_KEY` | Yes (VPS) | Bearer token auth for HTTP access |
| `DOCUMENTS_ROOT` | No | Override config (default: from config.yaml) |
| `INDEX_ROOT` | No | Override config (default: from config.yaml) |
| `PORT` | No | Server port (default: 7788) |

---

## What Stays the Same (no changes needed)

- LanceDB as the storage engine
- All cloud providers (already stateless API calls)
- Hybrid search pipeline (vector + BM25 + RRF + reranker)
- LLM enrichment + taxonomy integration
- Frontmatter extraction (works with any YAML frontmatter)
- Chunking strategies
- Full test suite

---

## Ingestion Methods (Phase 2+)

| Method | How it works | When to add |
|--------|-------------|-------------|
| **Filesystem scan** | Point `documents_root` at a folder, run indexer | Already works (Phase 1) |
| **Upload API** | `POST /api/upload` with file | Phase 2 |
| **Git sync** | Clone/pull a repo to `documents_root` on schedule | Phase 2 (cron + git pull) |
| **Watch directory** | inotify/fswatch triggers re-index | Phase 2 (optional) |
| **S3/R2 sync** | Pull from bucket on schedule | Future (if needed) |

---

## Compatible Software

Works with any tool that produces markdown with optional YAML frontmatter:

Obsidian, HackMD/CodiMD, Notion (export), Logseq, GitBook, Typora, iA Writer, VS Code, Google Docs (export), Jekyll, Hugo — all use standard `---` delimited YAML frontmatter.
