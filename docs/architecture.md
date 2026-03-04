# Architecture Reference

Developer-facing details about the search pipeline, storage schema, providers, and configuration.

## Env vars

| Variable             | When needed                                              |
|----------------------|----------------------------------------------------------|
| `GEMINI_API_KEY`     | Default cloud config — OCR via Gemini Vision (not needed if using local DeepSeek OCR2) |
| `OPENROUTER_API_KEY` | Default cloud config — embeddings (Qwen3-Embedding-8B) + enrichment (GPT-4.1 Mini) |
| `BASETEN_API_KEY`    | Default cloud config — reranker (Qwen3-Reranker-8B) (not needed if using local llama_cpp) |

Store these in a `.env` file in the project root. The MCP server and indexer load it automatically.

## Key components

| What              | Component                                                     |
|-------------------|---------------------------------------------------------------|
| Chunking          | Heading-aware (MD) + page-aware (PDF) + `SentenceSplitter`. Semantic fallback via Ollama 0.6B for large sections. Contextual headers on all chunks. |
| Semantic chunking | Qwen3-Embedding-0.6B via Ollama (local). Detects topic boundaries for sections >3600 chars. |
| LLM enrichment   | GPT-4.1 Mini via OpenRouter. Structured JSON schema output. Extracts summary, doc_type, entities, topics, keywords, key_facts per document. |
| Embeddings        | Qwen3-Embedding-8B via OpenRouter (batch_size: 64, concurrency: 2) |
| Vector store      | LanceDB (`LanceDBStore` — direct search with native pre-filters) |
| Full-text search  | LanceDB + tantivy (BM25)                                     |
| OCR (PDF pages)   | Gemini Vision (cloud, default) or DeepSeek OCR2 (local) — text extraction from scanned PDFs/images |
| Image description | Gemini Vision `describe()` — text + detailed visual description (when Gemini OCR enabled) |
| Image metadata    | Pillow EXIF extraction (camera, date, GPS, dimensions)        |
| PDF metadata      | PyMuPDF (title, author, dates, page count)                    |
| Reranking         | Qwen3-Reranker-8B via Baseten (cloud, scale-to-zero with cold-start check) |
| Orchestration     | Prefect 3.x — flow/task logging, retry, dashboard at `http://127.0.0.1:4200` |

## Search pipeline

1. **Pre-filter:** Build SQL WHERE clause from filters (source_type, folder, tags, enr_doc_type, enr_topics, etc.) and apply at LanceDB level via `.where(clause, prefilter=True)` — filters run before ANN/FTS scoring so full `top_k` results match the criteria
2. **Parallel retrieval:** vector search (semantic) + BM25/FTS (keyword) — run concurrently, both with pre-filters applied
3. **RRF fusion:** Reciprocal Rank Fusion merges both ranked lists
4. **Cross-encoder reranking:** Qwen3-Reranker-8B (Baseten) scores query-document pairs (timeout: 120s)
5. **Diagnostics:** Every response includes `{vector_search_active, keyword_search_active, reranker_applied, degraded}` so callers detect silent degradation

## LanceDB schema (per chunk)

| Field | Type | Description |
|-------|------|-------------|
| `chunk_uid` | string | Unique ID: `doc_id::loc` |
| `doc_id` | string | Vault-relative path (e.g. `Projects/notes.md`) |
| `loc` | string | Location within doc (`c:0`, `p:2:c:1`, `img:c:0`) |
| `text` | string | Contextual header + extracted text (header aids search retrieval) |
| `snippet` | string | First ~200 chars of raw text (without header) for clean display |
| `section` | string | Heading breadcrumb for MD chunks (e.g. `Setup > Prerequisites`) |
| `source_type` | string | `md`, `pdf`, or `img` |
| `mtime` | float | Source file's last-modified time |
| `size` | int | Source file size in bytes |
| `title` | string | From YAML frontmatter → first `# heading` → filename |
| `tags` | string | From YAML frontmatter, comma-separated (e.g. `"recipe,korean"`) |
| `folder` | string | Top-level directory from vault path |
| `status` | string | From YAML frontmatter (e.g. `active`, `archived`) |
| `created` | string | From YAML frontmatter (e.g. `2026-01-15`) |
| `description` | string | From YAML frontmatter `description` field |
| `author` | string | From YAML frontmatter `author` field |
| `keywords` | string | From YAML frontmatter, comma-separated (e.g. `"budget,timeline"`). Separate from LLM `enr_keywords`. |
| `custom_meta` | string | JSON dict of remaining frontmatter fields (e.g. `{"source": "...", "published": "..."}`) |
| `enr_summary` | string | LLM-generated document summary (empty if not enriched) |
| `enr_doc_type` | string | Comma-separated document types (e.g. `report,engineering`) |
| `enr_entities_people` | string | Comma-separated person names |
| `enr_entities_places` | string | Comma-separated locations/addresses |
| `enr_entities_orgs` | string | Comma-separated organization names |
| `enr_entities_dates` | string | Comma-separated dates (YYYY-MM-DD) |
| `enr_topics` | string | Comma-separated high-level topics |
| `enr_keywords` | string | Comma-separated specific terms and phrases |
| `enr_key_facts` | string | Comma-separated key facts/conclusions |
| `vector` | float[] | Embedding vector (2560-dim for Qwen3, 768-dim for Gemini) |

Metadata fields (`title`, `tags`, `status`, `created`, `description`, `author`, `keywords`, `folder`) are automatically enriched during indexing from Obsidian YAML frontmatter and file path. LLM enrichment fields (prefixed `enr_`: `enr_summary`, `enr_doc_type`, `enr_entities_*`, `enr_topics`, `enr_keywords`, `enr_key_facts`) are extracted by GPT-4.1 Mini via OpenRouter during indexing when `enrichment.enabled` is true. The `enr_` prefix avoids collisions with user frontmatter fields (e.g. a user's `summary` frontmatter won't overwrite the LLM-generated `enr_summary`). All metadata is returned in search results and can be used as filters. Dynamic metadata fields from frontmatter (e.g. `priority`, `category`) are automatically promoted to LanceDB columns and appear in search results, facets, and filters.
