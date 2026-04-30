"""Prefect flow: scan vault, diff with store, process docs, delete removed, log stats.

Uses LlamaIndex for chunking (SentenceSplitter) and embeddings.
Storage via our LanceDBStore (which wraps LlamaIndex's LanceDBVectorStore).
Extraction via extractors.py (Markdown, PDF, images, documents, spreadsheets, plain text).

Chunking enhancements:
- Contextual headers: each chunk is prepended with document metadata
  (title, type, folder, tags, page/section) before embedding so that
  chunks are self-describing and retrieve better in isolation.
- Heading-aware splitting for Markdown: text is first split at heading
  boundaries (h1-h3), then SentenceSplitter runs within each section.
  Each chunk carries its heading hierarchy as context.

LLM document enrichment (optional):
- When enrichment.enabled is true, the configured LLM provider extracts
  structured metadata (summary, doc_type, entities, topics, keywords,
  key_facts) from every new/modified document before chunking.
- Summary and topics are prepended to chunk contextual headers.
- All enrichment fields are stored in LanceDB for filtering and search.

Auto-recovery:
- After indexing, the flow validates the table is readable.
- If LanceDB corruption is detected (missing files from interrupted writes),
  it walks versions backward, exports data from the last clean version,
  and recreates the table automatically.

Complex objects (store, embed_provider, splitter, ocr_provider) are built inside
the flow and passed to tasks via a shared module-level dict (_RUNTIME) rather
than as task arguments.  This avoids Prefect 3's input-serialisation warnings
while keeping the flow easy to read.
"""

import logging
import os
import re
import shutil
import threading
from pathlib import Path
from typing import Any

from prefect import flow, task
from prefect.logging import get_run_logger


def _get_logger():
    """Get a Prefect run logger if available, otherwise fall back to stdlib."""
    try:
        return get_run_logger()
    except Exception:
        return logging.getLogger(__name__)

from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.node_parser import SentenceSplitter

from core.config import load_config
from core.source_types import SOURCE_TYPE_BY_EXTENSION, canonical_source_type
from doc_enrichment import enrich_document, empty_enrichment, ENRICHMENT_FIELDS, failed_enrichment
from extractors import extract_text, extract_title, derive_folder, normalize_tags
from providers.embed import build_embed_provider
from providers.embed.base import EmbedProvider
from providers.ocr import build_ocr_provider
from doc_id_store import (
    DocIDStore, extract_id_from_filename, inject_id_into_filename,
    strip_id_from_filename as _strip_id_from_filename,
)
from hooks.dispatcher import dispatch_event
from hooks.events import build_document_indexed_event
from lancedb_store import LanceDBStore


# Module-level runtime context populated by the flow, read by tasks.
# Avoids passing unpickleable objects as Prefect task arguments.
_RUNTIME: dict[str, Any] = {}


# --- Helpers ---


_SOURCE_TYPE_MAP = SOURCE_TYPE_BY_EXTENSION


_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


def _split_markdown_by_headings(text: str) -> list[tuple[str, str]]:
    """Split markdown at heading boundaries (h1-h3).

    Returns (heading_breadcrumb, section_text) pairs.  The breadcrumb tracks
    the heading hierarchy, e.g. "Setup > Prerequisites".  Text before the
    first heading gets an empty breadcrumb.  Each section includes its own
    heading line so the content is self-contained.
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]

    sections: list[tuple[str, str]] = []
    heading_stack: list[tuple[int, str]] = []

    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))

    for i, match in enumerate(matches):
        level = len(match.group(1))
        heading_text = match.group(2).strip()

        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, heading_text))

        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()

        if section_text:
            breadcrumb = " > ".join(h[1] for h in heading_stack)
            sections.append((breadcrumb, section_text))

    return sections


def _build_chunk_context(
    doc_meta: dict,
    page: int | None = None,
    section: str | None = None,
) -> str:
    """Build a minimal contextual header prepended to chunk text before embedding.

    Keeps only what's needed to situate the chunk within its parent document.
    Deliberately lean to avoid diluting lexical (FTS/BM25) retrieval with
    metadata terms — all metadata is stored in separate LanceDB columns and
    searchable via filters.

    Example output::

        [Document: TaxReturn | Topics: tax filing | Page: 31]
        Summary: Joint federal tax filing for 2022.

        <actual chunk text>
    """
    parts: list[str] = []
    if doc_meta.get("title"):
        parts.append(f"Document: {doc_meta['title']}")
    if doc_meta.get("enr_topics"):
        parts.append(f"Topics: {doc_meta['enr_topics']}")
    if page is not None:
        parts.append(f"Page: {page}")
    if section:
        parts.append(f"Section: {section}")
    if not parts:
        return ""

    header = f"[{' | '.join(parts)}]"
    summary = doc_meta.get("enr_summary", "")
    if summary:
        header += f"\nSummary: {summary}"
    return header + "\n\n"


def _semantic_subsplit(text: str, semantic_splitter) -> list[str]:
    """Use SemanticSplitterNodeParser to find topic boundaries in large text."""
    from llama_index.core.schema import Document as LIDocument

    doc = LIDocument(text=text)
    nodes = semantic_splitter.get_nodes_from_documents([doc])
    return [n.text for n in nodes if n.text.strip()]


def _split_section(
    text: str,
    splitter,
    semantic_splitter=None,
    semantic_threshold: int = 0,
) -> list[str]:
    """Split a text section into chunks.

    If the section exceeds *semantic_threshold* and a semantic splitter is
    available, it first finds topic boundaries via embedding similarity,
    then runs SentenceSplitter within each topic sub-section.  Otherwise
    falls back to SentenceSplitter directly.
    """
    if semantic_splitter and semantic_threshold and len(text) > semantic_threshold:
        sub_sections = _semantic_subsplit(text, semantic_splitter)
        chunks: list[str] = []
        for sub in sub_sections:
            chunks.extend(splitter.split_text(sub))
        return chunks
    return splitter.split_text(text)


def _matches_any(rel_str: str, patterns: list[str]) -> bool:
    """Check if a vault-relative path matches any glob pattern (supports ** for any depth)."""
    import fnmatch
    for pat in patterns:
        # fnmatch doesn't handle ** well; expand: **/*.md should match both root and nested
        if pat.startswith("**/"):
            # Match the suffix at any depth: *.md for root, **/*.md for nested
            suffix = pat[3:]  # e.g. "*.md"
            if fnmatch.fnmatch(rel_str, suffix):
                return True
            if fnmatch.fnmatch(rel_str, pat):
                return True
            # Also try: any/path/*.md via full glob
            if fnmatch.fnmatch(rel_str, "*/" + suffix):
                return True
            # Recursive match via os-level
            parts = rel_str.split("/")
            if fnmatch.fnmatch(parts[-1], suffix):
                return True
        else:
            if fnmatch.fnmatch(rel_str, pat):
                return True
    return False


# --- Tasks (one responsibility each) ---


def scan_filesystem_records(
    vault_root: str | Path,
    include: list[str],
    exclude: list[str],
    doc_id_store: DocIDStore | None = None,
    logger=None,
) -> list[dict]:
    """Scan vault; return list of file records as dicts.

    Each file gets a persistent 5-char base-62 doc_id embedded in its filename
    as @XXXXX@. Files already carrying an ID keep it; new files get one assigned
    and are renamed on disk.
    """
    root = Path(vault_root)
    if not root.exists():
        return []

    if logger is None:
        logger = _get_logger()

    records = []
    # Track IDs seen this scan to detect collisions (two files with same @XXXXX@)
    seen_ids: dict[str, str] = {}  # doc_id → rel_path of first file seen

    # os.walk with followlinks=True so symlinked directories (e.g. NAS mounts) are traversed.
    # Path.rglob does not follow symlinks in Python 3.13+.
    # Track visited real paths to guard against symlink cycles.
    visited: set[str] = set()
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in visited:
            continue
        visited.add(real)
        for fname in filenames:
            full_path = Path(dirpath) / fname
            rel_str = str(full_path.relative_to(root)).replace("\\", "/")
            if _matches_any(rel_str, exclude):
                continue
            if not _matches_any(rel_str, include):
                continue
            try:
                stat = full_path.stat()
            except OSError:
                continue
            if stat.st_size == 0:
                logger.warning("Skipping empty file: %s", rel_str)
                continue

            # --- Persistent doc_id assignment ---
            existing_id = extract_id_from_filename(fname)
            if existing_id and doc_id_store:
                # Check for retired ID: was this ID previously deleted?
                # Copy-pasted files may carry a stale @XXXXX@ from a deleted doc.
                if doc_id_store.is_retired(existing_id):
                    retired = doc_id_store.retired_info(existing_id)
                    last_path = retired["last_path"] if retired else "unknown"
                    logger.warning(
                        "Retired ID %s (was %s) found on %s — assigning fresh ID",
                        existing_id, last_path, rel_str,
                    )
                    doc_id_store.log_event(
                        DocIDStore.COLLISION, existing_id, rel_str,
                        detail=f"retired ID, previously used by {last_path}",
                    )
                    existing_id = None  # fall through to "no ID" path below
                # Check for collision: another file already claimed this ID
                elif existing_id in seen_ids:
                    # Collision! Re-assign a fresh ID to this file
                    logger.warning(
                        "ID collision: %s already used by %s — re-assigning %s",
                        existing_id, seen_ids[existing_id], rel_str,
                    )
                    doc_id_store.log_event(
                        DocIDStore.COLLISION, existing_id, rel_str,
                        detail=f"already claimed by {seen_ids[existing_id]}",
                    )
                    existing_id = None  # fall through to "no ID" path below
                else:
                    doc_id = existing_id
                    seen_ids[doc_id] = rel_str
                    # Register/update mapping in SQLite
                    stored_path = doc_id_store.lookup_path(doc_id)
                    if stored_path != rel_str:
                        doc_id_store.register(doc_id, rel_str)

            if existing_id is None and doc_id_store:
                # No ID in filename (or collision stripped it) — assign one and rename
                doc_id = doc_id_store.next_id()
                # Strip any old @XXXXX@ from filename before injecting new one
                clean_fname = _strip_id_from_filename(fname)
                new_fname = inject_id_into_filename(clean_fname, doc_id)
                new_full_path = full_path.parent / new_fname
                try:
                    full_path.rename(new_full_path)
                except OSError as exc:
                    # Rename failed (permissions, read-only mount, etc.)
                    # Fall back to using rel_path as doc_id — file stays un-tagged
                    logger.warning("Cannot rename %s → %s: %s", fname, new_fname, exc)
                    doc_id_store.log_event(
                        DocIDStore.RENAME_FAILED, doc_id, rel_str,
                        detail=str(exc),
                    )
                    doc_id = rel_str
                    doc_id_store.register(doc_id, rel_str)
                    seen_ids[doc_id] = rel_str
                else:
                    full_path = new_full_path
                    rel_str = str(full_path.relative_to(root)).replace("\\", "/")
                    doc_id_store.register(doc_id, rel_str)
                    seen_ids[doc_id] = rel_str
            elif not doc_id_store:
                # Fallback: no doc_id_store (shouldn't happen in normal flow)
                doc_id = rel_str

            records.append({
                "doc_id": doc_id,
                "rel_path": rel_str,
                "abs_path": str(full_path.resolve()),
                "mtime": stat.st_mtime,
                "size": stat.st_size,
                "ext": full_path.suffix.lower().lstrip(".") or "bin",
            })
    return records


@task
def scan_vault_task(vault_root: str | Path, include: list[str], exclude: list[str]) -> list[dict]:
    """Prefect wrapper for filesystem scan."""
    return scan_filesystem_records(
        vault_root,
        include,
        exclude,
        doc_id_store=_RUNTIME.get("doc_id_store"),
        logger=_get_logger(),
    )


@task
def diff_index_task(
    scanned: list[dict],
    stored_doc_mtimes: dict[str, float],
) -> tuple[list[dict], list[str]]:
    """Compare scanned files vs stored docs. Return (to_add_or_update, to_delete).

    A file is added/updated if:
      - It's new (not in store), OR
      - Its mtime has changed since last index
    A file is deleted if it's in the store but no longer on disk.
    """
    scanned_ids = {r["doc_id"] for r in scanned}
    stored_ids = set(stored_doc_mtimes.keys())

    to_add_or_update = []
    for r in scanned:
        doc_id = r["doc_id"]
        if doc_id not in stored_ids:
            # New file
            to_add_or_update.append(r)
        elif r["mtime"] != stored_doc_mtimes.get(doc_id, 0.0):
            # Modified file (mtime changed)
            to_add_or_update.append(r)
        # else: unchanged, skip

    to_delete = list(stored_ids - scanned_ids)
    return to_add_or_update, to_delete


@task(retries=1, timeout_seconds=1800)
def process_doc_task(doc: dict) -> None:
    """Extract text, chunk with LlamaIndex, embed, upsert into store.

    Reads store / embed_provider / splitter / ocr_provider / config from _RUNTIME.
    Handles Markdown, PDF, images, documents (docx/pptx/html/etc.), spreadsheets, and plain text.
    """
    store: LanceDBStore = _RUNTIME["store"]
    embed_provider: EmbedProvider = _RUNTIME["embed_provider"]
    splitter: SentenceSplitter = _RUNTIME["splitter"]
    semantic_splitter = _RUNTIME.get("semantic_splitter")
    semantic_threshold: int = _RUNTIME.get("semantic_threshold", 0)
    ocr_provider = _RUNTIME.get("ocr_provider")  # may be None
    config: dict = _RUNTIME.get("config", {})

    logger = get_run_logger()
    doc_id = doc["doc_id"]
    rel_path = doc.get("rel_path", doc_id)
    mtime = doc["mtime"]
    size = doc["size"]
    ext = doc.get("ext", "")
    source_name = doc.get("source_name", "documents")
    logger.info(f"Processing: {rel_path} (id={doc_id})")

    # --- Determine source_type ---
    # Prefer explicit source_type from the record (set by SourceRecord); fall back
    # to the extension-based map for backward compat.
    source_type = canonical_source_type(doc.get("source_type") or ext)

    # --- Extract text via Source dispatch ---
    # Look up the owning Source and original SourceRecord, then call source.extract().
    # This is source-agnostic: FilesystemSource reads files; PostgresSource uses
    # cached text from scan(); future sources can do whatever they need.
    sources_by_name: dict = _RUNTIME.get("sources_by_name", {})
    source_records_by_ns_doc_id: dict = _RUNTIME.get("source_records_by_ns_doc_id", {})
    src = sources_by_name.get(source_name)
    source_record = source_records_by_ns_doc_id.get(doc_id)

    if src is not None and source_record is not None:
        result = src.extract(source_record)
    else:
        # Fallback: direct extract_text for records that pre-date the source refactor
        # (e.g., tasks spawned from a _RUNTIME that doesn't have sources_by_name yet).
        pdf_cfg = config.get("pdf", {})
        abs_path = doc.get("abs_path", doc_id)
        result = extract_text(
            file_path=abs_path,
            ext=ext,
            ocr_provider=ocr_provider,
            pdf_strategy=pdf_cfg.get("strategy", "text_then_ocr"),
            min_text_chars=pdf_cfg.get("min_text_chars_before_ocr", 200),
            ocr_page_limit=pdf_cfg.get("ocr_page_limit", 200),
        )

    if not result.full_text.strip():
        logger.warning(f"No text extracted: {doc_id}")
        return

    # --- Extract document-level metadata ---
    fm = result.frontmatter  # from Markdown frontmatter; empty dict for PDF/images
    title = fm.get("title") or extract_title(result.full_text, doc_id)
    tags = normalize_tags(fm.get("tags"))
    folder = derive_folder(rel_path)
    status = fm.get("status", "archived" if folder.lower() in ("archive", "archived") else "active")
    created = str(fm["created"]) if "created" in fm else ""
    description = str(fm.get("description", "")).strip()
    author = str(fm.get("author", "")).strip()
    keywords = normalize_tags(fm.get("keywords"))

    # Collect remaining frontmatter fields into custom_meta JSON
    _KNOWN_FM_KEYS = {"title", "tags", "status", "created", "description", "author", "keywords"}
    extra_fm = {k: str(v) for k, v in fm.items() if k not in _KNOWN_FM_KEYS and v is not None}
    import json as _json
    custom_meta = _json.dumps(extra_fm, default=str) if extra_fm else ""

    # Shared metadata for all chunks of this doc
    doc_meta = {
        "doc_id": doc_id,
        "rel_path": rel_path,
        "source_type": source_type,
        "source_name": source_name,
        "mtime": mtime,
        "size": size,
        "title": title,
        "tags": tags,
        "folder": folder,
        "status": status,
        "created": created,
        "description": description,
        "author": author,
        "keywords": keywords,
        "custom_meta": custom_meta,
        "section": "",
    }
    # Promote extra frontmatter to real columns (skip collisions with reserved keys)
    for k, v in extra_fm.items():
        if k not in doc_meta:
            doc_meta[k] = v

    # --- LLM document enrichment (summary, entities, topics, etc.) ---
    llm_generator = _RUNTIME.get("llm_generator")
    taxonomy_store = _RUNTIME.get("taxonomy_store")
    enrichment_cfg = _RUNTIME.get("config", {}).get("enrichment", {})
    if llm_generator:
        enrichment = enrich_document(
            text=result.full_text,
            title=title,
            source_type=source_type,
            generator=llm_generator,
            max_input_chars=enrichment_cfg.get("max_input_chars", 4000),
            max_output_tokens=enrichment_cfg.get("max_output_tokens", 512),
            taxonomy_store=taxonomy_store,
        )
        if enrichment.get("_enrichment_failed"):
            reason = enrichment.pop("_enrichment_failed")
            logger.warning("Enrichment failed for '%s': %s", doc_id, reason)
            _RUNTIME.setdefault("_warnings", []).append(
                f"enrichment_failed:{doc_id}:{reason}"
            )
        elif not enrichment.get("enr_summary"):
            logger.warning("Enrichment returned empty summary for '%s' — LLM may have failed silently", doc_id)
        enrichment.pop("_enrichment_failed", None)
        doc_meta.update(enrichment)
    else:
        doc_meta.update(empty_enrichment())

    # --- Importance: frontmatter overrides LLM, track source ---
    fm_importance = fm.get("importance")
    if fm_importance is not None:
        # User provided importance in YAML frontmatter — use it
        try:
            imp_val = max(0.0, min(1.0, float(fm_importance)))
        except (TypeError, ValueError):
            imp_val = 0.5
        doc_meta["enr_importance"] = str(imp_val)
        doc_meta["enr_importance_source"] = "frontmatter"
    elif doc_meta.get("enr_importance"):
        # LLM generated it during enrichment
        doc_meta["enr_importance_source"] = "llm"
    else:
        # No enrichment and no frontmatter — neutral default
        doc_meta["enr_importance"] = "0.5"
        doc_meta["enr_importance_source"] = "default"

    # --- Chunk and build nodes ---
    # Three paths:
    #   1. Multi-page PDF  → page-aware chunks, context header per page
    #   2. Markdown         → heading-aware sections, context header per section
    #   3. Images / 1-page PDF → flat chunks with context header
    # Every chunk is prepended with a contextual header before embedding
    # so it is self-describing in isolation.
    nodes: list[TextNode] = []

    if ext == "pdf" and len(result.pages) > 1:
        for page_text in result.pages:
            if not page_text.text.strip():
                continue
            raw_chunks = _split_section(
                page_text.text, splitter, semantic_splitter, semantic_threshold
            )
            ctx = _build_chunk_context(doc_meta, page=page_text.page)
            contextualized = [ctx + c for c in raw_chunks]
            page_vectors = embed_provider.embed_texts(contextualized)

            for i, (ctx_text, raw_text, vector) in enumerate(
                zip(contextualized, raw_chunks, page_vectors, strict=True)
            ):
                loc = f"p:{page_text.page}:c:{i}"
                chunk_uid = f"{doc_id}::{loc}"
                snippet = (raw_text[:200] + "...") if len(raw_text) > 200 else raw_text
                node = TextNode(
                    text=ctx_text,
                    id_=chunk_uid,
                    embedding=vector,
                    metadata={**doc_meta, "loc": loc, "snippet": snippet},
                )
                node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
                nodes.append(node)

    elif source_type in ("md", "doc", "pres", "html", "epub", "csv"):
        # Heading-aware: split at h1-h3 boundaries, then SentenceSplitter within.
        sections = _split_markdown_by_headings(result.full_text)
        all_raw: list[str] = []
        all_ctx: list[str] = []
        all_sections: list[str] = []

        for heading_ctx, section_text in sections:
            raw_chunks = _split_section(
                section_text, splitter, semantic_splitter, semantic_threshold
            )
            ctx = _build_chunk_context(
                doc_meta, section=heading_ctx if heading_ctx else None
            )
            for raw in raw_chunks:
                all_raw.append(raw)
                all_ctx.append(ctx + raw)
                all_sections.append(heading_ctx)

        vectors = embed_provider.embed_texts(all_ctx)

        for i, (ctx_text, raw_text, sec, vector) in enumerate(
            zip(all_ctx, all_raw, all_sections, vectors, strict=True)
        ):
            loc = f"c:{i}"
            chunk_uid = f"{doc_id}::{loc}"
            snippet = (raw_text[:200] + "...") if len(raw_text) > 200 else raw_text
            meta = {**doc_meta, "loc": loc, "snippet": snippet}
            meta["section"] = sec
            node = TextNode(
                text=ctx_text,
                id_=chunk_uid,
                embedding=vector,
                metadata=meta,
            )
            node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
            nodes.append(node)

    else:
        # Images or single-page PDFs
        loc_prefix = "img" if source_type == "img" else ""
        raw_chunks = _split_section(
            result.full_text, splitter, semantic_splitter, semantic_threshold
        )
        ctx = _build_chunk_context(doc_meta)
        contextualized = [ctx + c for c in raw_chunks]
        vectors = embed_provider.embed_texts(contextualized)

        for i, (ctx_text, raw_text, vector) in enumerate(
            zip(contextualized, raw_chunks, vectors, strict=True)
        ):
            loc = f"{loc_prefix}:c:{i}" if loc_prefix else f"c:{i}"
            chunk_uid = f"{doc_id}::{loc}"
            snippet = (raw_text[:200] + "...") if len(raw_text) > 200 else raw_text
            node = TextNode(
                text=ctx_text,
                id_=chunk_uid,
                embedding=vector,
                metadata={**doc_meta, "loc": loc, "snippet": snippet},
            )
            node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
            nodes.append(node)

    # --- Upsert into store ---
    store.upsert_nodes(nodes)
    logger.info(f"Upserted {len(nodes)} chunks: {doc_id}")

    chunks = []
    for node in nodes:
        node_meta = getattr(node, "metadata", {}) or {}
        chunks.append({
            "loc": str(node_meta.get("loc") or ""),
            "snippet": str(node_meta.get("snippet") or ""),
            "text": getattr(node, "text", "") or "",
        })

    event = build_document_indexed_event(
        doc_id=doc_id,
        source_name=source_name,
        source_type=source_type,
        rel_path=rel_path,
        abs_path=str(doc.get("abs_path", "")),
        text=result.full_text,
        metadata=doc_meta,
        chunks=chunks,
    )
    warnings = dispatch_event(config.get("event_hooks"), event)
    if warnings:
        _RUNTIME.setdefault("_warnings", []).extend(warnings)
        for warning in warnings:
            logger.warning(warning)


def _process_docs(docs: list[dict], concurrency: int = 1) -> list[str]:
    """Process a batch of docs. Return list of failed doc_ids.

    concurrency=1 is the serial baseline (identical to the pre-refactor loop).
    concurrency>1 uses a ThreadPoolExecutor; exceptions in one worker do not
    affect others. FTS index rebuild is deliberately NOT done here — it is
    invoked exactly once by the flow after all docs have been processed.
    """
    logger = _get_logger()
    failed_docs: list[str] = []
    debug_concurrency = os.environ.get("INDEXER_DEBUG_CONCURRENCY", "").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
    }
    active_workers = 0
    peak_active_workers = 0
    active_lock = threading.Lock()

    def _run_one(doc: dict) -> str | None:
        nonlocal active_workers, peak_active_workers
        if debug_concurrency:
            with active_lock:
                active_workers += 1
                current_active = active_workers
                peak_active_workers = max(peak_active_workers, current_active)
                current_peak = peak_active_workers
            logger.info(
                "concurrency-debug start doc_id=%s thread=%s active=%d peak_active=%d",
                doc["doc_id"],
                threading.current_thread().name,
                current_active,
                current_peak,
            )
        try:
            process_doc_task(doc)
            return None
        except Exception as exc:
            logger.error("Skipping %s after retries exhausted: %s", doc["doc_id"], exc)
            return doc["doc_id"]
        finally:
            if debug_concurrency:
                with active_lock:
                    active_workers -= 1
                    current_active = active_workers
                    current_peak = peak_active_workers
                logger.info(
                    "concurrency-debug finish doc_id=%s thread=%s active=%d peak_active=%d",
                    doc["doc_id"],
                    threading.current_thread().name,
                    current_active,
                    current_peak,
                )

    if concurrency <= 1 or len(docs) <= 1:
        for doc in docs:
            bad = _run_one(doc)
            if bad is not None:
                failed_docs.append(bad)
        if debug_concurrency:
            logger.info(
                "concurrency-debug summary docs=%d concurrency=%d peak_active=%d failed=%d",
                len(docs),
                concurrency,
                peak_active_workers,
                len(failed_docs),
            )
        return failed_docs

    # Process the first doc serially to ensure the LanceDB table exists before
    # concurrent workers start. LanceDBVectorStore.add() creates the table on
    # first call and that initialization is not thread-safe — racing workers
    # can hit "Table already exists" errors.
    first_bad = _run_one(docs[0])
    if first_bad is not None:
        failed_docs.append(first_bad)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for result in ex.map(_run_one, docs[1:]):
            if result is not None:
                failed_docs.append(result)
    if debug_concurrency:
        logger.info(
            "concurrency-debug summary docs=%d concurrency=%d peak_active=%d failed=%d",
            len(docs),
            concurrency,
            peak_active_workers,
            len(failed_docs),
        )
    return failed_docs


@task
def delete_docs_task(doc_ids: list[str]) -> None:
    """Remove all chunk nodes for the given doc_ids."""
    if doc_ids:
        store: LanceDBStore = _RUNTIME["store"]
        store.delete_by_doc_ids(doc_ids)


@task
def index_stats_task(
    to_add_count: int,
    to_delete_count: int,
    run_seconds: float | None = None,
) -> None:
    """Log counts and optional duration."""
    logger = get_run_logger()
    logger.info(
        f"Index stats: added/updated={to_add_count}, deleted={to_delete_count}, "
        f"seconds={run_seconds:.1f}" if run_seconds else
        f"Index stats: added/updated={to_add_count}, deleted={to_delete_count}"
    )


@task
def write_index_metadata_task(
    index_root: str | Path,
    doc_count: int,
    chunk_count: int | None,
    failed_docs: list[str] | None = None,
    warnings: list[str] | None = None,
) -> None:
    """Write index_metadata.json for file_status (last_run_at, counts, failures, warnings)."""
    import json
    from datetime import datetime, timezone
    path = Path(index_root) / "index_metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    meta: dict = {
        "last_run_at": datetime.now(timezone.utc).isoformat(),
        "doc_count": doc_count,
        "chunk_count": chunk_count,
    }
    if failed_docs:
        meta["failed_count"] = len(failed_docs)
        meta["failed_docs"] = failed_docs[:20]
    if warnings:
        meta["warnings"] = warnings[:50]
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def _recover_corrupt_table(index_root: str | Path, table_name: str, logger) -> LanceDBStore | None:
    """Detect and auto-recover a corrupt LanceDB table.

    Walks versions backward to find the last readable one, exports its data,
    and recreates the table. Returns a new LanceDBStore on success, None on failure.
    """
    import lance

    lance_path = str(Path(index_root) / f"{table_name}.lance")
    if not Path(lance_path).exists():
        return None

    logger.warning("Corruption detected — attempting auto-recovery for %s", lance_path)

    # Find the last clean version
    try:
        ds = lance.dataset(lance_path)
        versions = sorted(ds.versions(), key=lambda x: x["version"], reverse=True)
    except Exception as exc:
        logger.error("Cannot read dataset versions: %s", exc)
        return None

    # Find the last clean version (cheap probe first, full read only once)
    clean_version = None
    for v in versions:
        ver = v["version"]
        try:
            ds_v = lance.dataset(lance_path, version=ver)
            ds_v.to_table(limit=1)  # cheap validation — don't load full dataset
            clean_version = ver
            break
        except Exception:
            continue

    if clean_version is None:
        logger.error("No readable version found — manual recovery needed")
        return None

    # Full read only for the validated version
    clean_table = lance.dataset(lance_path, version=clean_version).to_table()
    logger.info(
        "Found clean version %d with %d rows — rebuilding table",
        clean_version, clean_table.num_rows,
    )

    # Move corrupt dataset aside and write clean one
    corrupt_path = lance_path + ".corrupt"
    if Path(corrupt_path).exists():
        shutil.rmtree(corrupt_path)
    shutil.move(lance_path, corrupt_path)
    lance.write_dataset(clean_table, lance_path)

    # Verify the new dataset
    ds_new = lance.dataset(lance_path)
    _ = ds_new.to_table(limit=1)
    logger.info(
        "Recovery complete: %d rows restored. Corrupt backup at %s",
        ds_new.count_rows(), corrupt_path,
    )

    # Return a fresh store pointing at the rebuilt table
    return LanceDBStore(index_root, table_name)


# --- Flow ---


@flow(name="index_vault_flow")
def index_vault_flow(config_path: str = "config.yaml") -> None:
    """Scan vault, diff with store, process new/updated docs, delete removed, log stats."""
    _RUNTIME.clear()
    import time
    logger = get_run_logger()
    config = load_config(config_path)
    index_root = Path(config["index_root"])
    chunk_cfg = config.get("chunking", {})

    # --- Build components from config (stored in _RUNTIME for tasks) ---
    table_name = config.get("lancedb", {}).get("table", "chunks")
    store = LanceDBStore(index_root, table_name)

    # Persistent document ID registry
    doc_id_store = DocIDStore(Path(index_root) / "doc_registry.db")

    embed_provider = build_embed_provider(config)

    splitter = SentenceSplitter(
        chunk_size=chunk_cfg.get("max_chars", 1800),
        chunk_overlap=chunk_cfg.get("overlap", 200),
    )

    # OCR provider (may be None if disabled or provider="none")
    ocr_provider = build_ocr_provider(config)
    if ocr_provider:
        ocr_cfg = config.get("ocr", {})
        extract_name = ocr_cfg.get("extract", {}).get("provider", ocr_cfg.get("provider", "none"))
        describe_name = ocr_cfg.get("describe", {}).get("provider", ocr_cfg.get("provider", "none"))
        if extract_name == describe_name:
            logger.info("OCR enabled: %s", extract_name)
        else:
            logger.info("OCR enabled: extract=%s, describe=%s", extract_name, describe_name)
    else:
        logger.info("OCR disabled (set ocr.enabled=true in config to enable)")

    # --- Build Sources from config (always has a 'sources' key via Task 3 shim) ---
    from sources import build_source

    sources_cfg = config.get("sources", [])
    if not sources_cfg:
        raise ValueError("No sources configured. Set 'sources:' or legacy 'documents_root:' in config.yaml")

    pdf_cfg_for_sources = config.get("pdf", {})
    all_sources = [
        build_source(s, registry=doc_id_store, pdf_config=pdf_cfg_for_sources)
        for s in sources_cfg
    ]

    # Inject OCR provider into filesystem sources that support it
    for src in all_sources:
        if hasattr(src, "set_ocr_provider"):
            src.set_ocr_provider(ocr_provider)

    # Semantic splitter (optional — for large sections that need topic-boundary detection)
    semantic_cfg = chunk_cfg.get("semantic", {})
    semantic_splitter = None
    semantic_threshold = 0
    if semantic_cfg.get("enabled", False):
        try:
            from llama_index.core.node_parser import SemanticSplitterNodeParser

            # Build dedicated embed provider for semantic chunking (defaults to main provider)
            from providers.embed.semantic_adapter import SemanticEmbeddingAdapter

            sem_provider = semantic_cfg.get("provider")
            if sem_provider == "ollama":
                from providers.embed.ollama_embed import OllamaEmbedProvider

                sem_embed_provider = OllamaEmbedProvider(
                    base_url=semantic_cfg.get("base_url", "http://localhost:11434"),
                    model_name=semantic_cfg.get("model", "qwen3-embedding:0.6b"),
                )
                semantic_embed = SemanticEmbeddingAdapter(sem_embed_provider)
            else:
                semantic_embed = SemanticEmbeddingAdapter(embed_provider)

            semantic_splitter = SemanticSplitterNodeParser(
                buffer_size=semantic_cfg.get("buffer_size", 1),
                breakpoint_percentile_threshold=semantic_cfg.get("breakpoint_percentile", 95),
                embed_model=semantic_embed,
            )
            semantic_threshold = semantic_cfg.get(
                "threshold", chunk_cfg.get("max_chars", 1800) * 2
            )
            logger.info(
                "Semantic chunking enabled (model=%s, threshold=%d chars)",
                semantic_embed.model_name, semantic_threshold,
            )
        except Exception as exc:
            logger.warning("Failed to load semantic chunking model, falling back to SentenceSplitter: %s", exc)
            _RUNTIME.setdefault("_warnings", []).append(f"semantic_chunking_failed: {exc}")

    # Taxonomy store (optional — provides controlled vocabulary for enrichment)
    taxonomy_store = None
    try:
        from core.taxonomy import load_taxonomy_store, sync_folder_taxonomy_from_sources
        taxonomy_store = load_taxonomy_store(config)
        sync_stats = sync_folder_taxonomy_from_sources(taxonomy_store, all_sources)
        tax_count = taxonomy_store.count()
        if tax_count > 0:
            logger.info("Taxonomy store loaded (%d entries)", tax_count)
            if sync_stats.get("added", 0):
                logger.info(
                    "Taxonomy folder sync added=%d existing=%d discovered=%d sources=%d",
                    sync_stats.get("added", 0),
                    sync_stats.get("existing", 0),
                    sync_stats.get("discovered", 0),
                    sync_stats.get("sources", 0),
                )
        else:
            taxonomy_store = None
            logger.info("Taxonomy store empty, skipping taxonomy-guided enrichment")
    except Exception as exc:
        logger.info("Taxonomy store not available: %s", exc)

    # LLM enrichment (optional — extracts summary, entities, topics from each doc)
    llm_generator = None
    enrichment_cfg = config.get("enrichment", {})
    if enrichment_cfg.get("enabled", False):
        try:
            from providers.llm import build_llm_provider
            llm_generator = build_llm_provider(config)
            if llm_generator:
                logger.info(
                    "LLM enrichment enabled (model=%s)",
                    enrichment_cfg.get("model", "qwen3:14b-udq6"),
                )
        except Exception as exc:
            logger.warning("Failed to load LLM enrichment model: %s", exc)

    _RUNTIME["store"] = store
    _RUNTIME["doc_id_store"] = doc_id_store
    _RUNTIME["embed_provider"] = embed_provider
    _RUNTIME["splitter"] = splitter
    _RUNTIME["semantic_splitter"] = semantic_splitter
    _RUNTIME["semantic_threshold"] = semantic_threshold
    _RUNTIME["ocr_provider"] = ocr_provider
    _RUNTIME["llm_generator"] = llm_generator
    _RUNTIME["taxonomy_store"] = taxonomy_store
    _RUNTIME["config"] = config
    _RUNTIME["sources_by_name"] = {s.name: s for s in all_sources}

    # --- Migration: first run with existing LanceDB data but empty registry ---
    # If the registry is empty but the store has data, the old doc_ids were paths.
    # Wipe the LanceDB table to force full re-index with new persistent IDs.
    if doc_id_store.count() == 0:
        existing_doc_ids = store.list_doc_ids()
        if existing_doc_ids:
            logger.info(
                "Migration: registry empty but store has %d docs — wiping table for re-index",
                len(existing_doc_ids),
            )
            import lancedb as _ldb
            try:
                db = _ldb.connect(str(index_root))
                db.drop_table(table_name)
            except Exception as exc:
                logger.warning("Failed to drop table during migration: %s", exc)
            store = LanceDBStore(index_root, table_name)
            _RUNTIME["store"] = store

    # --- Run pipeline ---
    t0 = time.perf_counter()

    # Multi-source scan: iterate over all configured sources, namespace doc_ids
    all_records: list[dict] = []
    source_records_by_ns_doc_id: dict[str, object] = {}  # namespaced doc_id → SourceRecord
    for src in all_sources:
        for rec in src.scan():
            ns_doc_id = f"{src.name}::{rec.doc_id}"
            all_records.append({
                "doc_id": ns_doc_id,
                "rel_path": rec.natural_key,
                "abs_path": rec.metadata.get("abs_path", rec.natural_key),
                "mtime": rec.mtime,
                "size": rec.size,
                "ext": rec.metadata.get("ext", ""),
                "source_type": rec.source_type,
                "source_name": src.name,
            })
            source_records_by_ns_doc_id[ns_doc_id] = rec
            # Register the namespaced doc_id in the persistent registry so that:
            # 1. all_mappings() returns {namespaced_id: rel_path} for test/tool use
            # 2. distinct_source_names() can enumerate all sources that have indexed docs
            doc_id_store.register(ns_doc_id, rec.natural_key, source_name=src.name)

    _RUNTIME["source_records_by_ns_doc_id"] = source_records_by_ns_doc_id
    scanned = all_records

    stored_mtimes = store.list_doc_mtimes()
    to_add_or_update, to_delete = diff_index_task(scanned, stored_mtimes)

    concurrency = int(enrichment_cfg.get("concurrency", 1) or 1)
    if concurrency > 1:
        logger.info("Processing %d docs with concurrency=%d", len(to_add_or_update), concurrency)
    failed_docs = _process_docs(to_add_or_update, concurrency=concurrency)

    if failed_docs:
        logger.warning("Failed to process %d docs: %s", len(failed_docs), failed_docs[:20])

    delete_docs_task(to_delete)

    # Clean up deleted doc IDs from the persistent registry
    if to_delete:
        for did in to_delete:
            try:
                doc_id_store.delete(did)
            except Exception:
                pass

    # Rebuild FTS index for keyword search (BM25/tantivy) after data changes.
    # Also self-heal no-op runs when the FTS index is missing or unusable.
    should_rebuild_fts = bool(to_add_or_update or to_delete)
    if not should_rebuild_fts and not store.fts_available():
        logger.info("FTS index unavailable after no-op diff — rebuilding FTS index")
        should_rebuild_fts = True

    if should_rebuild_fts:
        logger.info("Rebuilding FTS index...")
        try:
            store.create_fts_index()
        except Exception as exc:
            logger.error("FTS index rebuild failed: %s", exc)
            _RUNTIME.setdefault("_warnings", []).append(f"fts_rebuild_failed: {exc}")

    run_seconds = time.perf_counter() - t0
    index_stats_task(len(to_add_or_update), len(to_delete), run_seconds)

    # Read final counts — auto-recover if table is corrupt
    try:
        doc_count = len(store.list_doc_ids())
        chunk_count = store.count_chunks()
    except Exception as exc:
        logger.error("Post-index read failed (possible corruption): %s", exc)
        _RUNTIME.setdefault("_warnings", []).append(f"corruption_detected: {exc}")
        recovered_store = _recover_corrupt_table(index_root, table_name, logger)
        if recovered_store:
            store = recovered_store
            _RUNTIME["store"] = store
            try:
                store.create_fts_index()
            except Exception as fts_exc:
                logger.warning("FTS rebuild after recovery failed: %s", fts_exc)
            doc_count = len(store.list_doc_ids())
            chunk_count = store.count_chunks()
            _RUNTIME.setdefault("_warnings", []).append("auto_recovery_succeeded")
        else:
            logger.error("Auto-recovery failed — manual intervention needed")
            raise

    write_index_metadata_task(
        index_root, doc_count, chunk_count,
        failed_docs or None,
        _RUNTIME.get("_warnings") or None,
    )
    logger.info(f"index_vault_flow finished in {run_seconds:.1f}s")
