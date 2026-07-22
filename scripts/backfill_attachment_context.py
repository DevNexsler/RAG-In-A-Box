#!/usr/bin/env python3
"""Embed stored attachment conversation context without reprocessing media.

Default: dry-run, five documents. Use ``--apply`` to write. Run inside
doc-organizer container or another environment with same config and mounts.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attachment_context_refresh import (
    context_text_from_sidecar,
    plan_document_context,
    refresh_document_context,
)
from core.config import load_config
from core.index_write_lock import IndexWriteLockBusy, index_write_lock
from lancedb_store import open_store_with_recovery
from providers.embed import build_embed_provider


@dataclass(frozen=True)
class Candidate:
    doc_id: str
    source_type: str
    sidecar_path: Path


def candidates_from_rows(
    rows: list[dict[str, Any]],
    *,
    source_types: set[str],
    limit: int,
    doc_ids: set[str] | None = None,
) -> list[Candidate]:
    """Normalize raw Lance rows into deterministic document candidates."""
    by_doc_id: dict[str, Candidate] = {}
    for row in rows:
        metadata = row.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        doc_id = str(row.get("doc_id") or metadata.get("doc_id") or "")
        source_type = str(metadata.get("source_type") or "")
        sidecar_path = str(metadata.get("sidecar_path") or "")
        if not doc_id or source_type not in source_types or not sidecar_path:
            continue
        if doc_ids is not None and doc_id not in doc_ids:
            continue
        by_doc_id.setdefault(
            doc_id,
            Candidate(doc_id, source_type, Path(sidecar_path)),
        )
    candidates = [by_doc_id[key] for key in sorted(by_doc_id)]
    return candidates[:limit] if limit > 0 else candidates


def _load_rows(index_root: Path, table_name: str) -> list[dict[str, Any]]:
    import lancedb

    table = lancedb.connect(str(index_root)).open_table(table_name)
    return (
        table.search(None)
        .select(["doc_id", "metadata"])
        .limit(10_000_000)
        .to_list()
    )


def _emit(**payload: Any) -> None:
    print(json.dumps(payload, sort_keys=True, default=str), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", type=Path)
    parser.add_argument("--index-root", type=Path)
    parser.add_argument("--table")
    parser.add_argument("--limit", type=int, default=5, help="0 means all")
    parser.add_argument(
        "--source-type",
        action="append",
        choices=("img", "video", "audio"),
        dest="source_types",
    )
    parser.add_argument("--doc-id", action="append", dest="doc_ids")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be zero or positive")

    config = load_config(args.config)
    index_root = args.index_root or Path(config["index_root"])
    table_name = args.table or config.get("lancedb", {}).get("table", "chunks")
    source_types = set(args.source_types or ("img", "video", "audio"))
    selected_ids = set(args.doc_ids) if args.doc_ids else None
    session = (
        index_write_lock(index_root, table_name, blocking=False)
        if args.apply
        else nullcontext()
    )

    try:
        with session:
            rows = _load_rows(index_root, table_name)
            candidates = candidates_from_rows(
                rows,
                source_types=source_types,
                doc_ids=selected_ids,
                limit=args.limit,
            )
            store = open_store_with_recovery(index_root, table_name)
            embed_provider = build_embed_provider(config) if args.apply else None
            max_time_window_minutes = config.get("communication_context", {}).get(
                "max_time_window_minutes", 15
            )
            counts = {
                "selected": len(candidates),
                "changed": 0,
                "unchanged": 0,
                "no_context": 0,
                "missing": 0,
                "failed": 0,
            }
            for candidate in candidates:
                try:
                    context_text = context_text_from_sidecar(
                        candidate.sidecar_path,
                        doc_id=candidate.doc_id,
                        max_time_window_minutes=max_time_window_minutes,
                    )
                    if not context_text:
                        counts["no_context"] += 1
                        _emit(doc_id=candidate.doc_id, status="no_context")
                        continue
                    plan = plan_document_context(
                        store, candidate.doc_id, context_text
                    )
                    if plan is None:
                        counts["missing"] += 1
                        _emit(doc_id=candidate.doc_id, status="missing")
                        continue
                    if not plan.needs_refresh:
                        counts["unchanged"] += 1
                        _emit(doc_id=candidate.doc_id, status="unchanged", loc=plan.loc)
                        continue
                    if args.apply:
                        changed = refresh_document_context(
                            store,
                            embed_provider,
                            candidate.doc_id,
                            context_text,
                        )
                        status = "changed" if changed else "stale_read"
                        if not changed:
                            counts["failed"] += 1
                            _emit(doc_id=candidate.doc_id, status=status, loc=plan.loc)
                            continue
                    counts["changed"] += 1
                    _emit(
                        doc_id=candidate.doc_id,
                        status="changed" if args.apply else "would_change",
                        loc=plan.loc,
                        context_chars=len(context_text),
                    )
                except Exception as exc:
                    counts["failed"] += 1
                    _emit(doc_id=candidate.doc_id, status="failed", error=str(exc))
            if args.apply and counts["changed"]:
                store.ensure_fts_index(compact_data=False)
            _emit(mode="apply" if args.apply else "dry_run", summary=counts)
            return 1 if counts["failed"] else 0
    except IndexWriteLockBusy as exc:
        _emit(status="writer_busy", error=str(exc))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
