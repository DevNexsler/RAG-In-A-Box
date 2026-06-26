#!/usr/bin/env python3
"""drain_unindexed_backlog.py — force-process registry-but-unindexed documents.

Background: ~2,700 'documents' attachments are registered in doc_registry.db but
absent from the LanceDB index (lost during the 2026-06 OCR-down / vision-freeze
window). The incremental indexer (file_index_update) processes new arrivals but
does NOT re-attempt these older registered-but-unindexed docs, so the backlog
never drains on its own. This driver walks that exact set (registered, canonical,
file-present, not in the index), oldest-first, and pushes each through the real
single-doc path (index_document_flow) — emitting per-doc progress JSONL so an
external monitor can confirm the backlog is actually shrinking and flag a stall.

Run INSIDE the doc-organizer container (-w /app):
    python drain_unindexed_backlog.py --dry-run          # count only, no work
    python drain_unindexed_backlog.py --limit 50         # probe batch
    python drain_unindexed_backlog.py                    # full drain
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time

sys.path.insert(0, "/app")
from core.config import load_config  # noqa: E402
from flow_index_vault import index_document_flow  # noqa: E402
from lancedb_store import LanceDBStore  # noqa: E402

CONFIG = os.environ.get("DRAIN_CONFIG", "config.yaml")
PROGRESS = os.environ.get("DRAIN_PROGRESS", "/data/index/drain_progress.jsonl")
DOCROOT = os.environ.get("DRAIN_DOCROOT", "/data/documents")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def _emit(rec: dict) -> None:
    rec["ts"] = time.time()
    with open(PROGRESS, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()


def compute_backlog(config: dict) -> list[tuple[str, str, str]]:
    """Return [(doc_id, rel_path, ext)] registered-but-unindexed, oldest-first."""
    index_root = config["index_root"]
    table = config.get("lancedb", {}).get("table", "chunks")
    idx = set(LanceDBStore(index_root, table).list_doc_mtimes())
    con = sqlite3.connect(f"{index_root}/doc_registry.db")
    rows = con.execute(
        "SELECT doc_id, rel_path, first_seen_at FROM doc_registry "
        "WHERE source_name='documents' "
        "AND (dedupe_status IS NULL OR dedupe_status='canonical') "
        "ORDER BY first_seen_at ASC"
    ).fetchall()
    con.close()
    out = []
    for did, rp, _fs in rows:
        if not rp:
            continue
        ext = os.path.splitext(rp)[1].lower()
        if ext == ".json":
            continue
        if not os.path.exists(os.path.join(DOCROOT, rp)):
            continue
        # The registry stores doc_id inconsistently (some bare "000e2", some
        # already "documents::000e2"); the LanceDB index always keys on the
        # "documents::"-prefixed form. Normalize before the membership test so
        # we don't mis-flag already-indexed docs as pending.
        key = did if did.startswith("documents::") else f"documents::{did}"
        if key in idx:
            continue
        out.append((did, rp, ext))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--images-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    config = load_config(CONFIG)
    backlog = compute_backlog(config)
    if args.images_only:
        backlog = [b for b in backlog if b[2] in IMG_EXT]
    total = len(backlog)

    if args.dry_run:
        from collections import Counter
        by_ext = Counter(b[2] for b in backlog)
        print(f"backlog total: {total}")
        print(f"by ext: {dict(by_ext.most_common())}")
        print(f"oldest 3: {[b[0] for b in backlog[:3]]}")
        return

    batch = backlog[: args.limit] if args.limit else backlog
    _emit({"event": "start", "batch": len(batch), "total_backlog": total,
           "pid": os.getpid()})
    ok = skip = fail = 0
    t_start = time.time()
    for i, (did, rp, ext) in enumerate(batch, 1):
        t0 = time.time()
        try:
            res = index_document_flow(CONFIG, target=rp, source_name="documents")
            st = res.get("status", "?")
            detail = res.get("reason", "")
        except Exception as e:  # noqa: BLE001 — log and keep draining
            st, detail = "exception", str(e)[:200]
        dt = round(time.time() - t0, 1)
        if st == "indexed":
            ok += 1
        elif st == "skipped":
            skip += 1
        else:
            fail += 1
        rate = (i / (time.time() - t_start)) * 3600.0  # docs/hour so far
        _emit({"event": "doc", "i": i, "batch": len(batch), "doc_id": did,
               "ext": ext, "status": st, "detail": detail, "elapsed_s": dt,
               "ok": ok, "skip": skip, "fail": fail,
               "docs_per_hr": round(rate, 1)})
    _emit({"event": "done", "batch": len(batch), "ok": ok, "skip": skip,
           "fail": fail, "wall_s": round(time.time() - t_start, 1)})


if __name__ == "__main__":
    main()
