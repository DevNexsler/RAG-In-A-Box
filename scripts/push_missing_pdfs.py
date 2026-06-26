#!/usr/bin/env python3
"""push_missing_pdfs.py — index the genuinely-missing (non-duplicate) PDFs.

Targets exactly the documents-source PDFs that are: file-present, >10 bytes, NOT
in the LanceDB index, and NOT a content-duplicate (per the bare-id registry row's
dedupe_status — the prefixed twin row is unreliable, see the dual-row pitfall).
Pushes each through index_document_flow(force=True) and VERIFIES real chunks were
written (status alone lies — the dedup/empty paths return "indexed" with 0 chunks).

Run inside the doc-organizer container (-w /app):
    python push_missing_pdfs.py --list      # print targets, do nothing
    python push_missing_pdfs.py             # index them, verify, log JSONL
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
import lancedb  # noqa: E402

CONFIG = "config.yaml"
OUT = "/data/index/push_pdfs.jsonl"
DOCROOT = "/data/documents"


def _key(d: str) -> str:
    return d if d.startswith("documents::") else f"documents::{d}"


def find_missing_pdfs(config: dict) -> list[tuple[str, str, int]]:
    table = config.get("lancedb", {}).get("table", "chunks")
    idx = set(LanceDBStore(config["index_root"], table).list_doc_mtimes())
    con = sqlite3.connect(f'{config["index_root"]}/doc_registry.db')
    # Dedup truth lives on the bare-id row; prefer a row that actually carries it.
    truth: dict[str, tuple] = {}
    for did, ds, cid in con.execute(
        "SELECT doc_id, dedupe_status, canonical_doc_id "
        "FROM doc_registry WHERE source_name='documents'"
    ):
        bare = did.split("::", 1)[1] if did.startswith("documents::") else did
        if bare not in truth or (ds and ds != "canonical"):
            truth[bare] = (ds, cid)
    rows = con.execute(
        "SELECT doc_id, rel_path FROM doc_registry WHERE source_name='documents' "
        "AND (dedupe_status IS NULL OR dedupe_status='canonical')"
    ).fetchall()
    con.close()
    out, seen = [], set()
    for did, rp in rows:
        if not rp or os.path.splitext(rp)[1].lower() != ".pdf":
            continue
        full = os.path.join(DOCROOT, rp)
        if not os.path.exists(full) or _key(did) in idx:
            continue
        bare = did.split("::", 1)[1] if did.startswith("documents::") else did
        if bare in seen:
            continue
        seen.add(bare)
        ds, _cid = truth.get(bare, (None, None))
        if ds == "duplicate":
            continue
        try:
            sz = os.path.getsize(full)
        except OSError:
            sz = 0
        if sz < 10:
            continue
        out.append((did, rp, sz))
    return out


def _emit(rec: dict) -> None:
    rec["ts"] = time.time()
    with open(OUT, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    config = load_config(CONFIG)
    pdfs = find_missing_pdfs(config)
    table = config.get("lancedb", {}).get("table", "chunks")

    if args.list:
        for did, rp, sz in pdfs:
            print(f"  {did:16} {sz:>9}B  {rp}")
        print(f"total: {len(pdfs)} missing PDFs")
        return

    _emit({"event": "start", "count": len(pdfs)})
    indexed = empty = failed = 0
    for i, (did, rp, sz) in enumerate(pdfs, 1):
        t0 = time.time()
        try:
            res = index_document_flow(CONFIG, target=rp, source_name="documents",
                                      force=True)
            st = res.get("status", "?")
        except Exception as e:  # noqa: BLE001
            st, res = "exception", {"error": str(e)[:300]}
        # Verify real chunks (status="indexed" can still mean 0 chunks).
        try:
            tbl = lancedb.connect(config["index_root"]).open_table(table)
            chunks = len(tbl.search().where(f"doc_id = '{_key(did)}'")
                         .limit(5).to_pandas())
        except Exception:
            chunks = -1
        dt = round(time.time() - t0, 1)
        verdict = ("INDEXED" if chunks > 0
                   else ("EMPTY/0-chunk" if st == "indexed" else "FAILED"))
        if verdict == "INDEXED":
            indexed += 1
        elif verdict.startswith("EMPTY"):
            empty += 1
        else:
            failed += 1
        _emit({"event": "pdf", "i": i, "count": len(pdfs), "doc_id": did,
               "size": sz, "status": st, "chunks": chunks, "verdict": verdict,
               "elapsed_s": dt, "rel_path": rp, "detail": res})
    _emit({"event": "done", "indexed": indexed, "empty": empty, "failed": failed})


if __name__ == "__main__":
    main()
