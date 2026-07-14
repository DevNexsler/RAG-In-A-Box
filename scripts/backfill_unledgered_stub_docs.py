#!/usr/bin/env python3
"""One-time maintenance: ledger image docs that were indexed as metadata-only stubs.

Background (#0264)
------------------
During the 2026-07 ollama-vision outage, image docs indexed through the
targeted single-doc path (REST per-attachment / MCP file_index_document) could
come back with an empty describe() and still index "successfully" as a
metadata-only stub (EXIF header, no visual content) — with NO entry in
``degraded_docs.json``. Nothing retries such a doc: the diff sees it as
unchanged forever, and #0251's reopen only iterates existing ledger entries.
The code gap is fixed (the describe provider is always wrapped: an unconfirmed
empty raises transient and is ledgered as ``ocr_describe_failed``; the targeted
path now persists degradation outcomes), but docs stub-indexed before the fix
stay lost unless swept into the ledger.

This script scans the LanceDB ``chunks`` table for image docs whose entire
indexed text is metadata-only (context header + enrichment summary + EXIF
lines, no describe/OCR content) and adds a ``{reasons:
[ocr_describe_empty_backfill], attempts: 0}`` ledger entry for each one that
has none — so the existing retry/self-heal machinery re-describes them on the
next indexing runs. Docs already in the ledger are left untouched (capped ones
are #0251/PR #60 territory).

Usage
-----
    # inside the doc-organizer container (index at /data/index):
    python3 scripts/backfill_unledgered_stub_docs.py            # dry-run
    python3 scripts/backfill_unledgered_stub_docs.py --apply    # write ledger
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKFILL_REASON = "ocr_describe_empty_backfill"

# Every line _format_image_metadata_header (extractors.py) can emit. A chunk
# whose body consists solely of these lines carries no describe/OCR content.
_METADATA_LINE_PREFIXES = (
    "Image dimensions:",
    "Format:",
    "Camera:",
    "Date taken:",
    "Software:",
    "GPS:",
)
# Copied from message metadata by _with_communication_caption, not extracted
# from the image — its presence doesn't mean the image content was captured.
_CAPTION_PREFIX = "Communication message/caption:"


def is_metadata_stub_text(text: str) -> bool:
    """True if a chunk's text contains no describe/OCR content.

    Chunk layout (flow_index_vault._build_chunk_context):
        [Document: <title> | Topics: ...]        <- optional context line
        Summary: <enrichment summary>            <- optional, runs to blank line
        <blank>
        <body: extracted text>
    A stub's body holds only EXIF/metadata header lines (and possibly the
    communication caption) — everything derivable without seeing the image.
    """
    lines = text.strip().splitlines()
    i = 0
    if i < len(lines) and lines[i].startswith("["):
        i += 1
    if i < len(lines) and lines[i].startswith("Summary:"):
        i += 1
        while i < len(lines) and lines[i].strip():
            i += 1
    body = [ln.strip() for ln in lines[i:] if ln.strip()]
    return all(
        ln.startswith(_METADATA_LINE_PREFIXES) or ln.startswith(_CAPTION_PREFIX)
        for ln in body
    )


def find_stub_docs(rows: list[tuple[str, str]]) -> list[str]:
    """Return doc_ids whose EVERY chunk is metadata-only.

    rows: (doc_id, chunk_text) pairs for image docs.
    """
    by_doc: dict[str, bool] = {}
    for doc_id, text in rows:
        by_doc[doc_id] = by_doc.get(doc_id, True) and is_metadata_stub_text(text or "")
    return sorted(doc_id for doc_id, is_stub in by_doc.items() if is_stub)


def backfill(ledger: dict, stub_doc_ids: list[str]) -> tuple[dict, list[str]]:
    """Add attempts=0 entries for stub docs absent from the ledger.

    Returns (updated_ledger, newly_added_doc_ids). Existing entries — whatever
    their reasons/attempts — are never modified.
    """
    docs = dict(ledger.get("docs", {}))
    added = []
    for doc_id in stub_doc_ids:
        if doc_id in docs:
            continue
        docs[doc_id] = {"reasons": [BACKFILL_REASON], "attempts": 0}
        added.append(doc_id)
    return {**ledger, "docs": docs}, added


def _load_image_chunk_rows(index_root: Path, table: str) -> list[tuple[str, str]]:
    import lancedb

    db = lancedb.connect(str(index_root))
    tbl = db.open_table(table)
    df = (
        tbl.search()
        .where("metadata.source_type = 'img'")
        .select(["doc_id", "text"])
        .limit(10_000_000)
        .to_pandas()
    )
    return [(str(r["doc_id"]), str(r["text"])) for _, r in df.iterrows()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index-root", default="/data/index", type=Path,
                    help="LanceDB index root (contains degraded_docs.json)")
    ap.add_argument("--table", default="chunks", help="LanceDB table name")
    ap.add_argument("--apply", action="store_true",
                    help="write changes (default is dry-run)")
    args = ap.parse_args()

    ledger_path = args.index_root / "degraded_docs.json"
    if not ledger_path.exists():
        print(f"ledger not found: {ledger_path}", file=sys.stderr)
        return 2
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))

    rows = _load_image_chunk_rows(args.index_root, args.table)
    image_docs = {doc_id for doc_id, _ in rows}
    stubs = find_stub_docs(rows)
    already = [d for d in stubs if d in ledger.get("docs", {})]
    updated, added = backfill(ledger, stubs)

    print(f"index: {args.index_root}  image docs: {len(image_docs)}  "
          f"metadata-only stubs: {len(stubs)}")
    print(f"\nALREADY LEDGERED ({len(already)}) — left untouched:")
    for doc_id in already:
        print(f"  {doc_id}  {ledger['docs'][doc_id]}")
    print(f"\nBACKFILL ({len(added)}) — no ledger entry, adding "
          f"{{reasons: ['{BACKFILL_REASON}'], attempts: 0}}:")
    for doc_id in added:
        print(f"  {doc_id}")

    if args.apply and added:
        ledger_path.write_text(
            json.dumps(updated, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"\nWROTE {ledger_path} — {len(added)} docs enqueued for re-describe.")
    elif added:
        print("\nDRY-RUN — no changes written. Re-run with --apply to commit.")
    else:
        print("\nNothing to backfill.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
