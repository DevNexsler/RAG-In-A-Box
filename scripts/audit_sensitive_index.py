#!/usr/bin/env python3
"""Audit LanceDB for credential shapes; optionally delete affected docs for safe reindex.

Output contains counts and credential categories only. It never prints chunk text,
metadata values, or document identifiers.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.sensitive_content import (  # noqa: E402
    find_sensitive_content,
    sanitize_metadata,
)


@dataclass(frozen=True)
class SensitiveIndexAudit:
    scanned_chunks: int
    flagged_chunks: int
    flagged_docs: int
    finding_counts: dict[str, int]
    doc_ids: tuple[str, ...] = field(repr=False)

    def public_report(self) -> dict[str, Any]:
        return {
            "scanned_chunks": self.scanned_chunks,
            "flagged_chunks": self.flagged_chunks,
            "flagged_docs": self.flagged_docs,
            "finding_counts": dict(sorted(self.finding_counts.items())),
        }


def audit_rows(rows: Iterable[dict[str, Any]]) -> SensitiveIndexAudit:
    """Audit projected Lance rows without retaining or reporting secret values."""
    scanned_chunks = 0
    flagged_chunks = 0
    doc_ids: set[str] = set()
    finding_counts: Counter[str] = Counter()
    for row in rows:
        scanned_chunks += 1
        text_kinds = {
            finding.kind
            for finding in find_sensitive_content(str(row.get("text") or ""))
        }
        metadata = row.get("metadata")
        _clean_metadata, metadata_kinds = sanitize_metadata(
            metadata if isinstance(metadata, dict) else {}
        )
        kinds = text_kinds | set(metadata_kinds)
        if not kinds:
            continue
        flagged_chunks += 1
        doc_id = str(row.get("doc_id") or "")
        if doc_id:
            doc_ids.add(doc_id)
        finding_counts.update(kinds)
    return SensitiveIndexAudit(
        scanned_chunks=scanned_chunks,
        flagged_chunks=flagged_chunks,
        flagged_docs=len(doc_ids),
        finding_counts=dict(finding_counts),
        doc_ids=tuple(sorted(doc_ids)),
    )


def remediate_sensitive_docs(store: Any, audit: SensitiveIndexAudit) -> int:
    """Delete affected indexed docs so normal source diff reindexes under policy."""
    if not audit.doc_ids:
        return 0
    store.delete_by_doc_ids(list(audit.doc_ids))
    return len(audit.doc_ids)


def _load_rows(index_root: Path, table_name: str) -> list[dict[str, Any]]:
    import lancedb

    table = lancedb.connect(str(index_root)).open_table(table_name)
    return (
        table.search()
        .select(["doc_id", "text", "metadata"])
        .limit(10_000_000)
        .to_list()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, default=Path("/data/index"))
    parser.add_argument("--table", default="chunks")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="delete flagged docs for source-backed reindex (default: audit only)",
    )
    args = parser.parse_args(argv)

    audit = audit_rows(_load_rows(args.index_root, args.table))
    report = audit.public_report()
    report["mode"] = "apply" if args.apply else "audit"

    if args.apply:
        from lancedb_store import LanceDBStore

        store = LanceDBStore(args.index_root, args.table)
        report["remediated_docs"] = remediate_sensitive_docs(store, audit)

    print(json.dumps(report, sort_keys=True))
    if audit.flagged_docs and not args.apply:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
