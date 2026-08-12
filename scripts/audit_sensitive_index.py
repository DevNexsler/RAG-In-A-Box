#!/usr/bin/env python3
"""Audit LanceDB for credential shapes; optionally delete affected docs for safe reindex.

Output contains counts and credential categories only. It never prints chunk text,
metadata values, or document identifiers.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

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


def remediate_sensitive_docs(
    store: Any,
    audit: SensitiveIndexAudit,
    *,
    live_doc_ids: set[str],
    orphan_policy: str = "abort",
) -> int:
    """Delete source-backed docs, requiring an explicit decision for orphans."""
    if not audit.doc_ids:
        return 0

    orphan_count = len(set(audit.doc_ids) - live_doc_ids)
    if orphan_count and orphan_policy != "drop":
        raise RuntimeError(
            f"{orphan_count} source-missing documents require explicit orphan policy"
        )
    store.delete_by_doc_ids(list(audit.doc_ids))
    return len(audit.doc_ids)


def _load_rows(
    index_root: Path, table_name: str, *, batch_size: int = 1024
) -> Iterator[dict[str, Any]]:
    """Stream projected Lance rows in bounded batches."""
    import lancedb

    table = lancedb.connect(str(index_root)).open_table(table_name)
    scanner = table.to_lance().scanner(
        columns=["doc_id", "text", "metadata"], batch_size=batch_size
    )
    for batch in scanner.to_batches():
        yield from batch.to_pylist()


def _filesystem_live_doc_ids(
    source_config: dict[str, Any],
    target_doc_ids: set[str],
    index_root: Path,
) -> set[str]:
    from flow_index_vault import _is_communication_sidecar, _matches_any

    registry_path = index_root / "doc_registry.db"
    if not registry_path.is_file():
        return set()

    connection = sqlite3.connect(
        f"file:{registry_path}?mode=ro", uri=True
    )
    try:
        rows = connection.execute(
            "SELECT doc_id, rel_path, source_name FROM doc_registry"
        )
        root = Path(source_config["root"])
        source_name = str(source_config["name"])
        scan_config = source_config.get("scan", {})
        include = scan_config.get("include", ["**/*.md"])
        exclude = scan_config.get("exclude", [])
        live: set[str] = set()
        for stored_doc_id, rel_path, stored_source_name in rows:
            namespaced = (
                str(stored_doc_id)
                if "::" in str(stored_doc_id)
                else f"{stored_source_name or source_name}::{stored_doc_id}"
            )
            rel_path = str(rel_path)
            path = root / rel_path
            if namespaced not in target_doc_ids or not path.is_file():
                continue
            if _matches_any(rel_path, exclude) or not _matches_any(rel_path, include):
                continue
            if path.suffix.lower() == ".json" and _is_communication_sidecar(path):
                continue
            live.add(namespaced)
        return live
    finally:
        connection.close()


def resolve_live_doc_ids(
    config: dict[str, Any],
    target_doc_ids: Iterable[str],
    index_root: Path,
) -> set[str]:
    """Resolve flagged IDs against current configured sources without logging data."""
    from sources import build_source

    remaining = set(target_doc_ids)
    live: set[str] = set()
    for source_config in config.get("sources", []):
        source_name = str(source_config.get("name") or "")
        source_targets = {
            doc_id for doc_id in remaining if doc_id.startswith(f"{source_name}::")
        }
        if not source_targets:
            continue

        if source_config.get("type") == "filesystem":
            found = _filesystem_live_doc_ids(
                source_config, source_targets, index_root
            )
        else:
            source = build_source(source_config)
            found: set[str] = set()
            records = source.scan()
            try:
                for record in records:
                    namespaced = f"{source_name}::{record.doc_id}"
                    if namespaced in source_targets:
                        found.add(namespaced)
                        if found == source_targets:
                            break
            finally:
                close_records = getattr(records, "close", None)
                if close_records is not None:
                    close_records()
                source.close()

        live.update(found)
        remaining.difference_update(found)
    return live


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, default=Path("/data/index"))
    parser.add_argument("--table", default="chunks")
    parser.add_argument("--config", type=Path, default=Path("/app/config.yaml"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plan",
        action="store_true",
        help="classify flagged docs by current source presence without deleting",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="delete flagged docs after checking current configured sources",
    )
    parser.add_argument(
        "--orphan-policy",
        choices=("abort", "drop"),
        default="abort",
        help="decision for source-missing docs (default: abort)",
    )
    args = parser.parse_args(argv)

    audit = audit_rows(_load_rows(args.index_root, args.table))
    report = audit.public_report()
    report["mode"] = "apply" if args.apply else "plan" if args.plan else "audit"

    if args.plan or args.apply:
        from core.config import load_config

        live_doc_ids = resolve_live_doc_ids(
            load_config(args.config), audit.doc_ids, args.index_root
        )
        orphan_docs = audit.flagged_docs - len(live_doc_ids)
        report["source_backed_docs"] = len(live_doc_ids)
        report["source_missing_docs"] = orphan_docs
        report["orphan_policy"] = args.orphan_policy
        if args.plan:
            print(json.dumps(report, sort_keys=True))
            return 0
        if orphan_docs and args.orphan_policy == "abort":
            report["remediated_docs"] = 0
            report["status"] = "aborted"
            print(json.dumps(report, sort_keys=True))
            return 2

        from lancedb_store import LanceDBStore

        store = LanceDBStore(args.index_root, args.table)
        report["remediated_docs"] = remediate_sensitive_docs(
            store,
            audit,
            live_doc_ids=live_doc_ids,
            orphan_policy=args.orphan_policy,
        )

    print(json.dumps(report, sort_keys=True))
    if audit.flagged_docs and not args.apply:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
