#!/usr/bin/env python3
"""One-time maintenance: clear registry-retired ids from the terminal degraded ledger.

Background (#2074, follow-up to #2022)
--------------------------------------
A degraded document that a successful scan can no longer resolve is aged for
``_DEGRADED_MAX_UNRESOLVED_RUNS`` runs and then escalated out of the active
retry ledger into ``degraded_unresolved.json`` (flow_index_vault, #0618). That
terminal ledger exists to hold ONE thing: a document the pipeline can no longer
account for, i.e. possible silent loss.

Prod's copy also holds 16 entries that are nothing of the sort — the
``%20``-mangled zoho_cliq ids and duplicate-delivery zoho_mail ids of #0618,
every one of them deliberately removed from the registry weeks before it was
escalated. They keep the published ``degraded_unresolved`` warning permanently
non-green and mask the first entry that is real.

Blind truncation would clear them and also discard a genuinely lost document,
which is the one thing the sink must not do. So the drop is driven by the
registry's own record instead: an entry goes only if ``retired_ids`` says *we*
retired that id. Anything the registry cannot account for is printed and kept.

#2022 stops the reconcile escalating a retired id in the first place. Run this
only once #2022 is deployed, or the same entries come straight back on the next
escalation.

Usage
-----
    # inside the doc-organizer container (index at /data/index):
    python3 scripts/clear_retired_unresolved_docs.py            # dry-run
    python3 scripts/clear_retired_unresolved_docs.py --apply    # write the ledger
    python3 scripts/clear_retired_unresolved_docs.py --index-root /data/index
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sqlite3
import sys
from pathlib import Path

DEFAULT_INDEX_ROOT = Path("/data/index")
# Both file names are owned by the flow: flow_index_vault._degraded_unresolved_path
# and the DocIDStore the indexer opens next to it.
LEDGER_NAME = "degraded_unresolved.json"
REGISTRY_NAME = "doc_registry.db"


@contextlib.contextmanager
def open_registry(registry_path: Path):
    """Open the doc registry read-only — this script never writes to it.

    Read-only matters twice over: a dry-run must not touch prod state at all,
    and ``DocIDStore`` would open the live indexer's database read-write and
    replay its schema migrations just to answer a question.
    """
    if not registry_path.is_file():
        raise FileNotFoundError(
            f"doc registry not found: {registry_path} — cannot prove any entry "
            f"was retired, so nothing may be dropped"
        )
    connection = sqlite3.connect(f"file:{registry_path}?mode=ro", uri=True)
    try:
        yield connection
    finally:
        connection.close()


def is_registry_retired(connection: sqlite3.Connection, doc_id: str) -> bool:
    """True if the registry reports this ledger key as retired.

    The ledger stores the namespaced id (``comm_messages::zoho_mail/<...>``)
    while ``retired_ids`` holds whichever form the registry row itself used:
    ``DocIDStore.delete`` retires the exact id when a row matches it, and falls
    back to the bare suffix for legacy rows reached through an
    ``all_mappings()`` key. Ask in that same order so both are recognised.
    """
    candidates = [doc_id]
    if "::" in doc_id:
        candidates.append(doc_id.split("::", 1)[-1])
    for candidate in candidates:
        row = connection.execute(
            "SELECT 1 FROM retired_ids WHERE doc_id = ?", (candidate,)
        ).fetchone()
        if row is not None:
            return True
    return False


def clear_retired(index_root: Path, apply: bool) -> tuple[dict, dict]:
    """Drop registry-retired entries from the terminal ledger; keep the rest.

    Returns ``(dropped, kept)``. With ``apply`` false nothing is written.
    """
    ledger_path = Path(index_root) / LEDGER_NAME
    if not ledger_path.is_file():
        print(f"ledger: {ledger_path} — not present, already clear.")
        return {}, {}

    # A JSONDecodeError propagates on purpose: the flow's loader treats an
    # unreadable ledger as empty, but a maintenance script that would then
    # WRITE that emptiness back must fail instead of erasing the entries.
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    docs = ledger.get("docs", {})

    with open_registry(Path(index_root) / REGISTRY_NAME) as connection:
        dropped, kept = {}, {}
        for doc_id, entry in sorted(docs.items()):
            target = dropped if is_registry_retired(connection, doc_id) else kept
            target[doc_id] = entry

    print(f"ledger: {ledger_path}  ({len(docs)} terminal entries)")
    print(f"\nDROP ({len(dropped)}) — retired in doc_registry, benign:")
    for doc_id, entry in dropped.items():
        print(f"  {doc_id}  {entry.get('reasons', [])}")
    print(f"\nKEEP ({len(kept)}) — not retired in doc_registry, possible loss:")
    for doc_id, entry in kept.items():
        print(f"  {doc_id}  {entry.get('reasons', [])}")

    if not dropped:
        print("\nNothing to clear.")
        return dropped, kept

    if apply:
        # Replace only the docs map so any other top-level field the flow keeps
        # (or adds) survives, and serialize exactly as _save_degraded_unresolved
        # does so the next run's write is not a spurious reformat.
        ledger["docs"] = kept
        ledger_path.write_text(
            json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"\nWROTE {ledger_path} — {len(dropped)} entries cleared, {len(kept)} kept.")
    else:
        print("\nDRY-RUN — no changes written. Re-run with --apply to commit.")
    return dropped, kept


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--index-root", default=DEFAULT_INDEX_ROOT, type=Path,
        help=f"index root holding {LEDGER_NAME} and {REGISTRY_NAME}",
    )
    ap.add_argument(
        "--apply", action="store_true", help="write changes (default is dry-run)"
    )
    args = ap.parse_args()

    try:
        clear_retired(args.index_root, args.apply)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
