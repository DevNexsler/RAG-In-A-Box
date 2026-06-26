#!/usr/bin/env python3
"""One-time maintenance: re-open OCR-timeout-capped docs in the degraded ledger.

Background
----------
Docs that degrade during indexing (OCR/vision timeouts, enrichment failures)
land in ``degraded_docs.json`` and are re-queued on later runs until they either
succeed or reach ``_DEGRADED_MAX_ATTEMPTS`` (5), after which they are abandoned
as "persistent" and never retried again (flow_index_vault._include_degraded_docs).

A batch of scan-only PDFs hit that cap purely because the deepseek OCR per-page
timeout (120s) had almost no headroom over the measured ~94s baseline for a
full-page scan — so every page timed out every run. Those docs are fully
OCR-able; the timeout was the bug. After raising ocr.timeout (120 -> 300 in
config.yaml), they should succeed — but they stay stranded because they are
already past the cap.

This script resets ``attempts`` to 0 for capped docs whose failures are ALL
OCR-timeout-class, so the next indexing run re-queues and re-OCRs them under the
new timeout. Docs capped for other reasons (e.g. enrichment_failed) are left
untouched, since the timeout change does not address them.

Usage
-----
    # inside the doc-organizer container (ledger at /data/index):
    python3 reopen_capped_ocr_docs.py                 # dry-run, shows what would change
    python3 reopen_capped_ocr_docs.py --apply         # write the change
    python3 reopen_capped_ocr_docs.py --ledger /data/index/degraded_docs.json --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Must match flow_index_vault._DEGRADED_MAX_ATTEMPTS.
DEFAULT_MAX_ATTEMPTS = 5

# Failure reasons fixable by an OCR/vision timeout bump. A doc is reopened only
# if EVERY one of its recorded reasons is in this class.
OCR_TIMEOUT_PREFIXES = ("ocr_page_failed", "ocr_describe_failed")


def _is_ocr_timeout_only(reasons: list[str]) -> bool:
    return bool(reasons) and all(
        r.startswith(OCR_TIMEOUT_PREFIXES) for r in reasons
    )


def reopen(ledger_path: Path, max_attempts: int, apply: bool) -> int:
    data = json.loads(ledger_path.read_text(encoding="utf-8"))
    docs = data.get("docs", {})

    reopened, skipped = [], []
    for doc_id, entry in docs.items():
        attempts = int(entry.get("attempts", 0))
        reasons = entry.get("reasons", [])
        if attempts < max_attempts:
            continue  # still within retry budget — re-queues on its own
        if _is_ocr_timeout_only(reasons):
            reopened.append((doc_id, attempts, reasons))
            if apply:
                entry["attempts"] = 0
        else:
            skipped.append((doc_id, attempts, reasons))

    print(f"ledger: {ledger_path}  ({len(docs)} degraded docs total)")
    print(f"capped (attempts >= {max_attempts}): "
          f"{len(reopened) + len(skipped)}\n")

    print(f"REOPEN ({len(reopened)}) — OCR-timeout-class, attempts -> 0:")
    for doc_id, attempts, reasons in reopened:
        print(f"  {doc_id}  (was {attempts})  {reasons}")

    if skipped:
        print(f"\nLEAVE CAPPED ({len(skipped)}) — other failure class, "
              f"not fixed by timeout bump:")
        for doc_id, attempts, reasons in skipped:
            print(f"  {doc_id}  (attempts {attempts})  {reasons}")

    if apply and reopened:
        ledger_path.write_text(
            json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"\nWROTE {ledger_path} — {len(reopened)} docs reopened.")
    elif reopened:
        print("\nDRY-RUN — no changes written. Re-run with --apply to commit.")
    else:
        print("\nNothing to reopen.")
    return len(reopened)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default="/data/index/degraded_docs.json",
                    type=Path, help="path to degraded_docs.json")
    ap.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
                    help="cap value to match flow_index_vault._DEGRADED_MAX_ATTEMPTS")
    ap.add_argument("--apply", action="store_true",
                    help="write changes (default is dry-run)")
    args = ap.parse_args()

    if not args.ledger.exists():
        print(f"ledger not found: {args.ledger}", file=sys.stderr)
        return 2
    reopen(args.ledger, args.max_attempts, args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
