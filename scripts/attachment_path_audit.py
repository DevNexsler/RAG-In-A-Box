#!/usr/bin/env python3
"""Fail-closed audit for attachment context indexing and media retrieval."""

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from check_tool_coverage import _iter_jsonl  # noqa: E402


INDEX_CHILDREN = (
    "extract",
    "attachment.context.resolve",
    "embed",
)
STORE_CHILDREN = ("store.insert", "store.upsert")
SEARCH_CHILDREN = (
    "search.media_recall",
    "search.media_quota",
)


def audit_attachment_path(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = run_dir / "attachment-path-evidence.json"
    output_path = run_dir / "attachment-path-audit.json"
    failed: list[str] = []
    checkpoints = {name: 0 for name in (*INDEX_CHILDREN, "store.write", *SEARCH_CHILDREN)}

    try:
        evidence = json.loads(evidence_path.read_text())
    except (OSError, json.JSONDecodeError):
        evidence = None
        failed.append("evidence:missing")

    spans: list[dict] = []
    traces_dir = run_dir / "traces"
    if traces_dir.is_dir():
        for path in sorted(traces_dir.rglob("*.jsonl")):
            try:
                spans.extend(_iter_jsonl(path))
            except OSError:
                continue

    if evidence is not None:
        for name, passed in sorted((evidence.get("checks") or {}).items()):
            if passed is not True:
                failed.append(f"evidence:{name}")

        target_doc_id = str(evidence.get("target_doc_id") or "")
        process_spans = [
            span for span in spans
            if span.get("name") == "process_doc"
            and str((span.get("attributes") or {}).get("doc_id") or "") == target_doc_id
        ]
        process_ids = {
            (span.get("trace_id"), span.get("span_id")) for span in process_spans
        }
        for name in INDEX_CHILDREN:
            checkpoints[name] = sum(
                (span.get("trace_id"), span.get("parent_span_id")) in process_ids
                for span in spans if span.get("name") == name
            )
            if checkpoints[name] == 0:
                failed.append(f"span:{name}")

        checkpoints["store.write"] = sum(
            (span.get("trace_id"), span.get("parent_span_id")) in process_ids
            for span in spans if span.get("name") in STORE_CHILDREN
        )
        if checkpoints["store.write"] == 0:
            failed.append("span:store.write")

        search_ids = {
            (span.get("trace_id"), span.get("span_id"))
            for span in spans if span.get("name") == "search.hybrid"
        }
        for name in SEARCH_CHILDREN:
            checkpoints[name] = sum(
                (span.get("trace_id"), span.get("parent_span_id")) in search_ids
                for span in spans if span.get("name") == name
            )
            if checkpoints[name] == 0:
                failed.append(f"span:{name}")

    report = {
        "passed": not failed,
        "target_doc_id": (evidence or {}).get("target_doc_id", ""),
        "failed_checks": failed,
        "checkpoints": checkpoints,
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    report = audit_attachment_path(args.run_dir)
    verdict = "PASS" if report["passed"] else "FAIL"
    print(f"attachment-path-audit: {verdict} -> {Path(args.run_dir) / 'attachment-path-audit.json'}")
    if report["failed_checks"]:
        print("failed: " + ", ".join(report["failed_checks"]))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
