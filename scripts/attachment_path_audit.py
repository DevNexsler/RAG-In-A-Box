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
REQUIRED_EVIDENCE_CHECKS = (
    "sidecar_detected",
    "sidecar_not_indexed",
    "before_embedded",
    "after_embedded",
    "before_query_retrieved",
    "after_query_retrieved",
    "media_query_retrieved",
)
REQUIRED_QUERY_FINGERPRINTS = ("before_context", "after_context", "media")


def audit_attachment_path(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = run_dir / "attachment-path-evidence.json"
    output_path = run_dir / "attachment-path-audit.json"
    failed: list[str] = []
    checkpoints = {
        name: 0
        for name in (
            *INDEX_CHILDREN,
            "store.write",
            "search.context.before",
            "search.context.after",
            *SEARCH_CHILDREN,
        )
    }

    try:
        loaded_evidence = json.loads(evidence_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        evidence = None
        failed.append("evidence:missing")
    else:
        evidence = loaded_evidence if isinstance(loaded_evidence, dict) else None
        if evidence is None:
            failed.append("evidence:schema")

    spans: list[dict] = []
    traces_dir = run_dir / "traces"
    if traces_dir.is_dir():
        for path in sorted(traces_dir.rglob("*.jsonl")):
            try:
                spans.extend(_iter_jsonl(path))
            except (OSError, UnicodeError):
                continue

    if evidence is not None:
        evidence_checks = evidence.get("checks")
        query_fingerprints = evidence.get("search_query_fingerprints")
        target_doc_id = str(evidence.get("target_doc_id") or "")
        schema_valid = (
            isinstance(evidence_checks, dict)
            and isinstance(query_fingerprints, dict)
            and bool(target_doc_id)
            and all(
                isinstance(query_fingerprints.get(name), str)
                and bool(query_fingerprints[name])
                for name in REQUIRED_QUERY_FINGERPRINTS
            )
        )
        if not schema_valid:
            failed.append("evidence:schema")
        else:
            for name in sorted(set(REQUIRED_EVIDENCE_CHECKS) | set(evidence_checks)):
                if evidence_checks.get(name) is not True:
                    failed.append(f"evidence:{name}")

        if not schema_valid:
            evidence_checks = {}
            query_fingerprints = {}

        process_spans = [
            span for span in spans
            if span.get("name") == "process_doc"
            and span.get("status") != "ERROR"
            and str((span.get("attributes") or {}).get("doc_id") or "") == target_doc_id
        ]
        process_ids = {
            (span.get("trace_id"), span.get("span_id")) for span in process_spans
        }
        for name in INDEX_CHILDREN:
            checkpoints[name] = sum(
                (span.get("trace_id"), span.get("parent_span_id")) in process_ids
                and (
                    name != "attachment.context.resolve"
                    or (
                        isinstance((span.get("attributes") or {}).get("before_count"), int)
                        and (span.get("attributes") or {})["before_count"] > 0
                        and isinstance((span.get("attributes") or {}).get("after_count"), int)
                        and (span.get("attributes") or {})["after_count"] > 0
                        and (span.get("attributes") or {}).get("has_context") is True
                    )
                )
                for span in spans
                if span.get("name") == name and span.get("status") != "ERROR"
            )
            if checkpoints[name] == 0:
                failed.append(f"span:{name}")

        checkpoints["store.write"] = sum(
            (span.get("trace_id"), span.get("parent_span_id")) in process_ids
            for span in spans
            if span.get("name") in STORE_CHILDREN and span.get("status") != "ERROR"
        )
        if checkpoints["store.write"] == 0:
            failed.append("span:store.write")

        search_ids_by_kind = {
            kind: {
                (span.get("trace_id"), span.get("span_id"))
                for span in spans
                if span.get("name") == "search.hybrid"
                and span.get("status") != "ERROR"
                and (span.get("attributes") or {}).get("query_fingerprint") == fingerprint
            }
            for kind, fingerprint in query_fingerprints.items()
            if kind in REQUIRED_QUERY_FINGERPRINTS
        }
        for kind, checkpoint in (
            ("before_context", "search.context.before"),
            ("after_context", "search.context.after"),
        ):
            checkpoints[checkpoint] = len(search_ids_by_kind.get(kind, set()))
            if checkpoints[checkpoint] == 0:
                failed.append(f"span:{checkpoint}")

        media_search_ids = search_ids_by_kind.get("media", set())
        for name in SEARCH_CHILDREN:
            checkpoints[name] = sum(
                (span.get("trace_id"), span.get("parent_span_id")) in media_search_ids
                and span.get("status") != "ERROR"
                and "video" in str((span.get("attributes") or {}).get("wanted_types") or "")
                and (
                    name != "search.media_recall"
                    or (
                        isinstance((span.get("attributes") or {}).get("recalled_count"), int)
                        and (span.get("attributes") or {})["recalled_count"] >= 0
                        and (
                            "video" not in str(
                                (span.get("attributes") or {}).get("missing_types") or ""
                            )
                            or (span.get("attributes") or {})["recalled_count"] > 0
                        )
                    )
                )
                and (
                    name != "search.media_quota"
                    or (
                        isinstance(
                            (span.get("attributes") or {}).get("returned_media_count"), int
                        )
                        and (span.get("attributes") or {})["returned_media_count"] > 0
                        and isinstance(
                            (span.get("attributes") or {}).get("injected_count"), int
                        )
                        and (span.get("attributes") or {})["injected_count"] >= 0
                    )
                )
                for span in spans
                if span.get("name") == name
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
