#!/usr/bin/env python3
"""Two-sided tool-coverage check for the staging-e2e gate tier.

Every tool discovered on the live staging MCP endpoint (list_tools — the
source of truth, so new tools automatically require coverage) must be:

  1. covered client-side  — at least one successful e2e call recorded in the
     coverage JSONL (tests/e2e/client.py RecordingSession appends a line per
     successful call);
  2. traced server-side   — at least one `mcp.tool.<name>` span present in the
     trace artifacts collected from the container (<run_dir>/traces/**/*.jsonl).

A tool covered client-side but missing server-side spans means traceability is
broken; it is reported separately from an uncovered tool. Either failure mode
exits nonzero (fails the staging-e2e tier). Also writes the per-tool matrix to
<run_dir>/tool-coverage.json for the gate report.

Must run INSIDE the compose window (the stack up) so list_tools is reachable.
Usage: python scripts/check_tool_coverage.py --run-dir <dir>
"""
import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

SPAN_PREFIX = "mcp.tool."

# Defaults mirror tests/e2e/client.py exactly: the e2e suite honors these env
# vars, so the checker must read the SAME coverage file / endpoint — otherwise
# an env override would make it compare against a stale default path (silent
# false green on the client side).
DEFAULT_COVERAGE_FILE = os.environ.get("E2E_COVERAGE_FILE", ".evals/e2e-tool-coverage.jsonl")
DEFAULT_MCP_URL = os.environ.get("E2E_BASE_URL", "http://localhost:17788") + "/mcp"
DEFAULT_API_KEY = os.environ.get("E2E_API_KEY", "staging-test-key")


# --- pure logic (unit-tested) -------------------------------------------------

def check_coverage(discovered: set, covered: set) -> set:
    """Tools discovered on the server but never called by any e2e test."""
    return set(discovered) - set(covered)


def _strip_spans(span_names: list) -> list:
    """Tool names extracted from mcp.tool.<name> spans (other spans dropped)."""
    return [n[len(SPAN_PREFIX):] for n in span_names if n.startswith(SPAN_PREFIX)]


def check_tool_spans(discovered: set, span_names: list) -> set:
    """Tools discovered on the server with no mcp.tool.<name> span on disk."""
    return set(discovered) - set(_strip_spans(span_names))


def build_matrix(discovered: set, coverage_records: list, span_names: list) -> dict:
    """Per discovered tool: sorted covering test names + server span count."""
    tests_by_tool = {}
    for rec in coverage_records:
        tests_by_tool.setdefault(rec["tool"], set()).add(rec.get("test", "?"))
    span_counts = Counter(_strip_spans(span_names))
    return {
        tool: {
            "tests": sorted(tests_by_tool.get(tool, ())),
            "span_count": span_counts.get(tool, 0),
        }
        for tool in sorted(discovered)
    }


def _iter_jsonl(path: Path):
    """Yield parsed dicts from a JSONL file, skipping malformed/non-dict lines."""
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(rec, dict):
            yield rec


def load_coverage(path) -> list:
    """Load coverage records ({"tool": ..., "test": ...}); skip malformed lines."""
    path = Path(path)
    if not path.exists():
        return []
    return [rec for rec in _iter_jsonl(path) if isinstance(rec.get("tool"), str)]


def load_span_names(traces_dir) -> list:
    """All span names from every *.jsonl under traces_dir (recursive)."""
    traces_dir = Path(traces_dir)
    if not traces_dir.is_dir():
        return []
    names = []
    for f in sorted(traces_dir.rglob("*.jsonl")):
        names.extend(
            rec["name"] for rec in _iter_jsonl(f) if isinstance(rec.get("name"), str)
        )
    return names


# --- live MCP discovery (inlined from the tests/e2e/client.py pattern; this
# script runs standalone, so it must not depend on test-package imports) ------

async def _list_tools(mcp_url: str, api_key: str) -> set:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    headers = {"Authorization": f"Bearer {api_key}"}
    async with streamablehttp_client(mcp_url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return {t.name for t in result.tools}


def discover_tools(mcp_url: str, api_key: str) -> set:
    try:
        return asyncio.run(_list_tools(mcp_url, api_key))
    except BaseException as exc:  # anyio wraps failures in BaseExceptionGroup
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise SystemExit(
            f"FAIL tool-coverage: cannot list tools from {mcp_url} — this check "
            f"must run inside the compose window with the staging stack up "
            f"({type(exc).__name__}: {exc})"
        ) from exc


# --- CLI -----------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True,
                        help="gate run dir (reads traces/, writes tool-coverage.json)")
    parser.add_argument("--coverage-file", default=DEFAULT_COVERAGE_FILE)
    parser.add_argument("--mcp-url", default=DEFAULT_MCP_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    discovered = discover_tools(args.mcp_url, args.api_key)
    if not discovered:
        # 0 discovered tools makes both checks vacuously pass ("0/0 covered")
        # — a broken or empty list_tools must never greenlight the gate.
        raise SystemExit(
            f"FAIL tool-coverage: list_tools returned 0 tools from {args.mcp_url}"
        )
    records = load_coverage(args.coverage_file)
    span_names = load_span_names(run_dir / "traces")

    covered = {r["tool"] for r in records}
    uncovered = check_coverage(discovered, covered)
    untraced = check_tool_spans(discovered, span_names)
    matrix = build_matrix(discovered, records, span_names)

    n = len(discovered)
    report = {
        "discovered": n,
        "covered": n - len(uncovered),
        "traced": n - len(untraced),
        "uncovered": sorted(uncovered),
        "untraced": sorted(untraced),
        "tools": matrix,
    }
    out = run_dir / "tool-coverage.json"
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"tool-coverage: {n} tools discovered; "
          f"{report['covered']}/{n} covered (client), "
          f"{report['traced']}/{n} traced (server) -> {out}")
    width = max((len(t) for t in matrix), default=0)
    for tool, row in matrix.items():
        mark = "ok" if tool not in uncovered and tool not in untraced else "!!"
        print(f"  {mark} {tool:<{width}}  tests={len(row['tests'])}  "
              f"spans={row['span_count']}")

    failed = False
    if uncovered:
        print(f"FAIL tool-coverage: UNCOVERED (no client-side e2e call): "
              f"{', '.join(sorted(uncovered))}")
        failed = True
    if untraced:
        print(f"FAIL tool-coverage: UNTRACED (no {SPAN_PREFIX}<name> span in "
              f"trace artifacts — broken traceability): {', '.join(sorted(untraced))}")
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
