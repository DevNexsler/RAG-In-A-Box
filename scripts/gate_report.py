#!/usr/bin/env python3
"""Gate run report generator (Task 11): renders <run_dir>/report.md.

Reads whatever artifacts a gate run left behind — junit XML per tier,
tool-coverage.json, traces/*.jsonl OTEL spans, repo-level .evals/llm-traces —
and renders a single deterministic markdown report. Every section degrades to
a "not available" fallback; this script must NEVER exit nonzero (a broken
report must not fail the gate — gate.py runs it with check=False regardless).

Usage: python scripts/gate_report.py <run_dir>
"""
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

# `scripts` has no __init__.py. When run as `python scripts/gate_report.py`,
# sys.path[0] is scripts/ so the sibling import just works; under pytest
# (importlib mode, repo root on sys.path) it does not — insert the script dir
# so both entry points share check_tool_coverage's JSONL loader.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from check_tool_coverage import _iter_jsonl  # noqa: E402

# canonical tier order; static emits no junit artifact so it has no row
TIER_FILES = [("unit", "unit.xml"), ("integration", "integration.xml"),
              ("staging-e2e", "e2e.xml"), ("live", "live.xml"),
              ("e2e-real", "e2e-real.xml")]
DOC_CAP = 20
OK_STATUSES = {"UNSET", "OK", "", None}


def _load_result(run_dir: Path):
    """Parsed result.json (the gate runner's own verdict), or None.

    None when absent/unreadable/not-a-dict — the report then falls back to
    pure artifact inference, so pre-result.json run dirs render unchanged.
    """
    try:
        data = json.loads((run_dir / "result.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _parse_junit(path: Path):
    """Sum testsuite attrs + collect failing testcase names; None if unreadable."""
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return None
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "time": 0.0}
    failing = []
    for suite in suites:
        for key in ("tests", "failures", "errors", "skipped"):
            totals[key] += int(suite.get(key, 0) or 0)
        totals["time"] += float(suite.get("time", 0) or 0)
        for tc in suite.iter("testcase"):
            if any(child.tag in ("failure", "error") for child in tc):
                failing.append(f"{tc.get('classname', '?')}::{tc.get('name', '?')}")
    totals["failing"] = sorted(failing)
    return totals


def _render_tiers(run_dir: Path, result=None):
    lines = ["## Tiers", "",
             "| tier | result | tests | failures | skipped | duration |",
             "|---|---|---|---|---|---|"]
    failing, any_failed, files_found = [], False, 0
    if result is not None:
        # static produces no junit artifact; its state exists only in
        # result.json (pass/fail/skipped/not_run) — counts render as "-".
        state = str((result.get("tiers") or {}).get("static", "not_run"))
        display = {"fail": "FAIL", "not_run": "not run"}.get(state, state)
        lines.append(f"| static | {display} | - | - | - | - |")
    for tier, fname in TIER_FILES:
        path = run_dir / fname
        if not path.exists():
            lines.append(f"| {tier} | not run | - | - | - | - |")
            continue
        files_found += 1
        t = _parse_junit(path)
        if t is None:
            lines.append(f"| {tier} | not available | - | - | - | - |")
            continue
        bad = t["failures"] + t["errors"]
        result = "FAIL" if bad else "pass"
        any_failed = any_failed or bool(bad)
        lines.append(f"| {tier} | {result} | {t['tests']} | {bad} | "
                     f"{t['skipped']} | {t['time']:.1f}s |")
        failing.extend(f"- {name}" for name in t["failing"])
    if failing:
        lines += ["", "Failing tests:", ""] + failing
    return lines, any_failed, files_found


def _render_coverage(run_dir: Path):
    lines = ["## Tool coverage", ""]
    path = run_dir / "tool-coverage.json"
    if not path.exists():
        lines.append("check not run (staging-e2e pytest failed before the "
                     "coverage check, or the tier was not reached)")
        return lines, False
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        lines.append("tool-coverage.json not available (unreadable)")
        return lines, False
    n = data.get("discovered", 0)
    uncovered, untraced = data.get("uncovered", []), data.get("untraced", [])
    failed = bool(uncovered or untraced)
    verdict = "FAIL" if failed else "pass"
    lines.append(f"{data.get('covered', 0)}/{n} covered, "
                 f"{data.get('traced', 0)}/{n} traced — {verdict}")
    if uncovered:
        lines.append(f"- uncovered (no client-side e2e call): {', '.join(sorted(uncovered))}")
    if untraced:
        lines.append(f"- untraced (no mcp.tool.<name> span): {', '.join(sorted(untraced))}")
    tools = data.get("tools", {})
    if tools:
        lines += ["", "| tool | tests | spans |", "|---|---|---|"]
        for tool in sorted(tools):
            row = tools[tool]
            lines.append(f"| {tool} | {len(row.get('tests', []))} | "
                         f"{row.get('span_count', 0)} |")
    return lines, failed


def _load_spans(traces_dir: Path):
    spans = []
    if not traces_dir.is_dir():
        return spans
    for f in sorted(traces_dir.rglob("*.jsonl")):
        try:
            for rec in _iter_jsonl(f):
                if (isinstance(rec.get("name"), str) and rec.get("trace_id")
                        and isinstance(rec.get("start_ns"), int)
                        and isinstance(rec.get("end_ns"), int)):
                    spans.append(rec)
        except OSError as exc:
            # one unreadable trace file degrades this section, not the report
            print(f"WARN gate-report: skipping unreadable trace file {f}: "
                  f"{exc}", file=sys.stderr)
    return spans


def _render_timelines(run_dir: Path):
    lines = ["## Document timelines", ""]
    spans = _load_spans(run_dir / "traces")
    if not spans:
        lines.append("no trace artifacts (traces/ absent or empty)")
        return lines
    by_trace = {}
    for s in spans:
        by_trace.setdefault(s["trace_id"], []).append(s)
    docs = []  # (label, doc_id, stages)
    for trace_spans in by_trace.values():
        pd = next((s for s in trace_spans if s["name"] == "process_doc"), None)
        if pd is None:
            continue
        attrs = pd.get("attributes") or {}
        label = attrs.get("rel_path") or attrs.get("doc_id") or pd["trace_id"]
        docs.append((str(label), str(attrs.get("doc_id", "")),
                     sorted(trace_spans, key=lambda s: s["start_ns"])))
    if not docs:
        lines.append("no process_doc traces found")
        return lines
    docs.sort(key=lambda d: (d[0], d[1]))
    error_count = sum(1 for _, _, stages in docs for s in stages
                      if s.get("status") not in OK_STATUSES)
    lines.append(f"⚠ {error_count} span(s) with ERROR status"
                 if error_count else "No ERROR-status spans.")
    for label, doc_id, stages in docs[:DOC_CAP]:
        suffix = f" ({doc_id})" if doc_id and doc_id != label else ""
        lines += ["", f"### `{label}`{suffix}", "", "```"]
        for s in stages:
            dur_ms = (s["end_ns"] - s["start_ns"]) // 1_000_000
            flag = "⚠ " if s.get("status") not in OK_STATUSES else "  "
            lines.append(f"{flag}{s['name']:<28} {dur_ms:>7} ms  {s.get('status')}")
        lines.append("```")
    if len(docs) > DOC_CAP:
        lines += ["", f"... {len(docs) - DOC_CAP} more document(s) not shown"]
    return lines


def _run_start(run_dir: Path):
    """Run start as an aware datetime, parsed from the dirname (local time)."""
    try:
        return datetime.strptime(run_dir.name, "%Y%m%d-%H%M%S").astimezone()
    except ValueError:
        return None


def _render_spend(run_dir: Path):
    lines = ["## Live spend", ""]
    # repo layout: <repo>/.evals/gate-runs/<ts> -> <repo>/.evals/llm-traces
    llm_dir = run_dir.resolve().parent.parent / "llm-traces"
    start = _run_start(run_dir)
    agg = {}  # (provider, model) -> [calls, latency_ms]
    if llm_dir.is_dir():
        for f in sorted(llm_dir.glob("*.jsonl")):
            for rec in _iter_jsonl(f):
                try:
                    ts = datetime.fromisoformat(rec["ts"])
                except (KeyError, TypeError, ValueError):
                    continue
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if start is not None and ts < start:
                    continue
                key = (str(rec.get("provider", "?")), str(rec.get("model", "?")))
                entry = agg.setdefault(key, [0, 0.0])
                entry[0] += 1
                try:
                    entry[1] += float(rec.get("latency_ms", 0) or 0)
                except (TypeError, ValueError):
                    pass
    if not agg:
        lines.append("no live LLM calls recorded")
        return lines
    lines += ["| provider/model | calls | total latency s |", "|---|---|---|"]
    for (provider, model), (calls, latency_ms) in sorted(agg.items()):
        lines.append(f"| {provider}/{model} | {calls} | {latency_ms / 1000:.1f} |")
    return lines


def build_report(run_dir: Path) -> str:
    run_dir = Path(run_dir)
    result = _load_result(run_dir)
    tier_lines, tiers_failed, junit_found = _render_tiers(run_dir, result)
    coverage_lines, coverage_failed = _render_coverage(run_dir)
    overall = (result or {}).get("overall")
    if overall in ("pass", "fail"):
        # The runner's own verdict wins: it sees tiers with no junit artifact
        # (static) and skipped/not_run states that artifact inference cannot.
        # Artifact-derived failures still veto a "pass" (defense in depth).
        status = "FAIL" if (overall == "fail" or tiers_failed
                            or coverage_failed) else "PASS"
    elif tiers_failed or coverage_failed:
        status = "FAIL"
    elif junit_found == 0:
        # zero junit artifacts: the run died before producing any tier
        # evidence (static or compose-up failure) — never title it PASS
        status = "INCOMPLETE"
    else:
        status = "PASS"
    sections = [[f"# Gate run {run_dir.name} — {status}"], tier_lines,
                coverage_lines, _render_timelines(run_dir), _render_spend(run_dir)]
    return "\n\n".join("\n".join(s) for s in sections) + "\n"


def main(argv=None) -> int:
    # Never exit nonzero: a broken report must not fail the gate.
    try:
        argv = sys.argv[1:] if argv is None else argv
        if len(argv) != 1:
            print("usage: gate_report.py <run_dir>", file=sys.stderr)
            return 0
        run_dir = Path(argv[0])
        out = run_dir / "report.md"
        out.write_text(build_report(run_dir))
        print(f"gate report: {out}")
    except Exception as exc:  # noqa: BLE001 — deliberate catch-all, see above
        print(f"WARN gate-report: report generation failed: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
