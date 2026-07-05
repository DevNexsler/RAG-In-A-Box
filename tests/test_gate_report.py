# NOTE: `scripts` has no __init__.py — this import works via conftest's sys.path
# insert + namespace packages. Do not "fix" by adding __init__.py.
import json

import scripts.gate_report as gate_report
from scripts.gate_report import build_report


# --- fixtures -----------------------------------------------------------------

UNIT_XML = (
    '<?xml version="1.0" encoding="utf-8"?><testsuites>'
    '<testsuite name="pytest" errors="0" failures="0" skipped="1" tests="10" time="2.500">'
    '<testcase classname="tests.test_a" name="test_ok" time="0.1"/>'
    "</testsuite></testsuites>"
)

INTEGRATION_XML = (
    '<?xml version="1.0" encoding="utf-8"?><testsuites>'
    '<testsuite name="pytest" errors="0" failures="0" skipped="0" tests="4" time="8.000"/>'
    "</testsuites>"
)

E2E_XML_FAILING = (
    '<?xml version="1.0" encoding="utf-8"?><testsuites>'
    '<testsuite name="pytest" errors="1" failures="1" skipped="0" tests="3" time="4.200">'
    '<testcase classname="tests.e2e.test_x" name="test_ok" time="1.0"/>'
    '<testcase classname="tests.e2e.test_x" name="test_bad" time="1.0">'
    '<failure message="boom">tb</failure></testcase>'
    '<testcase classname="tests.e2e.test_x" name="test_err" time="1.0">'
    '<error message="kaboom">tb</error></testcase>'
    "</testsuite></testsuites>"
)

COVERAGE_PASS = {
    "discovered": 20, "covered": 20, "traced": 20,
    "uncovered": [], "untraced": [],
    "tools": {
        "file_search": {"tests": ["t1", "t2"], "span_count": 5},
        "file_facets": {"tests": ["test_facets"], "span_count": 1},
    },
}

COVERAGE_FAIL = {
    "discovered": 20, "covered": 13, "traced": 13,
    "uncovered": ["file_taxonomy_add"], "untraced": ["file_taxonomy_add"],
    "tools": {"file_taxonomy_add": {"tests": [], "span_count": 0}},
}


def _span(name, trace, start_ms, dur_ms, status="UNSET", **attrs):
    return {
        "name": name, "trace_id": trace, "span_id": f"s-{trace}-{name}",
        "parent_span_id": None, "start_ns": start_ms * 1_000_000,
        "end_ns": (start_ms + dur_ms) * 1_000_000, "status": status,
        "attributes": attrs,
    }


def make_run(tmp_path, name="20260705-120000"):
    run_dir = tmp_path / ".evals" / "gate-runs" / name
    run_dir.mkdir(parents=True)
    return run_dir


def write_spans(run_dir, spans, fname="spans-1.jsonl", extra_lines=()):
    traces = run_dir / "traces"
    traces.mkdir(exist_ok=True)
    lines = [json.dumps(s) for s in spans] + list(extra_lines)
    (traces / fname).write_text("\n".join(lines) + "\n")


# --- tiers table --------------------------------------------------------------

def test_tier_table_counts_durations_and_not_run(tmp_path):
    run_dir = make_run(tmp_path)
    (run_dir / "unit.xml").write_text(UNIT_XML)
    (run_dir / "integration.xml").write_text(INTEGRATION_XML)
    md = build_report(run_dir)
    assert "| tier | result | tests | failures | skipped | duration |" in md
    assert "| unit | pass | 10 | 0 | 1 | 2.5s |" in md
    assert "| integration | pass | 4 | 0 | 0 | 8.0s |" in md
    assert "| staging-e2e | not run | - | - | - | - |" in md
    assert "| live | not run | - | - | - | - |" in md
    assert "— PASS" in md.splitlines()[0]


def test_failing_tier_lists_failing_tests_and_fails_overall(tmp_path):
    run_dir = make_run(tmp_path)
    (run_dir / "e2e.xml").write_text(E2E_XML_FAILING)
    md = build_report(run_dir)
    # failures column = failures + errors
    assert "| staging-e2e | FAIL | 3 | 2 | 0 | 4.2s |" in md
    assert "tests.e2e.test_x::test_bad" in md
    assert "tests.e2e.test_x::test_err" in md
    assert md.splitlines()[0] == "# Gate run 20260705-120000 — FAIL"


def test_malformed_junit_does_not_crash(tmp_path):
    run_dir = make_run(tmp_path)
    (run_dir / "unit.xml").write_text("<testsuite this is not xml")
    md = build_report(run_dir)
    assert "| unit | not available | - | - | - | - |" in md


# --- tool coverage: the three states ------------------------------------------

def test_coverage_passed(tmp_path):
    run_dir = make_run(tmp_path)
    (run_dir / "tool-coverage.json").write_text(json.dumps(COVERAGE_PASS))
    md = build_report(run_dir)
    assert "20/20 covered, 20/20 traced" in md
    # per-tool rows, sorted by tool name
    assert "| file_facets | 1 | 1 |" in md
    assert "| file_search | 2 | 5 |" in md
    assert md.index("file_facets |") < md.index("file_search |")


def test_coverage_failed_lists_tools_and_fails_overall(tmp_path):
    run_dir = make_run(tmp_path)
    (run_dir / "tool-coverage.json").write_text(json.dumps(COVERAGE_FAIL))
    md = build_report(run_dir)
    assert "13/20 covered, 13/20 traced" in md
    assert "uncovered" in md and "file_taxonomy_add" in md
    assert "untraced" in md
    # coverage failure alone must flip the run to FAIL even with green junit
    assert md.splitlines()[0].endswith("— FAIL")


def test_coverage_not_run_when_json_absent(tmp_path):
    run_dir = make_run(tmp_path)
    md = build_report(run_dir)
    assert "check not run" in md


# --- document timelines ---------------------------------------------------------

def test_doc_timelines_stages_order_and_error_flag(tmp_path):
    run_dir = make_run(tmp_path)
    spans = [
        # trace t1: doc b.pdf, with an ERROR stage
        _span("process_doc", "t1", 1000, 371,
              doc_id="documents::2", rel_path="b.pdf", source="documents"),
        _span("extract", "t1", 1010, 296),
        _span("embed", "t1", 1310, 14, status="ERROR"),
        # trace t2: doc a.txt — sorts before b.pdf
        _span("process_doc", "t2", 2000, 50,
              doc_id="documents::1", rel_path="a.txt", source="documents"),
        _span("store.upsert", "t2", 2020, 10, status="OK"),
        # trace t3: no process_doc span — must not render as a document
        _span("POST", "t3", 3000, 5),
    ]
    write_spans(run_dir, spans, extra_lines=["not json {{{", ""])
    md = build_report(run_dir)
    assert "a.txt" in md and "b.pdf" in md
    assert md.index("a.txt") < md.index("b.pdf")  # sorted by rel_path
    assert "extract" in md and "296 ms" in md
    # ERROR span flagged on its row and summarized at the top of the section
    assert "⚠" in md
    assert "1 span(s) with ERROR status" in md
    # non-process_doc trace (t3, POST) not rendered as a document
    timelines = md.split("## Document timelines")[1].split("\n## ")[0]
    assert "POST" not in timelines


def test_doc_timelines_capped_at_20(tmp_path):
    run_dir = make_run(tmp_path)
    spans = []
    for i in range(25):
        spans.append(_span("process_doc", f"t{i}", 1000 + i, 10,
                           rel_path=f"doc-{i:03d}.txt"))
    write_spans(run_dir, spans)
    md = build_report(run_dir)
    assert "doc-019.txt" in md
    assert "doc-020.txt" not in md
    assert "5 more document(s) not shown" in md


def test_no_traces_dir_renders_fallback(tmp_path):
    run_dir = make_run(tmp_path)
    md = build_report(run_dir)
    assert "no trace artifacts" in md


# --- live spend ------------------------------------------------------------------

def test_live_spend_filters_by_run_start_and_aggregates(tmp_path):
    run_dir = make_run(tmp_path)  # 20260705-120000 (local time)
    llm_dir = tmp_path / ".evals" / "llm-traces"
    llm_dir.mkdir(parents=True)
    records = [
        # >26h after the run start in any timezone — always counted
        {"ts": "2026-07-07T00:00:00+00:00", "provider": "openrouter",
         "model": "gpt-x", "latency_ms": 1500.0, "success": True},
        {"ts": "2026-07-07T01:00:00+00:00", "provider": "openrouter",
         "model": "gpt-x", "latency_ms": 1500.0, "success": True},
        # >26h before the run start in any timezone — always filtered out
        {"ts": "2026-07-01T00:00:00+00:00", "provider": "openrouter",
         "model": "gpt-x", "latency_ms": 9999.0, "success": True},
    ]
    lines = [json.dumps(r) for r in records] + ["garbage line"]
    (llm_dir / "2026-07-05-openrouter-gpt-x.jsonl").write_text("\n".join(lines))
    md = build_report(run_dir)
    assert "| openrouter/gpt-x | 2 | 3.0 |" in md


def test_live_spend_none_recorded(tmp_path):
    run_dir = make_run(tmp_path)
    md = build_report(run_dir)
    assert "no live LLM calls recorded" in md


# --- incomplete runs ---------------------------------------------------------------

def test_empty_run_dir_titles_incomplete_not_pass(tmp_path):
    # A run that died in static/compose-up leaves no junit at all; the report
    # must not claim PASS for it.
    run_dir = make_run(tmp_path)
    md = build_report(run_dir)
    assert md.splitlines()[0] == "# Gate run 20260705-120000 — INCOMPLETE"


def test_coverage_fail_beats_incomplete(tmp_path):
    # zero junit files, but a failed coverage check: FAIL wins over INCOMPLETE
    run_dir = make_run(tmp_path)
    (run_dir / "tool-coverage.json").write_text(json.dumps(COVERAGE_FAIL))
    md = build_report(run_dir)
    assert md.splitlines()[0].endswith("— FAIL")


def test_unreadable_trace_file_skipped_with_warning(tmp_path, capsys):
    run_dir = make_run(tmp_path)
    spans = [_span("process_doc", "t1", 1000, 50, rel_path="ok.txt")]
    write_spans(run_dir, spans)
    # a directory matching *.jsonl raises IsADirectoryError (an OSError) on
    # read — portable stand-in for an unreadable file, no chmod needed
    (run_dir / "traces" / "bad.jsonl").mkdir()
    md = build_report(run_dir)
    assert "ok.txt" in md  # readable file still rendered
    assert "bad.jsonl" in capsys.readouterr().err


# --- generator robustness --------------------------------------------------------

def test_main_writes_report_and_returns_zero(tmp_path):
    run_dir = make_run(tmp_path)
    (run_dir / "unit.xml").write_text(UNIT_XML)
    assert gate_report.main([str(run_dir)]) == 0
    report = (run_dir / "report.md").read_text()
    assert report.startswith("# Gate run 20260705-120000")


def test_main_never_exits_nonzero_on_missing_run_dir(tmp_path):
    assert gate_report.main([str(tmp_path / "does-not-exist")]) == 0


def test_no_wall_clock_generated_at_line(tmp_path):
    run_dir = make_run(tmp_path)
    md = build_report(run_dir)
    assert "generated at" not in md.lower()


# --- result.json preferred when present -------------------------------------------

def write_result(run_dir, overall, **tier_overrides):
    tiers = {"static": "not_run", "unit": "not_run", "integration": "not_run",
             "staging-e2e": "not_run", "live": "not_run"}
    tiers.update(tier_overrides)
    (run_dir / "result.json").write_text(
        json.dumps({"tiers": tiers, "overall": overall}))


def test_title_prefers_result_json_fail_over_green_junit(tmp_path):
    # static failed (no junit artifact of its own) but unit junit is green:
    # artifact inference alone would say PASS — result.json must win.
    run_dir = make_run(tmp_path)
    (run_dir / "unit.xml").write_text(UNIT_XML)
    write_result(run_dir, "fail", static="fail", unit="pass")
    md = build_report(run_dir)
    assert md.splitlines()[0].endswith("— FAIL")


def test_title_prefers_result_json_pass_over_incomplete(tmp_path):
    # `--only static` pass: zero junit artifacts, but the run genuinely passed.
    run_dir = make_run(tmp_path)
    write_result(run_dir, "pass", static="pass")
    md = build_report(run_dir)
    assert md.splitlines()[0].endswith("— PASS")


def test_incomplete_still_fallback_without_result_json_or_junit(tmp_path):
    run_dir = make_run(tmp_path)
    md = build_report(run_dir)
    assert md.splitlines()[0].endswith("— INCOMPLETE")


def test_static_row_rendered_from_result_json(tmp_path):
    run_dir = make_run(tmp_path)
    (run_dir / "unit.xml").write_text(UNIT_XML)
    write_result(run_dir, "pass", static="pass", unit="pass")
    md = build_report(run_dir)
    assert "| static | pass | - | - | - | - |" in md
    # static is the first tier: its row precedes unit's
    assert md.index("| static |") < md.index("| unit |")


def test_static_row_fail_and_skipped_states(tmp_path):
    run_dir = make_run(tmp_path)
    write_result(run_dir, "fail", static="fail")
    md = build_report(run_dir)
    assert "| static | FAIL | - | - | - | - |" in md

    run_dir2 = make_run(tmp_path, name="20260705-130000")
    write_result(run_dir2, "fail", static="skipped")
    md2 = build_report(run_dir2)
    assert "| static | skipped | - | - | - | - |" in md2


def test_no_static_row_without_result_json(tmp_path):
    # Old artifacts (pre-result.json runs) must render exactly as before.
    run_dir = make_run(tmp_path)
    (run_dir / "unit.xml").write_text(UNIT_XML)
    md = build_report(run_dir)
    assert "| static |" not in md
    assert "— PASS" in md.splitlines()[0]


def test_unreadable_result_json_falls_back_to_artifact_inference(tmp_path):
    run_dir = make_run(tmp_path)
    (run_dir / "unit.xml").write_text(UNIT_XML)
    (run_dir / "result.json").write_text("not json {{{")
    md = build_report(run_dir)
    assert "| static |" not in md
    assert "— PASS" in md.splitlines()[0]
