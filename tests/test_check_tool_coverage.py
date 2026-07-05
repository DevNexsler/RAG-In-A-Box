# NOTE: `scripts` has no __init__.py — this import works via conftest's sys.path
# insert + namespace packages. Do not "fix" by adding __init__.py.
import pytest

import scripts.check_tool_coverage as ctc
from scripts.check_tool_coverage import (
    build_matrix,
    check_coverage,
    check_tool_spans,
    load_coverage,
    load_span_names,
)


# --- check_coverage: client-side (coverage JSONL vs discovered tools) --------

def test_check_coverage_reports_missing():
    assert check_coverage(discovered={"a", "b"}, covered={"a"}) == {"b"}


def test_check_coverage_ok_when_all_covered():
    # extra covered tools (not discovered) must not upset the check
    assert check_coverage(discovered={"a"}, covered={"a", "extra"}) == set()


# --- check_tool_spans: server-side (mcp.tool.<name> spans vs discovered) -----

def test_check_tool_spans_reports_untraced():
    untraced = check_tool_spans(
        discovered={"a", "b"}, span_names=["mcp.tool.a", "other.span"]
    )
    assert untraced == {"b"}


def test_check_tool_spans_ok_when_all_traced():
    names = ["mcp.tool.a", "mcp.tool.a", "mcp.tool.b", "http.probe"]
    assert check_tool_spans(discovered={"a", "b"}, span_names=names) == set()


# --- build_matrix: tool -> sorted covering tests + span count ----------------

def test_build_matrix_tests_sorted_and_spans_counted():
    records = [
        {"tool": "a", "test": "t2"},
        {"tool": "a", "test": "t1"},
        {"tool": "a", "test": "t1"},  # duplicate call from the same test
        {"tool": "not_discovered", "test": "tx"},
    ]
    span_names = ["mcp.tool.a", "mcp.tool.a", "other.span"]
    matrix = build_matrix(discovered={"a", "b"}, coverage_records=records,
                          span_names=span_names)
    assert matrix == {
        "a": {"tests": ["t1", "t2"], "span_count": 2},
        "b": {"tests": [], "span_count": 0},
    }


# --- loaders: malformed JSONL lines skipped without crashing ------------------

def test_load_coverage_skips_malformed_lines(tmp_path):
    p = tmp_path / "cov.jsonl"
    p.write_text(
        '{"tool": "a", "test": "t"}\n'
        "not json at all\n"
        '{"no_tool_key": 1}\n'
        "\n"
        '["a list, not a dict"]\n'
        '{"tool": 123}\n'
        '{"tool": "b", "test": "u"}\n'
    )
    records = load_coverage(p)
    assert [(r["tool"], r["test"]) for r in records] == [("a", "t"), ("b", "u")]


def test_load_coverage_missing_file_is_empty(tmp_path):
    assert load_coverage(tmp_path / "does-not-exist.jsonl") == []


def test_load_span_names_recursive_and_skips_malformed(tmp_path):
    traces = tmp_path / "traces"
    deep = traces / "sub"
    deep.mkdir(parents=True)
    (traces / "pid1.jsonl").write_text(
        '{"name": "mcp.tool.x", "trace_id": "t1"}\n'
        "garbage line\n"
        '{"no_name": true}\n'
    )
    (deep / "pid2.jsonl").write_text('{"name": "other.span"}\n')
    (traces / "ignored.txt").write_text('{"name": "mcp.tool.never"}\n')
    assert sorted(load_span_names(traces)) == ["mcp.tool.x", "other.span"]


def test_load_span_names_missing_dir_is_empty(tmp_path):
    assert load_span_names(tmp_path / "no-traces-here") == []


# --- main: empty discovery must never greenlight the gate ---------------------

def test_main_fails_on_zero_discovered_tools(tmp_path, monkeypatch):
    # A broken/empty list_tools would otherwise make every check vacuously
    # pass (0 tools -> nothing uncovered, nothing untraced).
    monkeypatch.setattr(ctc, "discover_tools", lambda url, key: set())
    with pytest.raises(SystemExit, match="returned 0 tools"):
        ctc.main(["--run-dir", str(tmp_path)])
