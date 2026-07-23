import json

from scripts.attachment_path_audit import audit_attachment_path, main


def _span(name, trace_id, span_id, parent_span_id=None, **attributes):
    return {
        "name": name,
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "start_ns": 1,
        "end_ns": 2,
        "status": "UNSET",
        "attributes": attributes,
    }


def _write_fixture(run_dir, *, omit_span=None, failed_check=None):
    evidence = {
        "target_doc_id": "documents::clip",
        "search_query_fingerprints": {
            "before_context": "before-fingerprint",
            "after_context": "after-fingerprint",
            "media": "media-fingerprint",
        },
        "checks": {
            "sidecar_detected": True,
            "sidecar_not_indexed": True,
            "before_embedded": True,
            "after_embedded": True,
            "before_query_retrieved": True,
            "after_query_retrieved": True,
            "media_query_retrieved": True,
        },
    }
    if failed_check:
        evidence["checks"][failed_check] = False
    (run_dir / "attachment-path-evidence.json").write_text(json.dumps(evidence))
    spans = [
        _span("process_doc", "index", "p", None, doc_id="documents::clip"),
        _span("extract", "index", "e", "p"),
        _span(
            "attachment.context.resolve",
            "index",
            "c",
            "p",
            before_count=1,
            after_count=1,
            has_context=True,
        ),
        _span("embed", "index", "b", "p"),
        _span("store.upsert", "index", "u", "p"),
        _span("search.hybrid", "before", "sb", query_fingerprint="before-fingerprint"),
        _span("search.hybrid", "after", "sa", query_fingerprint="after-fingerprint"),
        _span("search.hybrid", "media", "sm", query_fingerprint="media-fingerprint"),
        _span(
            "search.media_recall",
            "media",
            "r",
            "sm",
            wanted_types="video",
            recalled_count=0,
        ),
        _span(
            "search.media_quota",
            "media",
            "q",
            "sm",
            wanted_types="video",
            injected_count=0,
            returned_media_count=1,
        ),
    ]
    traces = run_dir / "traces"
    traces.mkdir()
    (traces / "spans.jsonl").write_text(
        "\n".join(json.dumps(span) for span in spans if span["name"] != omit_span) + "\n"
    )


def test_attachment_path_audit_passes_only_with_all_independent_steps(tmp_path):
    _write_fixture(tmp_path)
    report = audit_attachment_path(tmp_path)
    assert report["passed"] is True
    assert report["failed_checks"] == []
    assert report["checkpoints"]["attachment.context.resolve"] == 1
    assert report["checkpoints"]["search.media_quota"] == 1


def test_attachment_path_audit_fails_closed_on_missing_span(tmp_path):
    _write_fixture(tmp_path, omit_span="embed")
    report = audit_attachment_path(tmp_path)
    assert report["passed"] is False
    assert "span:embed" in report["failed_checks"]


def test_attachment_path_audit_accepts_insert_for_new_documents(tmp_path):
    _write_fixture(tmp_path)
    trace_path = tmp_path / "traces/spans.jsonl"
    trace_path.write_text(trace_path.read_text().replace('"store.upsert"', '"store.insert"'))
    report = audit_attachment_path(tmp_path)
    assert report["passed"] is True
    assert report["checkpoints"]["store.write"] == 1


def test_attachment_path_audit_fails_closed_on_failed_e2e_assertion(tmp_path):
    _write_fixture(tmp_path, failed_check="after_query_retrieved")
    report = audit_attachment_path(tmp_path)
    assert report["passed"] is False
    assert "evidence:after_query_retrieved" in report["failed_checks"]


def test_attachment_path_audit_fails_closed_on_missing_required_evidence_check(tmp_path):
    _write_fixture(tmp_path)
    evidence_path = tmp_path / "attachment-path-evidence.json"
    evidence = json.loads(evidence_path.read_text())
    del evidence["checks"]["after_query_retrieved"]
    evidence_path.write_text(json.dumps(evidence))

    report = audit_attachment_path(tmp_path)

    assert report["passed"] is False
    assert "evidence:after_query_retrieved" in report["failed_checks"]


def test_attachment_path_audit_rejects_context_span_without_both_neighbors(tmp_path):
    _write_fixture(tmp_path)
    trace_path = tmp_path / "traces/spans.jsonl"
    trace_path.write_text(trace_path.read_text().replace('"before_count": 1', '"before_count": 0'))

    report = audit_attachment_path(tmp_path)

    assert report["passed"] is False
    assert "span:attachment.context.resolve" in report["failed_checks"]


def test_attachment_path_audit_rejects_quota_span_without_returned_media(tmp_path):
    _write_fixture(tmp_path)
    trace_path = tmp_path / "traces/spans.jsonl"
    trace_path.write_text(
        trace_path.read_text().replace('"returned_media_count": 1', '"returned_media_count": 0')
    )

    report = audit_attachment_path(tmp_path)

    assert report["passed"] is False
    assert "span:search.media_quota" in report["failed_checks"]


def test_attachment_path_audit_rejects_error_span(tmp_path):
    _write_fixture(tmp_path)
    trace_path = tmp_path / "traces/spans.jsonl"
    spans = [json.loads(line) for line in trace_path.read_text().splitlines()]
    next(span for span in spans if span["name"] == "attachment.context.resolve")["status"] = "ERROR"
    trace_path.write_text("\n".join(json.dumps(span) for span in spans) + "\n")

    report = audit_attachment_path(tmp_path)

    assert report["passed"] is False
    assert "span:attachment.context.resolve" in report["failed_checks"]


def test_attachment_path_audit_rejects_unrelated_search_spans(tmp_path):
    _write_fixture(tmp_path)
    trace_path = tmp_path / "traces/spans.jsonl"
    trace_path.write_text(trace_path.read_text().replace("media-fingerprint", "unrelated"))

    report = audit_attachment_path(tmp_path)

    assert report["passed"] is False
    assert "span:search.media_recall" in report["failed_checks"]
    assert "span:search.media_quota" in report["failed_checks"]


def test_attachment_path_audit_writes_failure_for_malformed_evidence_schema(tmp_path):
    _write_fixture(tmp_path)
    evidence_path = tmp_path / "attachment-path-evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["checks"] = ["not-an-object"]
    evidence_path.write_text(json.dumps(evidence))

    report = audit_attachment_path(tmp_path)

    assert report["passed"] is False
    assert "evidence:schema" in report["failed_checks"]
    assert json.loads((tmp_path / "attachment-path-audit.json").read_text())["passed"] is False


def test_attachment_path_audit_fails_closed_without_evidence(tmp_path):
    report = audit_attachment_path(tmp_path)
    assert report["passed"] is False
    assert report["failed_checks"] == ["evidence:missing"]


def test_attachment_path_audit_writes_failure_for_invalid_utf8_evidence(tmp_path):
    (tmp_path / "attachment-path-evidence.json").write_bytes(b"\xff")

    report = audit_attachment_path(tmp_path)

    assert report["passed"] is False
    assert "evidence:missing" in report["failed_checks"]
    assert json.loads((tmp_path / "attachment-path-audit.json").read_text())["passed"] is False


def test_attachment_path_audit_skips_invalid_utf8_trace_and_writes_failure(tmp_path):
    _write_fixture(tmp_path)
    (tmp_path / "traces/spans.jsonl").write_bytes(b"\xff")

    report = audit_attachment_path(tmp_path)

    assert report["passed"] is False
    assert "span:extract" in report["failed_checks"]
    assert json.loads((tmp_path / "attachment-path-audit.json").read_text())["passed"] is False


def test_attachment_path_audit_cli_returns_status_and_writes_artifact(tmp_path):
    _write_fixture(tmp_path)
    assert main(["--run-dir", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "attachment-path-audit.json").read_text())["passed"] is True

    missing = tmp_path / "missing"
    missing.mkdir()
    assert main(["--run-dir", str(missing)]) == 1
    assert json.loads((missing / "attachment-path-audit.json").read_text())["passed"] is False
