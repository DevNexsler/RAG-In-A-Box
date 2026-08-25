import json
import sqlite3
from types import SimpleNamespace

import pytest

import scripts.audit_sensitive_index as audit_script
from scripts.audit_sensitive_index import (
    _filesystem_live_doc_ids,
    _load_rows,
    audit_rows,
    remediate_sensitive_docs,
    resolve_live_doc_ids,
)


def test_audit_reports_counts_and_categories_without_secret_values():
    secret = "ghp_0123456789abcdefghijklmnopqrstuvwxyzAB"
    rows = [
        {
            "doc_id": "comm_messages::sensitive",
            "text": f"Migration token: {secret}",
            "metadata": {"sender": "System", "nested": {"authorization": secret}},
        },
        {
            "doc_id": "comm_messages::normal",
            "text": "Inspection moved to Thursday.",
            "metadata": {"sender": "Pat"},
        },
    ]

    audit = audit_rows(rows)
    serialized = json.dumps(audit.public_report(), sort_keys=True)

    assert audit.scanned_chunks == 2
    assert audit.flagged_chunks == 1
    assert audit.flagged_docs == 1
    assert audit.doc_ids == ("comm_messages::sensitive",)
    assert "github_token" in audit.finding_counts
    assert secret not in serialized
    assert "comm_messages::sensitive" not in serialized


def test_remediation_deletes_flagged_documents_for_safe_reindex():
    audit = audit_rows(
        [
            {
                "doc_id": "comm_messages::sensitive",
                "text": "api_key=synthetic-secret-value-0123456789",
                "metadata": {},
            }
        ]
    )

    class FakeStore:
        deleted = None

        def delete_by_doc_ids(self, doc_ids):
            self.deleted = doc_ids

    store = FakeStore()
    deleted = remediate_sensitive_docs(
        store, audit, live_doc_ids={"comm_messages::sensitive"}
    )

    assert deleted == 1
    assert store.deleted == ["comm_messages::sensitive"]


def test_remediation_refuses_to_delete_source_missing_documents_without_policy():
    audit = audit_rows(
        [
            {
                "doc_id": "comm_messages::source-missing",
                "text": "api_key=synthetic-secret-value-0123456789",
                "metadata": {},
            }
        ]
    )

    class FakeStore:
        deleted = None

        def delete_by_doc_ids(self, doc_ids):
            self.deleted = doc_ids

    store = FakeStore()
    with pytest.raises(RuntimeError, match="source-missing"):
        remediate_sensitive_docs(store, audit, live_doc_ids=set())

    assert store.deleted is None


def test_remediation_drops_source_missing_documents_only_with_explicit_policy():
    audit = audit_rows(
        [
            {
                "doc_id": "comm_messages::source-missing",
                "text": "api_key=synthetic-secret-value-0123456789",
                "metadata": {},
            }
        ]
    )

    class FakeStore:
        deleted = None

        def delete_by_doc_ids(self, doc_ids):
            self.deleted = doc_ids

    store = FakeStore()
    deleted = remediate_sensitive_docs(
        store, audit, live_doc_ids=set(), orphan_policy="drop"
    )

    assert deleted == 1
    assert store.deleted == ["comm_messages::source-missing"]


def test_filesystem_source_presence_requires_registry_row_and_live_file(tmp_path):
    root = tmp_path / "documents"
    root.mkdir()
    (root / "live.md").write_text("safe")
    (root / "99-Archive").mkdir()
    (root / "99-Archive" / "old.md").write_text("safe")
    registry = tmp_path / "doc_registry.db"
    connection = sqlite3.connect(registry)
    connection.execute(
        "CREATE TABLE doc_registry (doc_id TEXT, rel_path TEXT, source_name TEXT)"
    )
    connection.executemany(
        "INSERT INTO doc_registry VALUES (?, ?, ?)",
        [
            ("live", "live.md", "documents"),
            ("gone", "gone.md", "documents"),
            ("excluded", "99-Archive/old.md", "documents"),
        ],
    )
    connection.commit()
    connection.close()

    live = _filesystem_live_doc_ids(
        {
            "name": "documents",
            "root": str(root),
            "scan": {
                "include": ["**/*.md"],
                "exclude": ["**/99-Archive/**"],
            },
        },
        {
            "documents::live",
            "documents::gone",
            "documents::excluded",
        },
        tmp_path,
    )

    assert live == {"documents::live"}


def test_resolve_live_doc_ids_streams_configured_postgres_source(monkeypatch, tmp_path):
    closed = False

    class FakeSource:
        def scan(self):
            yield SimpleNamespace(doc_id="live")
            yield SimpleNamespace(doc_id="unrelated")

        def close(self):
            nonlocal closed
            closed = True

    monkeypatch.setattr("sources.build_source", lambda _config: FakeSource())

    live = resolve_live_doc_ids(
        {"sources": [{"name": "comm_messages", "type": "postgres"}]},
        {"comm_messages::live", "comm_messages::gone"},
        tmp_path,
    )

    assert live == {"comm_messages::live"}
    assert closed is True


def test_load_rows_uses_bounded_projected_batches(monkeypatch, tmp_path):
    requested = None

    class FakeBatch:
        def __init__(self, rows):
            self.rows = rows

        def to_pylist(self):
            return self.rows

    class FakeScanner:
        def to_batches(self):
            yield FakeBatch([{"doc_id": "one"}])
            yield FakeBatch([{"doc_id": "two"}])

    class FakeDataset:
        def scanner(self, **kwargs):
            nonlocal requested
            requested = kwargs
            return FakeScanner()

    class FakeTable:
        def to_lance(self):
            return FakeDataset()

    class FakeConnection:
        def open_table(self, name):
            assert name == "chunks"
            return FakeTable()

    monkeypatch.setattr("lancedb.connect", lambda _root: FakeConnection())

    assert list(_load_rows(tmp_path, "chunks", batch_size=2)) == [
        {"doc_id": "one"},
        {"doc_id": "two"},
    ]
    assert requested == {
        "columns": ["doc_id", "text", "metadata"],
        "batch_size": 2,
    }


def test_plan_mode_classifies_sources_without_opening_store(monkeypatch, capsys):
    rows = [
        {
            "doc_id": "comm_messages::live",
            "text": "api_key=synthetic-secret-value-0123456789",
            "metadata": {},
        }
    ]
    monkeypatch.setattr(audit_script, "_load_rows", lambda *_args, **_kwargs: rows)
    monkeypatch.setattr(
        audit_script,
        "resolve_live_doc_ids",
        lambda _config, _doc_ids, _index_root: {"comm_messages::live"},
    )
    monkeypatch.setattr("core.config.load_config", lambda _path: {"sources": []})

    assert audit_script.main(["--plan"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "plan"
    assert report["source_backed_docs"] == 1
    assert report["source_missing_docs"] == 0


def test_apply_mode_aborts_before_opening_store_when_source_is_missing(
    monkeypatch, capsys
):
    rows = [
        {
            "doc_id": "comm_messages::gone",
            "text": "api_key=synthetic-secret-value-0123456789",
            "metadata": {},
        }
    ]
    monkeypatch.setattr(audit_script, "_load_rows", lambda *_args, **_kwargs: rows)
    monkeypatch.setattr(
        audit_script,
        "resolve_live_doc_ids",
        lambda _config, _doc_ids, _index_root: set(),
    )
    monkeypatch.setattr("core.config.load_config", lambda _path: {"sources": []})
    monkeypatch.setattr(
        "lancedb_store.LanceDBStore",
        lambda *_args, **_kwargs: pytest.fail("store must remain unopened"),
    )

    assert audit_script.main(["--apply"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "aborted"
    assert report["remediated_docs"] == 0
    assert report["source_missing_docs"] == 1


def test_apply_mode_deletes_mixed_docs_after_explicit_drop_policy(
    monkeypatch, capsys
):
    rows = [
        {
            "doc_id": doc_id,
            "text": "api_key=synthetic-secret-value-0123456789",
            "metadata": {},
        }
        for doc_id in ("comm_messages::live", "comm_messages::gone")
    ]

    class FakeStore:
        deleted = None

        def delete_by_doc_ids(self, doc_ids):
            self.deleted = doc_ids

    store = FakeStore()
    monkeypatch.setattr(audit_script, "_load_rows", lambda *_args, **_kwargs: rows)
    monkeypatch.setattr(
        audit_script,
        "resolve_live_doc_ids",
        lambda _config, _doc_ids, _index_root: {"comm_messages::live"},
    )
    monkeypatch.setattr("core.config.load_config", lambda _path: {"sources": []})
    monkeypatch.setattr("lancedb_store.LanceDBStore", lambda *_args: store)

    assert audit_script.main(["--apply", "--orphan-policy", "drop"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["remediated_docs"] == 2
    assert report["source_backed_docs"] == 1
    assert report["source_missing_docs"] == 1
    assert store.deleted == ["comm_messages::gone", "comm_messages::live"]
