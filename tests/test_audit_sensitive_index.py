import json

from scripts.audit_sensitive_index import audit_rows, remediate_sensitive_docs


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
    deleted = remediate_sensitive_docs(store, audit)

    assert deleted == 1
    assert store.deleted == ["comm_messages::sensitive"]
