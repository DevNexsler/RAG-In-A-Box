"""Behavioural red/green for #1076 / PR #109.

`audit_sensitive_index.py --apply` deletes every flagged doc. A flagged doc whose
source row is gone does not come back from a reindex — it is simply erased from
the index, with no error and no operator decision. Written by the 2026-08-25
reconciliation because the PR's own red was a TypeError on an absent keyword,
which proves an API is missing, not that data is lost.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import audit_sensitive_index as asi


class _FakeStore:
    def __init__(self):
        self.deleted: list[str] = []

    def delete_by_doc_ids(self, doc_ids):
        self.deleted.extend(doc_ids)


_JWT = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIn0."
    "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
)


def _rows():
    return [
        {"doc_id": "documents::live1", "text": f"token {_JWT}", "metadata": {}},
        {"doc_id": "documents::gone1", "text": f"token {_JWT}", "metadata": {}},
    ]


def test_apply_does_not_silently_erase_a_source_missing_document():
    """A flagged doc with no source cannot be reindexed, so deleting it is a
    permanent recall loss and must not happen without an explicit decision."""
    audit = asi.audit_rows(_rows())
    assert set(audit.doc_ids) == {"documents::live1", "documents::gone1"}, audit.doc_ids

    store = _FakeStore()
    live = {"documents::live1"}  # 'gone1' has no source row left

    # Ask for the safe behaviour if the API can express it; otherwise run the
    # remediation exactly as `--apply` does today. Either way the assertion is
    # about the outcome, not the signature.
    try:
        asi.remediate_sensitive_docs(store, audit, live_doc_ids=live)
    except TypeError:
        asi.remediate_sensitive_docs(store, audit)
    except RuntimeError:
        assert store.deleted == [], (
            "aborted for an explicit orphan decision but still deleted rows"
        )
        return

    assert "documents::gone1" not in store.deleted, (
        "source-missing document was deleted with no operator decision — it will "
        "not come back from a reindex, so this is permanent recall loss"
    )
