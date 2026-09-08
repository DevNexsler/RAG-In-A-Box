"""Quiet contacts must not lose later intent changes to a recent-only window."""

from datetime import datetime, timezone

import cds_live


def test_dossier_includes_old_inbound_withdrawal_and_outbound_question(monkeypatch):
    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, sql, params=None):
            self.rows = []
            if "contact_history" in sql:
                self.rows = [
                    (2, "quo", "withdrawal", datetime(2026, 6, 25, 15, 16, tzinfo=timezone.utc), "inbound", "Test Prospect", None, "No thank you.", False),
                    (1, "quo", "question", datetime(2026, 6, 25, 15, 15, tzinfo=timezone.utc), "outbound", None, None, "Still interested in placing the deposit?", False),
                ]

        def fetchone(self):
            return (0, None)

        def fetchall(self):
            return self.rows

    class Connection:
        def cursor(self):
            return Cursor()

        def rollback(self):
            pass

    monkeypatch.setattr(cds_live, "_get_readonly_conn", Connection)
    result = cds_live.cds_source({"phone_e164": "+12025550123"})
    messages = result.get("conversation", {}).get("messages", [])
    assert {m["source_message_id"] for m in messages} == {"question", "withdrawal"}
    assert any(m["body"] == "No thank you." for m in messages)
