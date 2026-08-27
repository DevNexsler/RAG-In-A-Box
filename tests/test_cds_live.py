import datetime as dt

import cds_live


class FakeCursor:
    def __init__(self, rows_by_marker):
        self.rows_by_marker, self.executed, self._rows = rows_by_marker, [], []

    def execute(self, sql, params=None):
        self.executed.append(sql)
        self._rows = next((r for marker, r in self.rows_by_marker.items()
                           if marker in sql), [])

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else (0, None)


def test_outbound_evidence_merges_lanes_newest_first():
    ts1 = dt.datetime(2026, 8, 27, 15, tzinfo=dt.timezone.utc)
    ts2 = dt.datetime(2026, 8, 27, 16, tzinfo=dt.timezone.utc)
    ts3 = dt.datetime(2026, 8, 27, 17, tzinfo=dt.timezone.utc)
    # Distinct markers per raw lane (rather than the shared "raw_events" table
    # name both RAW_QUO_SQL and RAW_EMAIL_SQL join against) so each lane gets
    # its own canned row and all three lanes are exercised in one call.
    cur = FakeCursor({"outbound_actions": [(ts1, "quo.sms.send", "ref1")],
                      "source = 'quo'": [(ts2, "ACx")],
                      "source = 'zoho_mail'": [(ts3, "<msg@pfg.io>")]})
    out = cds_live.fetch_outbound_evidence(cur, "a@b.com", "+15550000000", "7")
    assert out[0]["at"] > out[1]["at"] > out[2]["at"]
    by_lane = {o["lane"]: o for o in out}
    assert set(by_lane) == {"outbound_actions", "raw_quo", "raw_email"}
    for entry in out:
        assert set(entry) == {"lane", "at", "operation", "ref"}
    assert by_lane["outbound_actions"]["operation"] == "quo.sms.send"
    assert by_lane["raw_quo"]["operation"] == "quo.sms.send"
    assert by_lane["raw_email"]["operation"] == "email.send"


def test_inbound_summary_shape():
    ts = dt.datetime(2026, 8, 27, 14, tzinfo=dt.timezone.utc)
    cur = FakeCursor({"message_participants": [(3, ts)]})
    out = cds_live.fetch_inbound_summary(cur, "a@b.com", "+15550000000")
    assert out == {"inbound_count_30d": 3,
                   "latest_inbound_at": ts.isoformat()}


def test_cds_source_degrades_on_connection_error(monkeypatch):
    monkeypatch.setattr(cds_live, "_get_readonly_conn",
                        lambda: (_ for _ in ()).throw(RuntimeError("no dsn")))
    out = cds_live.cds_source({"email": "a@b.com", "phone_e164": None, "lead_id": None})
    assert out["status"].startswith("error:")


def test_cds_source_no_identifiers():
    out = cds_live.cds_source({"email": None, "phone_e164": None, "lead_id": None})
    assert out["status"] == "no_identifiers"
