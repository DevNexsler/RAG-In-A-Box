"""Real PostgreSQL query tests. Set DOC_HISTORY_TEST_DSN to isolated doc_history_test DB."""

import datetime as dt
import os

import psycopg
from psycopg.types.json import Jsonb
import pytest

import cds_live
from cds_history import fetch_conversation
import mcp_server as srv


@pytest.fixture
def database():
    dsn = os.environ.get("DOC_HISTORY_TEST_DSN")
    if not dsn:
        pytest.skip("isolated DOC_HISTORY_TEST_DSN not configured")
    with psycopg.connect(dsn) as conn:
        assert conn.info.dbname == "doc_history_test", "refusing non-test database"
        conn.execute("""
            CREATE TEMP TABLE participants (id bigint PRIMARY KEY, email text, phone text, phone_number text);
            CREATE TEMP TABLE raw_events (id bigint PRIMARY KEY, payload jsonb);
            CREATE TEMP TABLE messages (id bigint PRIMARY KEY, source text, source_message_id text,
                sent_at timestamptz, direction text, sender_name text, subject text, body text,
                body_text text, content text, raw_event_id bigint, channel_id bigint);
            CREATE TEMP TABLE message_participants (message_id bigint, participant_id bigint);
            CREATE TEMP TABLE outbound_actions (id bigint, created_at timestamptz, operation text,
                provider_message_id text, action_uid text, status text, channel text);
        """)
        now = dt.datetime.now(dt.timezone.utc)
        old = now - dt.timedelta(days=90)
        conn.execute("INSERT INTO participants VALUES (1,'person@example.test','+12025550123',NULL),(2,NULL,'+12025550199',NULL),(3,NULL,'+12025550123',NULL)")
        for rid, payload in [
            (1, {"data": {"object": {"id": "question", "to": "+12025550123"}}}),
            (2, {"data": {"object": {"id": "other", "to": "+12025550199"}}}),
            (3, {"participants": [{"kind": "to", "address": "PERSON@example.test"}], "data": {"object": {"id": "mail"}}}),
            (4, {"participants": {"malformed": True}}),
        ]:
            conn.execute("INSERT INTO raw_events VALUES (%s,%s)", (rid, Jsonb(payload)))
        rows = [
            (1, "quo", "question", old, "outbound", None, None, "Still interested in placing deposit?", None, None, 1, 99),
            (2, "quo", "withdrawal", old + dt.timedelta(minutes=1), "inbound", "Test Prospect", None, "No thank you.", None, None, None, 99),
            (3, "quo", "other", old + dt.timedelta(minutes=2), "outbound", None, None, "Unrelated prospect", None, None, 2, 99),
            (4, "zoho_mail", "mail", old + dt.timedelta(minutes=3), "outbound", None, "Deposit", "Acknowledged withdrawal", None, None, 3, 99),
            (5, "zoho_mail", "malformed", old, "outbound", None, None, "Not for contact", None, None, 4, 99),
            (6, "quo", "future", now + dt.timedelta(days=1), "inbound", None, None, "Future data", None, None, None, 99),
        ]
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO messages VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", rows)
        conn.execute("INSERT INTO message_participants VALUES (2,1),(2,3),(6,1)")
        # Preserve production's existing optional alias-suggestion query contract.
        conn.execute("ALTER TABLE participants ADD COLUMN display_name text; ALTER TABLE participants ADD COLUMN participant_key text")
        conn.commit()
        yield conn, old


def test_real_context_builder_recovers_quiet_exchange_and_latest_inbound(database, monkeypatch):
    conn, old = database
    monkeypatch.setattr(cds_live, "_get_readonly_conn", lambda: conn)
    result = srv._context_builder_impl(phone="2025550123", email="person@example.test", include=["cds"],
                                       history_since=(old - dt.timedelta(days=1)).isoformat())
    cds = result["cds"]
    assert cds["status"] == "ok"
    assert cds["inbound_count_30d"] == 0
    assert dt.datetime.fromisoformat(cds["latest_inbound_at"]) == old + dt.timedelta(minutes=1)
    history = cds["conversation"]
    assert [m["source_message_id"] for m in history["messages"]] == ["mail", "withdrawal", "question"]
    assert history["coverage_complete"]
    assert history["messages"][1]["body"] == "No thank you."
    assert history["messages"][-1]["sender_name"] is None  # Never invent staff author.


def test_real_keyset_pages_do_not_skip_timestamp_ties_or_repeat_ids(database):
    conn, old = database
    conn.execute("UPDATE messages SET sent_at=%s WHERE id=1", (old + dt.timedelta(minutes=1),))
    contact = {"phone_e164": "+12025550123", "history": {"limit": 1}}
    with conn.cursor() as cur:
        first = fetch_conversation(cur, contact)
        second = fetch_conversation(cur, {**contact, "history": {"limit": 1, "cursor": first["next_cursor"]}})
    assert first["messages"][0]["id"] == "2"
    assert second["messages"][0]["id"] == "1"
    assert second["window_exhausted"] and not second["coverage_complete"]
    assert first["through"] == second["through"]


def test_real_body_truncation_and_parameterized_identity(database):
    conn, _ = database
    conn.execute("UPDATE messages SET body=repeat('x',5000) WHERE id=2")
    with conn.cursor() as cur:
        history = fetch_conversation(cur, {"phone_e164": "+12025550123"})
        hostile = fetch_conversation(cur, {"email": "x' OR true --"})
    assert history["body_truncated_ids"] == ["2"]
    assert not history["coverage_complete"]
    assert len(history["messages"][0]["body"]) == 4000
    assert hostile["messages"] == []


def test_real_recent_count_deduplicates_participant_matches(database):
    conn, _ = database
    conn.execute("UPDATE messages SET sent_at=now()-interval '1 day' WHERE id=2")
    with conn.cursor() as cur:
        result = cds_live.fetch_inbound_summary(cur, "person@example.test", "+12025550123")
    assert result["inbound_count_30d"] == 1


def test_raw_cliq_sender_email_recovers_event_without_participant_link(database):
    conn, old = database
    for rid, source, kind, address in [
        (10, 'zoho_cliq', 'sender', 'PERSON@example.test'),
        (11, 'zoho_cliq', 'to', 'person@example.test'),
        (12, 'unknown_provider', 'sender', 'person@example.test'),
        (13, 'zoho_cliq', 'sender', 'other@example.test'),
    ]:
        conn.execute('INSERT INTO raw_events VALUES (%s,%s)',
                     (rid, Jsonb({'participants': [{'kind': kind, 'address': address}]})))
        conn.execute('INSERT INTO messages VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',
                     (rid, source, 'raw-' + str(rid), old + dt.timedelta(minutes=10),
                      'inbound', 'Stored sender name', None, 'Mention person@example.test in body.',
                      None, None, rid, 99))
    with conn.cursor() as cur:
        result = fetch_conversation(cur, {'email': 'person@example.test'})
    assert [m['source_message_id'] for m in result['messages']] == ['raw-10', 'mail', 'withdrawal']
    assert result['coverage_complete']
    assert result['messages'][0]['sender_name'] == 'Stored sender name'
