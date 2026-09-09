"""Exercise call discovery against isolated PostgreSQL, never production writes."""
import datetime as dt
import os

import psycopg
import pytest

from cds_history import fetch_conversation


def test_real_call_history_scopes_endpoints_and_hosts_and_pages_timestamp_ties():
    dsn = os.environ.get("DOC_HISTORY_TEST_DSN")
    if not dsn:
        pytest.skip("isolated DOC_HISTORY_TEST_DSN not configured")
    with psycopg.connect(dsn) as conn:
        assert conn.info.dbname == "doc_history_test", "refusing non-test database"
        conn.execute("""CREATE TEMP TABLE participants
            (id bigint, email text, phone text, phone_number text);
            CREATE TEMP TABLE calls (id bigint, source text, source_call_id text,
            started_at timestamptz, direction text, host_participant_id bigint,
            from_number text, to_number text);""")
        conn.execute("INSERT INTO participants VALUES (1,'person@example.test',NULL,NULL),(2,NULL,'+12025550199',NULL)")
        at = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=90)
        for row_id, host, caller, callee, when in [
            (1, 2, '+12025550123', '+12025550199', at),
            (2, 2, '+12025550199', '+12025550123', at),
            (3, 1, None, None, at),
            (4, 2, '+12025550199', '+12025550198', at),
            (5, 1, '+12025550123', None, at + dt.timedelta(days=100)),
        ]:
            conn.execute("INSERT INTO calls VALUES (%s,'twilio',%s,%s,NULL,%s,%s,%s)",
                         (row_id, 'AC' + str(row_id) * 32, when, host, caller, callee))
        contact = {'phone_e164': '+12025550123', 'email': 'person@example.test',
                   'history': {'kind': 'calls', 'limit': 1}}
        seen, token = [], None
        with conn.cursor() as cur:
            while True:
                page = fetch_conversation(cur, {**contact, 'history': {**contact['history'], 'cursor': token}})
                seen.extend(page['messages'])
                token = page['next_cursor']
                if not token:
                    break
            hostile = fetch_conversation(cur, {'email': "x' OR true --", 'history': {'kind': 'calls'}})
        assert [r['id'] for r in seen] == ['call:3', 'call:2', 'call:1']
        assert all(r['event_kind'] == 'call_reference' and r['body'] == '' for r in seen)
        assert hostile['messages'] == []
