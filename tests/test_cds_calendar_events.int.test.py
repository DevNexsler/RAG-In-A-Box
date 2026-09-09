"""Typed calendar evidence preserves complete owner copies and source versions."""
import json
import os

import psycopg
import pytest

from cds_exact_events import event_request, fetch_event_page


def test_calendar_reference_preserves_all_fields_copies_and_paged_versions():
    dsn = os.environ.get('DOC_HISTORY_TEST_DSN')
    if not dsn:
        pytest.skip('isolated DOC_HISTORY_TEST_DSN not configured')
    with psycopg.connect(dsn) as conn:
        assert conn.info.dbname == 'doc_history_test', 'refusing non-test database'
        conn.execute('''CREATE TEMP TABLE messages (id bigint, source text,
            source_message_id text, sent_at timestamptz, direction text,
            sender_name text, subject text, body text, body_text text, content text);
            CREATE TEMP TABLE calendar_events (id bigint, source text, uid text,
            source_event_id text, calendar_owner text, description text, status text,
            last_modified timestamptz, updated_at timestamptz, created_at timestamptz,
            starts_at timestamptz, future_field text);''')
        for rid, owner, status in [(1, 'owner-a', 'confirmed'), (2, 'owner-b', 'cancelled')]:
            conn.execute('''INSERT INTO calendar_events VALUES
                (%s,'zoho_calendar','meeting@example.test','meeting@example.test',%s,%s,%s,
                 '2026-01-02T00:00:00Z',NULL,NULL,'2026-01-03T00:00:00Z','keep unknown fields')''',
                         (rid, owner, 'Full description ' + 'x' * 13000, status))
        refs = ['calendar:meeting@example.test']
        state = event_request(refs)
        chunks, first_cursor = [], None
        with conn.cursor() as cur:
            while True:
                page = fetch_event_page(cur, refs, state)
                assert page['status'] == 'ok'
                event = page['messages'][0]
                assert event['event_kind'] == 'calendar_record_set'
                assert 'not proof of attendance' in event['body_authority']
                assert event['body_offset'] == sum(len(c) for c in chunks)
                chunks.append(event['body'])
                if not page['next_cursor']:
                    break
                first_cursor = first_cursor or page['next_cursor']
                state = event_request(refs, page['next_cursor'])
            copies = json.loads(''.join(chunks))
            assert [r['calendar_owner'] for r in copies] == ['owner-a', 'owner-b']
            assert [r['status'] for r in copies] == ['confirmed', 'cancelled']
            assert all(r['future_field'] == 'keep unknown fields' for r in copies)
            assert all(r['description'].endswith('x' * 13000) for r in copies)
            conn.execute("UPDATE calendar_events SET description='changed' WHERE id=1")
            with pytest.raises(ValueError, match='changed during pagination'):
                fetch_event_page(cur, refs, event_request(refs, first_cursor))
