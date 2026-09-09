"""Typed call evidence through public Context Builder, isolated PostgreSQL."""
import os

import psycopg
import pytest

import cds_live
import mcp_server as srv


@pytest.mark.anyio
@pytest.mark.parametrize('scenario', ['metadata', 'empty', 'transcript', 'conflict', 'collision', 'missing'])
async def test_existing_transcriptless_call_is_metadata_not_missing(monkeypatch, scenario):
    dsn = os.environ.get('DOC_HISTORY_TEST_DSN')
    if not dsn:
        pytest.skip('isolated DOC_HISTORY_TEST_DSN not configured')
    ref = 'AC' + 'a' * 32
    with psycopg.connect(dsn) as conn:
        assert conn.info.dbname == 'doc_history_test'
        conn.execute('CREATE TEMP TABLE messages (id bigint, source text, source_message_id text, sent_at timestamptz, direction text, sender_name text, subject text, body text, body_text text, content text)')
        conn.execute('CREATE TEMP TABLE calls (id bigint, source text, source_call_id text, started_at timestamptz, direction text, transcript text, duration_seconds integer, status text, from_number text, to_number text)')
        conn.execute('CREATE TEMP TABLE transcripts (call_id bigint, transcript_text text)')
        conn.execute("INSERT INTO calls VALUES (1828,'quo',%s,'2026-07-11T00:00:00Z',NULL,NULL,9,'completed','+12025550123','+12025550456')", (ref,))
        if scenario == 'empty':
            conn.execute("UPDATE calls SET transcript='' ")
            conn.execute("INSERT INTO transcripts VALUES (1828,'')")
        if scenario in {'transcript', 'conflict'}:
            conn.execute("INSERT INTO transcripts VALUES (1828,'Tenant reports portal login failed.')")
        if scenario == 'conflict':
            conn.execute("UPDATE calls SET transcript='Tenant reports portal login succeeded.'")
        if scenario == 'collision':
            conn.execute("INSERT INTO messages (id,source,source_message_id,body) VALUES (1,'quo',%s,'SMS, not call')", (ref,))
        if scenario == 'missing':
            conn.execute('DELETE FROM calls')
        conn.commit()
        monkeypatch.setattr(cds_live, '_get_readonly_conn', lambda: conn)
        result = await srv.context_builder(event_refs=[ref])
        page = result['cds']['events']
        if scenario in {'conflict', 'collision', 'missing'}:
            assert page['messages'] == []
            assert page['missing_refs'] == ([ref] if scenario == 'missing' else [])
            assert page['ambiguous_refs'] == ([] if scenario == 'missing' else [ref])
            return
        assert page['missing_refs'] == []
        event = page['messages'][0]
        if scenario == 'transcript':
            assert event['event_kind'] == 'call_transcript'
            assert event['transcript_status'] == 'available'
            assert event['body'] == 'Tenant reports portal login failed.'
            return
        assert event['event_kind'] == 'call_metadata'
        assert event['transcript_status'] == 'unavailable'
        assert event['body'] == ''
        assert event['duration_seconds'] == 9
        assert event['call_status'] == 'completed'
        assert 'not evidence of what was said' in event['body_authority']
