"""Exact-event public tool backed by isolated PostgreSQL; no production writes."""
import hashlib
import os

import psycopg
import pytest

import cds_live
import mcp_server as srv


@pytest.fixture
def database(monkeypatch):
    dsn = os.environ.get('DOC_HISTORY_TEST_DSN')
    if not dsn:
        pytest.skip('isolated DOC_HISTORY_TEST_DSN not configured')
    with psycopg.connect(dsn) as conn:
        assert conn.info.dbname == 'doc_history_test'
        conn.execute('CREATE TEMP TABLE messages (id bigint,source text,source_message_id text,sent_at timestamptz,direction text,sender_name text,subject text,body text,body_text text,content text)')
        body = 'Résumé 🏠 ' * 2000 + 'FINAL correction: no payment confirmed.'
        conn.execute("INSERT INTO messages VALUES (1,'quo','long',now(),'inbound','Sender',NULL,%s,NULL,NULL)", (body,))
        conn.execute("INSERT INTO messages VALUES (2,'quo','duplicate',now(),NULL,NULL,NULL,'one',NULL,NULL),(3,'mail','duplicate',now(),NULL,NULL,NULL,'two',NULL,NULL)")
        conn.commit()
        monkeypatch.setattr(cds_live, '_get_readonly_conn', lambda: conn)
        yield conn, body


@pytest.mark.anyio
async def test_public_tool_preserves_unicode_full_body_and_explicit_gaps(database):
    _, body = database
    args = {'event_refs': ['long', 'duplicate', 'missing']}
    fragments, missing, ambiguous = [], [], []
    for _ in range(10):
        result = await srv.context_builder(**args)
        page = result['cds']['events']
        fragments.extend(m['body'] for m in page['messages'])
        for message in page['messages']:
            assert message['body_sha256'] == hashlib.sha256(body.encode()).hexdigest()
        missing.extend(page['missing_refs'])
        ambiguous.extend(page['ambiguous_refs'])
        if not page['has_more']:
            break
        args['event_cursor'] = page['next_cursor']
    assert ''.join(fragments) == body
    assert missing == ['missing']
    assert ambiguous == ['duplicate']


@pytest.mark.anyio
async def test_public_tool_rejects_event_mutation_between_body_pages(database):
    conn, _ = database
    first = await srv.context_builder(event_refs=['long'])
    conn.execute("UPDATE messages SET body=body || 'changed' WHERE id=1")
    conn.commit()
    second = await srv.context_builder(event_refs=['long'], event_cursor=first['cds']['events']['next_cursor'])
    assert 'changed during pagination' in second['error']
