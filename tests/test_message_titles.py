"""Message identity must survive the real indexing and persistence boundary."""
from unittest.mock import Mock

import pytest
from llama_index.core.node_parser import SentenceSplitter

import flow_index_vault as flow
from lancedb_store import LanceDBStore
from sources.base import SourceRecord
from sources.postgres import PostgresSource


MESSAGE_ID = '<CF.0D.28275.0E62DAA6@i-052407b4cdf7ba651.mta2vrest.sd.prd.sparkpost>'


@pytest.mark.parametrize('metadata, expected', [
    ({'subject': 'Weekend open houses', 'filename': 'mail.eml'}, 'Weekend open houses'),
    ({'subject': '  ', 'filename': 'mail.eml'}, 'mail.eml'),
    ({'subject': None, 'original_filename': 'original.eml'}, 'original.eml'),
    ({'subject': '', 'filename': MESSAGE_ID}, None),
    ({}, None),
])
def test_message_title_reaches_stored_row(tmp_path, monkeypatch, metadata, expected):
    doc_id = f'comm_messages::zoho_mail/{MESSAGE_ID}'
    record = SourceRecord(
        doc_id=f'zoho_mail/{MESSAGE_ID}', source_type='pg_message',
        natural_key=MESSAGE_ID, mtime=1.0, size=100,
        metadata={'_text': 'Open houses this weekend. Visit the new listings.',
                  'message_id': MESSAGE_ID, **metadata},
    )
    store = LanceDBStore(str(tmp_path / 'index'), 'chunks')
    embed = Mock()
    embed.embed_texts.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
    monkeypatch.setattr(flow, '_RUNTIME', {
        'store': store, 'embed_provider': embed,
        'splitter': SentenceSplitter(chunk_size=300, chunk_overlap=20),
        'config': {},
        'sources_by_name': {'comm_messages': PostgresSource('comm_messages', '', [])},
        'source_records_by_ns_doc_id': {doc_id: record},
    })
    flow.process_doc_task.fn({
        'doc_id': doc_id, 'source_name': 'comm_messages',
        'source_type': 'pg_message', 'mtime': 1.0, 'size': 100,
    })
    rows = store._vs.table.search(None).to_list()
    assert rows
    for row in rows:
        title = row['metadata']['title']
        if expected is not None:
            assert title == expected
        else:
            assert title.startswith('Message ')
            assert len(title) <= 32
        assert not title.startswith('<')
        assert row['metadata']['message_id'] == MESSAGE_ID
        assert row['text'].splitlines()[0] == f'[Document: {title}]'
        assert MESSAGE_ID not in row['text'].splitlines()[0]
    embedded = embed.embed_texts.call_args.args[0]
    assert all(text.startswith(f'[Document: {rows[0]["metadata"]["title"]}]') for text in embedded)
