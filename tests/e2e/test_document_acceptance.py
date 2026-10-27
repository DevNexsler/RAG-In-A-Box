"""Real container: identity receipt precedes provider processing and survives indexing."""
import hashlib
import uuid

import pytest

pytestmark = pytest.mark.anyio


async def test_acceptance_receipt_id_survives_targeted_indexing(api, mcp_session):
    filename = f'acceptance-{uuid.uuid4().hex}.md'
    content = b'# Identity first\n\nApproved bytes for acceptance contract.'
    headers = {'Idempotency-Key': filename}
    response = await api.post('/api/documents/accept', files={'file': (filename, content)}, headers=headers)
    assert response.status_code == 201, response.text
    receipt = response.json()
    assert receipt['doc_id'].startswith('documents::')
    assert receipt['index_status'] == 'pending'
    assert receipt['sha256'] == hashlib.sha256(content).hexdigest()
    download = await api.get('/api/documents/' + receipt['rel_path'])
    assert download.status_code == 200 and download.content == content
    replay = await api.post('/api/documents/accept', files={'file': (filename, content)}, headers=headers)
    assert replay.json() == receipt
    result = await mcp_session.call_tool_json('file_index_document', {'target': receipt['rel_path']})
    assert result.get('status') in {'indexed', 'skipped'}, result
    assert result['doc_id'] == receipt['doc_id']
    unchanged = await api.get('/api/documents/' + receipt['rel_path'])
    assert unchanged.status_code == 200 and unchanged.content == content
