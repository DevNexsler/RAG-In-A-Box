"""Identity-first receipts through REST, independent of the indexing pipeline."""
import concurrent.futures
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from api_server import build_api_app
from core.index_request_queue import pending_backlog
from doc_id_store import DocIDStore, extract_id_from_filename
from document_acceptance import DocumentAcceptance
from flow_index_vault import scan_filesystem_records


@pytest.fixture
def roots(tmp_path):
    docs, index = tmp_path / 'documents', tmp_path / 'index'
    docs.mkdir()
    index.mkdir()
    registry = DocIDStore(index / 'doc_registry.db')
    registry.close()
    return docs, index


@pytest.mark.anyio
async def test_receipt_has_persistent_identity_and_queued_index_work(roots):
    docs, index = roots
    app = build_api_app(docs, index_root=index, source_name='vault', table_name='test-chunks')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post('/documents/accept', files={'file': ('notice.pdf', b'%PDF approved')},
                                     data={'directory': 'case'}, headers={'Idempotency-Key': 'one'})
    assert response.status_code == 201
    receipt = response.json()
    assert receipt['doc_id'].startswith('vault::')
    assert receipt['sha256'] == hashlib.sha256(b'%PDF approved').hexdigest()
    assert receipt['index_status'] == 'pending'
    assert receipt['filename'] == 'notice.pdf'
    assert (docs / receipt['rel_path']).read_bytes() == b'%PDF approved'
    assert not (docs / 'case/notice.pdf').exists()
    registry = DocIDStore(index / 'doc_registry.db')
    try:
        assert registry.all_mappings()[receipt['doc_id']] == receipt['rel_path']
        assert extract_id_from_filename(receipt['rel_path']) == receipt['doc_id'].split('::')[1]
        assert registry.list_rows()[0]['source_name'] == 'vault'
    finally:
        registry.close()
    assert pending_backlog(index, 'test-chunks')['pending'] == 1


@pytest.mark.anyio
async def test_idempotent_retry_and_changed_input_conflict(roots):
    docs, index = roots
    app = build_api_app(docs, index_root=index)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        a = await client.post('/documents/accept', files={'file': ('report.md', b'first')},
                              headers={'Idempotency-Key': 'retry'})
        b = await client.post('/documents/accept', files={'file': ('report.md', b'first')},
                              headers={'Idempotency-Key': 'retry'})
        conflict = await client.post('/documents/accept', files={'file': ('report.md', b'changed')},
                                     headers={'Idempotency-Key': 'retry'})
    assert a.status_code == b.status_code == 201
    assert a.json() == b.json()
    assert conflict.status_code == 409
    assert len(list(docs.iterdir())) == 1
    assert pending_backlog(index, 'chunks')['pending'] == 1


def test_process_death_after_publish_recovers_before_scan(roots):
    docs, index = roots
    code = '''import os, sys
from pathlib import Path
from doc_id_store import DocIDStore
from document_acceptance import DocumentAcceptance
root, index = map(Path,sys.argv[1:])
registry=DocIDStore(index/'doc_registry.db')
registry.register=lambda *a, **kw: os._exit(73)
DocumentAcceptance(root,registry).accept(root,'interrupted.md',b'approved',idempotency_key='crash')
'''
    result = subprocess.run([sys.executable, '-c', code, str(docs), str(index)], check=False)
    assert result.returncode == 73
    pending = next((index / 'document-acceptance/pending').glob('*.json'))
    receipt = json.loads(pending.read_text())['receipt']
    registry = DocIDStore(index / 'doc_registry.db')
    try:
        records = scan_filesystem_records(docs, ['**/*.md'], [], doc_id_store=registry)
        assert len(records) == 1
        assert records[0]['doc_id'] == receipt['doc_id'].split('::')[1]
        assert records[0]['rel_path'] == receipt['rel_path']
        retry = DocumentAcceptance(docs, registry).accept(docs, 'interrupted.md', b'approved', idempotency_key='crash')
        assert retry == receipt
    finally:
        registry.close()
    assert pending_backlog(index, 'chunks')['pending'] == 1
    assert not list(docs.glob('.organizer-accept-*'))


def test_concurrent_retries_share_one_identity_and_work_request(roots):
    docs, index = roots
    def accept(_):
        registry = DocIDStore(index / 'doc_registry.db')
        try:
            return DocumentAcceptance(docs, registry).accept(docs, 'same.md', b'same bytes', idempotency_key='concurrent')
        finally:
            registry.close()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        receipts = list(executor.map(accept, range(12)))
    assert all(receipt == receipts[0] for receipt in receipts)
    assert len(list(docs.iterdir())) == 1
    assert pending_backlog(index, 'chunks')['pending'] == 1


def test_existing_paths_and_client_supplied_ids_are_preserved(roots):
    docs, index = roots
    existing = docs / 'notice.md'
    existing.write_bytes(b'existing approved')
    registry = DocIDStore(index / 'doc_registry.db')
    try:
        receipt = DocumentAcceptance(docs, registry).accept(docs, 'notice@zzzzz@.md', b'new approved')
        assert existing.read_bytes() == b'existing approved'
        assert receipt['doc_id'] != 'documents::zzzzz'
        assert receipt['filename'] == 'notice.md'
        moved = docs / 'moved'
        moved.mkdir()
        (docs / receipt['rel_path']).rename(moved / Path(receipt['rel_path']).name)
        records = scan_filesystem_records(docs, ['**/*.md'], [], doc_id_store=registry)
        assert any(row['doc_id'] == receipt['doc_id'].split('::')[1] and row['rel_path'].startswith('moved/') for row in records)
    finally:
        registry.close()


@pytest.mark.anyio
async def test_acceptance_keeps_upload_validation(roots):
    docs, index = roots
    app = build_api_app(docs, index_root=index)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        for name, content, data in [('unsafe.exe', b'bad', {}), ('empty.md', b'', {}),
                                    ('safe.md', b'ok', {'directory': '../../escape'})]:
            response = await client.post('/documents/accept', files={'file': (name, content)}, data=data)
            assert response.status_code == 400
    assert not list(docs.iterdir())


@pytest.mark.parametrize('phase', ['publish', 'enqueue'])
def test_interrupted_acceptance_retry_retains_allocated_identity(roots, monkeypatch, phase):
    import document_acceptance
    docs, index = roots
    registry = DocIDStore(index / 'doc_registry.db')
    acceptance = DocumentAcceptance(docs, registry)
    try:
        with monkeypatch.context() as patch:
            def fail(*args, **kwargs):
                raise OSError('simulated interruption')
            if phase == 'publish':
                patch.setattr(document_acceptance.os, 'link', fail)
            else:
                patch.setattr(document_acceptance.IndexRequestQueue, 'enqueue', fail)
            with pytest.raises(OSError):
                acceptance.accept(docs, 'retry.md', b'approved', idempotency_key='same')
        pending = next((index / 'document-acceptance/pending').glob('*.json'))
        original = json.loads(pending.read_text())['receipt']
        receipt = acceptance.accept(docs, 'retry.md', b'approved', idempotency_key='same')
        assert receipt == original
        assert len(list(docs.iterdir())) == 1
        assert pending_backlog(index, 'chunks')['pending'] == 1
    finally:
        registry.close()


def test_scan_cannot_adjudicate_between_publication_and_registration(roots, monkeypatch):
    import threading
    docs, index = roots
    writer = DocIDStore(index / 'doc_registry.db')
    reader = DocIDStore(index / 'doc_registry.db')
    published = threading.Event()
    release = threading.Event()
    original = writer.register
    def pause(*args, **kwargs):
        published.set()
        assert release.wait(10)
        return original(*args, **kwargs)
    monkeypatch.setattr(writer, 'register', pause)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            accepted = executor.submit(DocumentAcceptance(docs, writer).accept, docs, 'race.md', b'approved')
            assert published.wait(10)
            scanned = executor.submit(scan_filesystem_records, docs, ['**/*'], [], reader)
            release.set()
            receipt = accepted.result(timeout=10)
            records = scanned.result(timeout=10)
        assert len(records) == 1
        assert records[0]['doc_id'] == receipt['doc_id'].split('::')[1]
        assert (docs / receipt['rel_path']).read_bytes() == b'approved'
    finally:
        release.set()
        writer.close()
        reader.close()


def test_unjournaled_staging_is_never_indexed_and_is_cleaned_on_scan(roots):
    docs, index = roots
    stage = docs / '.organizer-accept-orphan'
    stage.write_bytes(b'not accepted')
    registry = DocIDStore(index / 'doc_registry.db')
    try:
        assert scan_filesystem_records(docs, ['**/*'], [], doc_id_store=registry) == []
        assert registry.count() == 0
        assert not stage.exists()
    finally:
        registry.close()


def test_long_filename_refused_before_creating_unrecoverable_journal(roots):
    from document_acceptance import AcceptanceConflict
    docs, index = roots
    registry = DocIDStore(index / 'doc_registry.db')
    try:
        with pytest.raises(AcceptanceConflict, match='filesystem limit'):
            DocumentAcceptance(docs, registry).accept(docs, 'x' * 250 + '.md', b'approved')
        assert not list(docs.iterdir())
        assert not list((index / 'document-acceptance/pending').glob('*.json'))
    finally:
        registry.close()


def test_existing_filename_id_requires_registry_recovery_before_acceptance(roots):
    from document_acceptance import AcceptanceConflict
    docs, index = roots
    legacy = docs / 'legacy@00001@.md'
    legacy.write_bytes(b'existing identity')
    registry = DocIDStore(index / 'doc_registry.db')
    try:
        with pytest.raises(AcceptanceConflict, match='identity scan'):
            DocumentAcceptance(docs, registry).accept(docs, 'new.md', b'new identity')
        assert registry.count() == 0
        scan_filesystem_records(docs, ['**/*.md'], [], doc_id_store=registry)
        receipt = DocumentAcceptance(docs, registry).accept(docs, 'new.md', b'new identity')
        assert receipt['doc_id'] != 'documents::00001'
        assert registry.lookup_path('00001') == legacy.name
        assert legacy.read_bytes() == b'existing identity'
    finally:
        registry.close()


def test_empty_registry_checks_other_filesystem_source_ids(roots, tmp_path):
    from document_acceptance import AcceptanceConflict
    docs, index = roots
    other = tmp_path / 'other-source'
    other.mkdir()
    (other / 'legacy@00001@.md').write_bytes(b'existing identity')
    registry = DocIDStore(index / 'doc_registry.db')
    try:
        acceptance = DocumentAcceptance(docs, registry, identity_roots=[docs, other])
        with pytest.raises(AcceptanceConflict, match='identity scan'):
            acceptance.accept(docs, 'new.md', b'new identity')
        assert registry.count() == 0 and not list(docs.iterdir())
    finally:
        registry.close()


def test_retry_recovers_owned_publication_before_first_registry_commit(roots, monkeypatch):
    docs, index = roots
    registry = DocIDStore(index / 'doc_registry.db')
    acceptance = DocumentAcceptance(docs, registry)
    try:
        with monkeypatch.context() as patch:
            def fail(*args, **kwargs):
                raise OSError('interrupted registry commit')
            patch.setattr(registry, 'register', fail)
            with pytest.raises(OSError):
                acceptance.accept(docs, 'retry.md', b'approved', idempotency_key='same')
        pending = next((index / 'document-acceptance/pending').glob('*.json'))
        expected = json.loads(pending.read_text())['receipt']
        assert registry.count() == 0
        receipt = acceptance.accept(docs, 'retry.md', b'approved', idempotency_key='same')
        assert receipt == expected
        assert registry.count() == 1
        assert len(list(docs.iterdir())) == 1
    finally:
        registry.close()


SIDECAR = {
    'schema_version': 1,
    'message': {'source': 'tenantcloud_api', 'message_id': 'm-51', 'body': 'Signed lease attached.'},
    'media': {'media_index': 0, 'media_type': 'application/pdf', 'original_filename': 'lease.pdf'},
}
ORIGIN = {'origin_system': 'comm-data-store', 'origin_ref': 'message_media:7001'}


async def _accept(app, *, data=None, key='origin', name='lease.pdf', content=b'%PDF lease'):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        return await client.post('/documents/accept', files={'file': (name, content)},
                                 data=data or {}, headers={'Idempotency-Key': key})


@pytest.mark.anyio
async def test_origin_and_sidecar_are_accepted_with_the_file(roots):
    """One call deposits the bytes, their attachment metadata and who they belong to."""
    docs, index = roots
    app = build_api_app(docs, index_root=index)
    data = {'directory': 'tenantcloud-attachments/unit-4b', **ORIGIN, 'sidecar': json.dumps(SIDECAR)}
    response = await _accept(app, data=data)
    assert response.status_code == 201, response.text
    receipt = response.json()
    assert receipt['origin'] == {'system': 'comm-data-store', 'ref': 'message_media:7001'}
    target = docs / receipt['rel_path']
    # The sidecar sits beside the file under the name it was deposited with:
    # only the document carries the identity tag.
    assert receipt['sidecar_rel_path'] == 'tenantcloud-attachments/unit-4b/lease.json'
    assert json.loads((docs / receipt['sidecar_rel_path']).read_text()) == SIDECAR
    registry = DocIDStore(index / 'doc_registry.db')
    try:
        assert registry.origin(receipt['doc_id']) == receipt['origin']
        # The scanner reads the sidecar as the file's context, never as a document.
        from sources.filesystem import _find_communication_sidecar
        assert _find_communication_sidecar(target) == docs / receipt['sidecar_rel_path']
        records = scan_filesystem_records(docs, ['**/*.pdf', '**/*.json'], [], doc_id_store=registry)
        assert [record['rel_path'] for record in records] == [receipt['rel_path']]
        assert (docs / receipt['sidecar_rel_path']).exists()
    finally:
        registry.close()
    # A replay answers the same receipt and leaves one file, one sidecar, one request.
    assert (await _accept(app, data=data)).json() == receipt
    assert sorted(path.name for path in target.parent.iterdir()) == sorted([target.name, 'lease.json'])
    assert pending_backlog(index, 'chunks')['pending'] == 1


@pytest.mark.anyio
@pytest.mark.parametrize('data', [
    {'origin_system': 'comm-data-store'},
    {'origin_ref': 'message_media:1'},
    {'origin_system': 'Comm Data Store', 'origin_ref': 'message_media:1'},
    {'origin_system': 'comm-data-store', 'origin_ref': ' '},
    {'origin_system': 'comm-data-store', 'origin_ref': 'x' * 201},
    {'sidecar': '{not json'},
    {'sidecar': json.dumps(['not', 'an', 'object'])},
    # Any other JSON beside a document would be indexed as a document.
    {'sidecar': json.dumps({'invoice': 12, 'total': '1450.00'})},
])
async def test_a_malformed_origin_or_sidecar_deposits_nothing(roots, data):
    docs, index = roots
    response = await _accept(build_api_app(docs, index_root=index), data=data)
    assert response.status_code == 400, response.text
    assert list(docs.iterdir()) == []
    assert pending_backlog(index, 'chunks')['pending'] == 0


@pytest.mark.anyio
async def test_acceptance_without_an_origin_is_unchanged(roots):
    docs, index = roots
    receipt = (await _accept(build_api_app(docs, index_root=index))).json()
    assert set(receipt) == {'accepted', 'doc_id', 'rel_path', 'filename', 'sha256', 'size', 'index_status'}
    registry = DocIDStore(index / 'doc_registry.db')
    try:
        assert registry.origin(receipt['doc_id']) is None
    finally:
        registry.close()


def test_the_indexed_event_names_the_depositor(roots):
    from flow_index_vault import _with_document_origin
    docs, index = roots
    registry = DocIDStore(index / 'doc_registry.db')
    try:
        owned = DocumentAcceptance(docs, registry).accept(
            docs, 'lease.pdf', b'%PDF lease', origin={'system': 'comm-data-store', 'ref': 'message_media:7001'})
        plain = DocumentAcceptance(docs, registry).accept(docs, 'memo.pdf', b'%PDF memo')
    finally:
        registry.close()
    config = {'index_root': str(index)}
    event = {'event': 'document.indexed', 'doc_id': owned['doc_id']}
    assert _with_document_origin(config, event) == {**event, 'origin': owned['origin']}
    event = {'event': 'document.indexed', 'doc_id': plain['doc_id']}
    assert _with_document_origin(config, event) == event
    # A registry from before origins existed has no such table: no origin, no error.
    import sqlite3
    with sqlite3.connect(index / 'doc_registry.db') as connection:
        connection.execute('DROP TABLE document_origins')
    assert _with_document_origin(config, {'event': 'document.indexed', 'doc_id': owned['doc_id']}) == {
        'event': 'document.indexed', 'doc_id': owned['doc_id']}


@pytest.mark.anyio
async def test_a_key_covers_the_origin_but_not_the_metadata(roots):
    """A depositor's retry rebuilds its metadata (a fetch time, an edited body);
    it must get its receipt back, not a refusal it can never get past."""
    docs, index = roots
    app = build_api_app(docs, index_root=index)
    data = {**ORIGIN, 'sidecar': json.dumps(SIDECAR)}
    first = await _accept(app, data=data)
    assert first.status_code == 201
    later = {**SIDECAR, 'media': {**SIDECAR['media'], 'fetched_at': '2026-11-25T01:00:00+00:00'}}
    retry = await _accept(app, data={**ORIGIN, 'sidecar': json.dumps(later)})
    assert (retry.status_code, retry.json()) == (201, first.json())
    assert json.loads((docs / first.json()['sidecar_rel_path']).read_text()) == SIDECAR
    assert len(list(docs.glob('*.pdf'))) == 1
    other_owner = await _accept(app, data={**data, 'origin_ref': 'message_media:7002'})
    assert (other_owner.status_code, other_owner.json()['code']) == (409, 'acceptance_conflict')


@pytest.mark.anyio
async def test_a_sidecar_needs_a_name_no_other_document_in_the_folder_has(roots):
    """A sidecar is found by name, so it would also describe a namesake."""
    docs, index = roots
    app = build_api_app(docs, index_root=index)
    (docs / 'lease.docx').write_bytes(b'someone else\'s lease')
    refused = await _accept(app, data={'sidecar': json.dumps(SIDECAR)})
    assert (refused.status_code, refused.json()['code']) == (409, 'acceptance_conflict')
    assert sorted(path.name for path in docs.iterdir()) == ['lease.docx']
    assert pending_backlog(index, 'chunks')['pending'] == 0
    registry = DocIDStore(index / 'doc_registry.db')
    try:
        assert registry.count() == 0
    finally:
        registry.close()
    # Without a sidecar the namesake is no obstacle, and a second attachment of
    # the same name with its own metadata is refused the same way.
    assert (await _accept(app, key='plain')).status_code == 201
    assert (await _accept(app, key='two', name='other.pdf', data={'sidecar': json.dumps(SIDECAR)})).status_code == 201
    again = await _accept(app, key='three', name='other.pdf', content=b'%PDF b', data={'sidecar': json.dumps(SIDECAR)})
    assert again.status_code == 409


def _crash_at(monkeypatch, step):
    """Make publication die just before ``step`` (a name ``_finish`` calls)."""
    import document_acceptance as module

    def crash(*args, **kwargs):
        raise OSError('crash')
    if step == 'sidecar':
        real = module._write_journal
        monkeypatch.setattr(module, '_write_journal', lambda path, record, **kw: (
            crash() if path.suffix == '.json' and 'document-acceptance' not in path.parts
            else real(path, record, **kw)))
    elif step == 'register':
        monkeypatch.setattr(DocIDStore, 'register', crash)
    elif step == 'receipt':
        real_replace = module.os.replace
        monkeypatch.setattr(module.os, 'replace', lambda src, dst: (
            crash() if 'receipts' in Path(dst).parts else real_replace(src, dst)))


@pytest.mark.parametrize('step', ['sidecar', 'register', 'receipt'])
def test_an_acceptance_interrupted_anywhere_recovers_whole(roots, monkeypatch, step):
    docs, index = roots
    registry = DocIDStore(index / 'doc_registry.db')
    origin = {'system': 'comm-data-store', 'ref': 'message_media:7001'}
    try:
        acceptance = DocumentAcceptance(docs, registry)
        with monkeypatch.context() as patch:
            _crash_at(patch, step)
            with pytest.raises(OSError):
                acceptance.accept(docs, 'lease.pdf', b'%PDF lease', idempotency_key='k', origin=origin, sidecar=SIDECAR)
        if step == 'receipt':
            # Indexing was queued before the crash and has since added to the sidecar.
            sidecar = docs / 'lease.json'
            sidecar.write_text(json.dumps({**SIDECAR, 'context': {'thread': 'added by the indexer'}}))
        acceptance.recover()  # what every identity scan runs first
        receipt = acceptance.accept(docs, 'lease.pdf', b'%PDF lease', idempotency_key='k', origin=origin,
                                    sidecar={**SIDECAR, 'schema_version': 1, 'retry': True})
        assert registry.origin(receipt['doc_id']) == origin
        stored = json.loads((docs / receipt['sidecar_rel_path']).read_text())
        assert stored['message'] == SIDECAR['message'] and 'retry' not in stored
        if step == 'receipt':
            assert stored['context'] == {'thread': 'added by the indexer'}
        assert sorted(path.name for path in docs.iterdir()) == sorted([Path(receipt['rel_path']).name, 'lease.json'])
        assert pending_backlog(index, 'chunks')['pending'] == 1
        assert registry.count() == 1
    finally:
        registry.close()


def test_what_is_published_is_readable_by_the_other_readers_of_the_tree(roots):
    docs, index = roots
    registry = DocIDStore(index / 'doc_registry.db')
    try:
        receipt = DocumentAcceptance(docs, registry).accept(docs, 'lease.pdf', b'%PDF', sidecar=SIDECAR)
    finally:
        registry.close()
    assert (docs / receipt['rel_path']).stat().st_mode & 0o777 == 0o644
    assert (docs / receipt['sidecar_rel_path']).stat().st_mode & 0o777 == 0o644


def test_a_receipt_journalled_before_origins_existed_still_replays(roots):
    docs, index = roots
    registry = DocIDStore(index / 'doc_registry.db')
    try:
        acceptance = DocumentAcceptance(docs, registry)
        receipt = acceptance.accept(docs, 'memo.pdf', b'%PDF memo', idempotency_key='old')
        (journalled,) = (index / 'document-acceptance' / 'receipts').glob('*.json')
        record = json.loads(journalled.read_text())
        assert len(record['fingerprint']) == 5  # the shape every earlier release wrote
        for key in ('origin', 'sidecar'):
            record.pop(key)
        journalled.write_text(json.dumps(record))
        assert acceptance.accept(docs, 'memo.pdf', b'%PDF memo', idempotency_key='old') == receipt
    finally:
        registry.close()


def test_an_unreadable_registry_loses_no_event(roots, monkeypatch, caplog):
    """The origin is an addition to the event; failing to read it must not cost the callback."""
    import flow_index_vault as flow
    docs, index = roots
    queued = []
    monkeypatch.setattr(flow, 'document_origin', lambda *a: (_ for _ in ()).throw(RuntimeError('locked')))
    monkeypatch.setattr(flow, 'queue_event', lambda hooks, event, outbox: queued.append(event) or [])
    monkeypatch.setattr(flow, 'send_enqueued', lambda *a, **k: {'retry_pending': 0, 'redrive_required': 0})
    event = {'event': 'document.indexed', 'doc_id': 'documents::00001'}
    import logging
    flow._dispatch_document_indexed_event({'index_root': str(index), 'event_hooks': {}}, event, logging.getLogger('t'))
    assert queued == [event]
