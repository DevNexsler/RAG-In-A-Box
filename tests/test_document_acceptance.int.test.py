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
