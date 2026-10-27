"""Durable identity-first file acceptance; indexing consumes the existing queue."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid

from core.index_request_queue import IndexRequestQueue
from doc_id_store import DocIDStore, extract_id_from_filename, inject_id_into_filename, strip_id_from_filename

STAGING_PREFIX = '.organizer-accept-'


class AcceptanceConflict(ValueError):
    """A retry changed its input, or publication would overwrite existing bytes."""


@contextmanager
def identity_lock(registry: DocIDStore):
    """Coordinate publication/registration with identity scans across processes."""
    if registry.db_path == ':memory:':
        with registry._lock:
            yield
        return
    with open(registry.db_path + '.identity.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _sync_directory(path: Path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _mkdir(path: Path):
    if not path.exists():
        _mkdir(path.parent)
        path.mkdir(exist_ok=True)
        _sync_directory(path.parent)


def _write_journal(path: Path, record: dict):
    _mkdir(path.parent)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(record, stream)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
            _sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)


class DocumentAcceptance:
    """Accept bytes once, return durable identity, recover interrupted publication."""

    def __init__(self, root: Path, registry: DocIDStore, *, source_name='documents', table_name='chunks', identity_roots=None):
        self.root = root.resolve()
        self.registry = registry
        self.source_name = source_name
        self.table_name = table_name
        self.identity_roots = tuple(Path(path) for path in (identity_roots or [self.root]))
        self.journal = Path(registry.db_path).parent / 'document-acceptance'

    def _path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise AcceptanceConflict('Acceptance path escapes documents root')
        return path

    def _finish(self, path: Path, record: dict) -> dict:
        receipt = record['receipt']
        target = self._path(receipt['rel_path'])
        stage = self._path(record['stage'])
        if not stage.name.startswith(STAGING_PREFIX):
            raise AcceptanceConflict('Invalid acceptance staging path')
        # A prior publication may have completed just before process death.
        if hashlib.sha256(stage.read_bytes()).hexdigest() != receipt['sha256']:
            raise AcceptanceConflict('Staged content changed before publication')
        if not target.exists():
            os.link(stage, target)  # Same-directory staging; exclusive, never replaces a file.
            _sync_directory(target.parent)
        elif not stage.exists() or not os.path.samefile(stage, target):
            raise AcceptanceConflict('Acceptance destination already exists')
        if self.registry.is_retired(record['id']):
            raise AcceptanceConflict('Accepted identity was retired; staff review required')
        previous = self.registry.lookup_path(record['id'])
        if previous not in (None, receipt['rel_path']):
            raise AcceptanceConflict('Accepted identity belongs to another path')
        self.registry.register(record['id'], receipt['rel_path'], source_name=record['source_name'])
        IndexRequestQueue(Path(self.registry.db_path).parent).enqueue(
            table_name=record['table_name'], source_name=record['source_name'],
            target=receipt['rel_path'], force=False,
        )
        completed = self.journal / 'receipts' / path.name
        _mkdir(completed.parent)
        os.replace(path, completed)
        _sync_directory(completed.parent)
        _sync_directory(path.parent)
        stage.unlink()
        _sync_directory(stage.parent)
        return receipt

    def recover(self):
        """Caller holds identity_lock; scanner recovers before adjudicating filenames."""
        for path in sorted((self.journal / 'pending').glob('*.json')):
            record = json.loads(path.read_text())
            if record['root'] == str(self.root):
                self._finish(path, record)

    def accept(self, directory: Path, filename: str, content: bytes, *, idempotency_key: str | None = None):
        directory = directory.resolve()
        if not directory.is_relative_to(self.root):
            raise AcceptanceConflict('Acceptance directory escapes documents root')
        filename = strip_id_from_filename(Path(filename).name)
        if not filename or filename.startswith('.'):
            raise AcceptanceConflict('Invalid filename')
        digest = hashlib.sha256(content).hexdigest()
        fingerprint = [str(self.root), self.source_name, str(directory.relative_to(self.root)), filename, digest]
        key = hashlib.sha256(json.dumps([str(self.root), idempotency_key or str(uuid.uuid4())]).encode()).hexdigest()
        pending = self.journal / 'pending' / (key + '.json')
        completed = self.journal / 'receipts' / pending.name
        with identity_lock(self.registry):
            interrupted = None
            for path in (completed, pending):
                if path.exists():
                    record = json.loads(path.read_text())
                    if record['fingerprint'] != fingerprint:
                        raise AcceptanceConflict('Idempotency-Key reused with different content or destination')
                    if path == completed:
                        return record['receipt']
                    interrupted = record
            owned_path = self._path(interrupted['receipt']['rel_path']) if interrupted else None
            if self.registry.count() == 0:
                # Filename claims are the only recovery evidence after a lost
                # registry. Creating its first row would disable scan bootstrap
                # and turn surviving identities into "unissued" tokens.
                for root in self.identity_roots:
                    if any(path.is_file() and path.resolve() != owned_path and extract_id_from_filename(path.name)
                           for path in root.rglob('*@?????@*')):
                        raise AcceptanceConflict('Run an identity scan to recover existing filename IDs first')
            if interrupted:
                return self._finish(pending, interrupted)
            _mkdir(directory)
            doc_id = self.registry.next_id()
            target = directory / inject_id_into_filename(filename, doc_id)
            if len(target.name.encode()) > os.pathconf(directory, 'PC_NAME_MAX'):
                raise AcceptanceConflict('Filename exceeds filesystem limit after adding identity')
            # ID allocation can skip values; it never adopts a client-supplied filename ID.
            while target.exists() or target.is_symlink():
                doc_id = self.registry.next_id()
                target = directory / inject_id_into_filename(filename, doc_id)
            with tempfile.NamedTemporaryFile(dir=directory, prefix=STAGING_PREFIX, delete=False) as stream:
                stage = Path(stream.name)
                try:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                except BaseException:
                    stage.unlink(missing_ok=True)
                    raise
            _sync_directory(directory)
            receipt = {'accepted': True, 'doc_id': f'{self.source_name}::{doc_id}',
                       'rel_path': target.relative_to(self.root).as_posix(),
                       'filename': filename, 'sha256': digest, 'size': len(content),
                       'index_status': 'pending'}
            record = {'root': str(self.root), 'id': doc_id, 'source_name': self.source_name,
                      'table_name': self.table_name, 'stage': stage.relative_to(self.root).as_posix(),
                      'fingerprint': fingerprint, 'receipt': receipt}
            try:
                _write_journal(pending, record)
            except BaseException:
                stage.unlink(missing_ok=True)
                raise
            return self._finish(pending, record)


def identity_scan(function):
    """Keep scans from observing half-accepted identities; recover before scanning."""
    @wraps(function)
    def locked(vault_root, include, exclude, doc_id_store=None, logger=None, no_rename_prefixes=None):
        exclusions = [*exclude, STAGING_PREFIX + '*', '**/' + STAGING_PREFIX + '*']
        args = (vault_root, include, exclusions, doc_id_store, logger, no_rename_prefixes)
        if doc_id_store is None:
            return function(*args)
        with identity_lock(doc_id_store):
            DocumentAcceptance(Path(vault_root), doc_id_store).recover()
            return function(*args)
    return locked
