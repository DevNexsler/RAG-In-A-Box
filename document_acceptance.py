"""Durable identity-first file acceptance; indexing consumes the existing queue."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import uuid

from core.artifacts import is_communication_sidecar_payload
from core.index_request_queue import IndexRequestQueue
from doc_id_store import DocIDStore, extract_id_from_filename, inject_id_into_filename, strip_id_from_filename

STAGING_PREFIX = '.organizer-accept-'
ORIGIN_SYSTEM = re.compile(r'[a-z0-9][a-z0-9._-]{0,63}')
ORIGIN_REF_MAX = 200
# Below the form parser's own per-field ceiling, so this is the limit a caller meets.
SIDECAR_MAX_BYTES = 500_000
# Other containers and the host's agents read the documents tree.
PUBLISHED_MODE = 0o644


class AcceptanceConflict(ValueError):
    """A retry changed its input, or publication would overwrite existing bytes."""


class InvalidAcceptance(AcceptanceConflict):
    """The request itself is malformed; a retry with the same input cannot succeed."""


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


def _write_journal(path: Path, record: dict, *, mode: int | None = None):
    _mkdir(path.parent)
    with tempfile.NamedTemporaryFile(
        mode='w', dir=path.parent, prefix=STAGING_PREFIX, delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(record, stream)
            stream.flush()
            if mode is not None:
                os.fchmod(stream.fileno(), mode)
            os.fsync(stream.fileno())
            os.replace(temporary, path)
            _sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)


def checked_origin(origin) -> dict | None:
    """``{system, ref}`` naming the depositor and its own record, or a refusal."""
    if origin is None:
        return None
    if not isinstance(origin, dict) or set(origin) != {'system', 'ref'}:
        raise InvalidAcceptance('Origin must name exactly a system and a ref')
    system, ref = origin['system'], origin['ref']
    if not isinstance(system, str) or not ORIGIN_SYSTEM.fullmatch(system):
        raise InvalidAcceptance('Invalid origin system')
    if not isinstance(ref, str) or not ref.strip() or len(ref) > ORIGIN_REF_MAX or ref != ref.strip():
        raise InvalidAcceptance('Invalid origin ref')
    return {'system': system, 'ref': ref}


def checked_sidecar(sidecar, filename: str) -> dict | None:
    """Attachment metadata the indexer reads beside the file, or a refusal.

    Only the communication sidecar the scanner already keeps out of the index
    is taken: any other JSON beside a document would be indexed as one.
    """
    if sidecar is None:
        return None
    if Path(filename).suffix.lower() == '.json':
        raise InvalidAcceptance('A JSON document cannot carry a sidecar')
    if not is_communication_sidecar_payload(sidecar):
        raise InvalidAcceptance('Sidecar is not communication attachment metadata')
    if len(json.dumps(sidecar).encode()) > SIDECAR_MAX_BYTES:
        raise InvalidAcceptance('Sidecar exceeds size limit')
    return sidecar


def _is_sidecar_file(entry: Path, sidecar_path: Path) -> bool:
    if entry.name != sidecar_path.name or entry.is_symlink() or not entry.is_file():
        return False
    try:
        return is_communication_sidecar_payload(json.loads(entry.read_text()))
    except (OSError, ValueError):
        return False


def _journal_name(root: Path, idempotency_key: str) -> str:
    return hashlib.sha256(json.dumps([str(root), idempotency_key]).encode()).hexdigest() + '.json'


def accepted_receipt(root: Path, registry_db: Path, idempotency_key: str) -> dict | None:
    """What a completed acceptance under this key answered, or None.

    A depositor that lost the answer asks here before sending the bytes
    again. The receipt is the acceptance as it happened; ``current_rel_path``
    is where the document is now, None once it is gone.

    Asking only reads: no registry is opened for writing, so a lookup is
    answered while an acceptance or a scan holds the registry.
    """
    journal = Path(registry_db).parent / 'document-acceptance'
    try:
        record = json.loads((journal / 'receipts' / _journal_name(Path(root).resolve(), idempotency_key)).read_text())
    except FileNotFoundError:
        return None
    return {**record['receipt'], 'current_rel_path': DocIDStore.read_path(registry_db, record['id'])}


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
        if record.get('sidecar') is not None:
            # Beside the file, before indexing is queued: the indexer reads it
            # as the document's communication context.
            # Written once and then left alone: the indexer adds to it, and a
            # retry of this acceptance must not undo that.
            sidecar = self._path(receipt['sidecar_rel_path'])
            if not os.path.lexists(sidecar):
                _write_journal(sidecar, record['sidecar'], mode=PUBLISHED_MODE)
        if self.registry.is_retired(record['id']):
            raise AcceptanceConflict('Accepted identity was retired; staff review required')
        previous = self.registry.lookup_path(record['id'])
        if previous not in (None, receipt['rel_path']):
            raise AcceptanceConflict('Accepted identity belongs to another path')
        self.registry.register(record['id'], receipt['rel_path'], source_name=record['source_name'])
        if record.get('origin') is not None:
            self.registry.record_origin(record['id'], **record['origin'])
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

    def _journal_name(self, idempotency_key: str) -> str:
        return _journal_name(self.root, idempotency_key)

    def recover(self):
        """Caller holds identity_lock; scanner recovers before adjudicating filenames."""
        for path in sorted((self.journal / 'pending').glob('*.json')):
            record = json.loads(path.read_text())
            if record['root'] == str(self.root):
                self._finish(path, record)

    def accept(self, directory: Path, filename: str, content: bytes, *,
               idempotency_key: str | None = None, origin: dict | None = None,
               sidecar: dict | None = None):
        """Publish ``content`` under a new identity and queue its indexing.

        ``origin`` names the depositing system and its own record; it is
        returned on the receipt and on the document's ``document.indexed``
        event. ``sidecar`` is the communication attachment metadata, written
        beside the file under the file's own name.

        An ``Idempotency-Key`` covers the bytes, the destination and the
        origin. The sidecar is not part of it: a retry carries metadata built
        at another moment (a fetch time, an edited message body), and the
        first acceptance's stands.
        """
        directory = directory.resolve()
        if not directory.is_relative_to(self.root):
            raise AcceptanceConflict('Acceptance directory escapes documents root')
        filename = strip_id_from_filename(Path(filename).name)
        if not filename or filename.startswith('.'):
            raise AcceptanceConflict('Invalid filename')
        origin = checked_origin(origin)
        sidecar = checked_sidecar(sidecar, filename)
        digest = hashlib.sha256(content).hexdigest()
        fingerprint = [str(self.root), self.source_name, str(directory.relative_to(self.root)), filename, digest]
        if origin is not None:
            # Appended only when given, so receipts journalled before origins
            # existed still replay under their key.
            fingerprint.append({'origin': origin})
        pending = self.journal / 'pending' / self._journal_name(idempotency_key or str(uuid.uuid4()))
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
            sidecar_path = (directory / filename).with_suffix('.json')
            if sidecar is not None and directory.is_dir() and any(
                Path(strip_id_from_filename(entry.name)).stem == Path(filename).stem
                for entry in directory.iterdir()
                # Metadata whose file is gone is no other document: it stays,
                # and the file deposited now is the one it describes.
                if not entry.name.startswith(STAGING_PREFIX) and not _is_sidecar_file(entry, sidecar_path)
            ):
                # A sidecar belongs to every document of its name in the folder,
                # so the name must be this document's alone. Refused before an
                # identity is spent or a byte is published.
                raise AcceptanceConflict(
                    'Another document of this name is here; its metadata would be shared')
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
                    os.fchmod(stream.fileno(), PUBLISHED_MODE)
                    os.fsync(stream.fileno())
                except BaseException:
                    stage.unlink(missing_ok=True)
                    raise
            _sync_directory(directory)
            receipt = {'accepted': True, 'doc_id': f'{self.source_name}::{doc_id}',
                       'rel_path': target.relative_to(self.root).as_posix(),
                       'filename': filename, 'sha256': digest, 'size': len(content),
                       'index_status': 'pending'}
            if origin is not None:
                receipt['origin'] = origin
            if sidecar is not None:
                # Without the identity tag, as depositors have always written
                # it: the scan re-issues any tag it finds on a sidecar.
                receipt['sidecar_rel_path'] = sidecar_path.relative_to(self.root).as_posix()
            record = {'root': str(self.root), 'id': doc_id, 'source_name': self.source_name,
                      'table_name': self.table_name, 'stage': stage.relative_to(self.root).as_posix(),
                      'fingerprint': fingerprint, 'receipt': receipt,
                      'origin': origin, 'sidecar': sidecar}
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
