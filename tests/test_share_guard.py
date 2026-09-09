"""Real scan behavior with active shares, audit failure, and unshared files."""
import json
import socketserver
import threading
from http.server import BaseHTTPRequestHandler

import pytest
from doc_id_store import DocIDStore
from flow_index_vault import scan_filesystem_records


def scan(root, registry):
    return scan_filesystem_records(root, ['**/*.md'], [], doc_id_store=registry)


def test_guard_outage_preserves_file_and_stable_identity(tmp_path, monkeypatch):
    root = tmp_path / 'docs'
    root.mkdir()
    (root / 'proposal.md').write_text('Proposal')
    monkeypatch.setenv('QUANTUM_SHARE_GUARD_SOCKET', str(tmp_path / 'missing.sock'))
    monkeypatch.setenv('QUANTUM_SHARE_GUARD_ROOT', str(root))
    registry = DocIDStore(tmp_path / 'registry.db')
    first = scan(root, registry)
    second = scan(root, registry)
    assert (root / 'proposal.md').read_text() == 'Proposal'
    assert first[0]['doc_id'] == second[0]['doc_id']
    registry.close()


@pytest.mark.parametrize('safe_to_mutate', [False, True])
def test_scan_respects_live_audit(tmp_path, monkeypatch, safe_to_mutate):
    root = tmp_path / 'docs'
    root.mkdir()
    (root / 'proposal.md').write_text('Proposal')
    socket = tmp_path / 'guard.sock'
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            requests.append(self.path)
            data = json.dumps({'safe_to_mutate': safe_to_mutate}).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = socketserver.UnixStreamServer(str(socket), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv('QUANTUM_SHARE_GUARD_SOCKET', str(socket))
    monkeypatch.setenv('QUANTUM_SHARE_GUARD_ROOT', str(root))
    registry = DocIDStore(tmp_path / 'registry.db')
    try:
        records = scan(root, registry)
        assert len(records) == 1
        assert (root / 'proposal.md').exists() is (not safe_to_mutate)
        assert requests == ['/check?path=proposal.md']
    finally:
        registry.close()
        server.shutdown()
        server.server_close()
        thread.join()
