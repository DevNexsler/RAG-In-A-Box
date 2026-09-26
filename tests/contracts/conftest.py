"""Real loopback peers for HTTP contracts; no production services or credentials."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest


@pytest.fixture
def http_peer():
    peer = SimpleNamespace(requests=[], respond=lambda request: (200, {}))

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            request = {
                "path": self.path,
                "headers": dict(self.headers.items()),
                "body": json.loads(raw),
            }
            peer.requests.append(request)
            status, body = peer.respond(request)
            encoded = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    peer.url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield peer
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive(), "contract-test server leaked"
