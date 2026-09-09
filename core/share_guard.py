"""Optional live-share rename protection. Existing indexing stays operational."""
import http.client
import json
import os
from pathlib import Path
import socket
from urllib.parse import urlencode


def preserve_shared_path(path: Path) -> bool:
    """Preserve active paths or fail closed when a configured audit is unavailable."""
    socket_path = os.environ.get('QUANTUM_SHARE_GUARD_SOCKET')
    if not socket_path:
        return False
    try:
        root = Path(os.environ['QUANTUM_SHARE_GUARD_ROOT']).resolve(strict=True)
        resolved = Path(path).resolve(strict=True)
        if not resolved.is_relative_to(root):
            return False
        relative = resolved.relative_to(root).as_posix()
        connection = http.client.HTTPConnection('localhost', timeout=5)
        connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.sock.settimeout(5)
        try:
            connection.sock.connect(socket_path)
            connection.request('GET', '/check?' + urlencode({'path': relative}))
            response = connection.getresponse()
            if response.status != 200:
                return True
            result = json.loads(response.read(1024 * 1024))
            return result.get('safe_to_mutate') is not True
        finally:
            connection.close()
    except (OSError, ValueError, KeyError, http.client.HTTPException):
        return True
