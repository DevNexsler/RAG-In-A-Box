"""HTTP event hook delivery."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any


def send_http_event(hook: dict[str, Any], event: dict[str, Any]) -> str | None:
    """Send one event to one HTTP hook. Return warning text on failure."""
    name = str(hook.get("name") or "unnamed")
    url = str(hook.get("url") or "").strip()
    if not url:
        return f"hook {name} disabled: missing url"

    headers = {"Content-Type": "application/json"}
    secret_env = str(hook.get("secret_env") or "").strip()
    if secret_env:
        secret = os.environ.get(secret_env)
        if not secret:
            return f"hook {name} disabled: secret env {secret_env} is not set"
        headers["X-RAG-Hook-Secret"] = secret

    try:
        timeout = float(hook.get("timeout_seconds") or 5)
    except (TypeError, ValueError):
        return f"hook {name} disabled: invalid timeout_seconds {hook.get('timeout_seconds')!r}"

    try:
        body = json.dumps(event, default=str).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 0)
            if status < 200 or status >= 300:
                return f"hook {name} failed: HTTP {status}"
    except Exception as exc:
        return f"hook {name} failed: {exc}"
    return None
