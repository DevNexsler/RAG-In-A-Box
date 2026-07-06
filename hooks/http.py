"""HTTP event hook delivery."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any


def _resolve_url(hook: dict[str, Any]) -> tuple[str, bool]:
    """Resolve a hook url, expanding a ${ENV_VAR} reference from the environment
    (same convention as the postgres source dsn).

    Returns (url, env_driven). env_driven is True when the url was a ${VAR}
    form — so an empty result is an intentional "disabled" state (a silent
    no-op), not a misconfiguration. This lets a config carry an optional
    cross-repo callback (url: ${CDS_HOOK_URL}) that stays dormant until the
    env var is set, without spamming a warning per indexed document.

    Whole-string only: the url must be exactly "${VAR}" (no in-string
    interpolation like "${VAR}/path" or "http://x${VAR}") — matches the
    sources/postgres.py dsn convention. Put the full url in the env var.
    """
    raw = str(hook.get("url") or "").strip()
    if raw.startswith("${") and raw.endswith("}"):
        return os.environ.get(raw[2:-1], "").strip(), True
    return raw, False


def send_http_event(hook: dict[str, Any], event: dict[str, Any]) -> str | None:
    """Send one event to one HTTP hook. Return warning text on failure."""
    name = str(hook.get("name") or "unnamed")
    url, env_driven = _resolve_url(hook)
    if not url:
        if env_driven:
            return None  # env-driven callback, env var unset → deliberate no-op
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
