"""HTTP event hook delivery."""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HookSendResult:
    accepted: bool
    outcome: str
    retryable: bool
    http_status: int | None = None
    error: str | None = None


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


def send_http_event(hook: dict[str, Any], event: dict[str, Any]) -> HookSendResult:
    """Send one event to one HTTP hook and return its safe delivery result."""
    url, env_driven = _resolve_url(hook)
    if not url:
        if env_driven:
            return HookSendResult(True, "disabled", False)
        return HookSendResult(False, "configuration_error", False, error="missing url")

    headers = {"Content-Type": "application/json"}
    secret_env = str(hook.get("secret_env") or "").strip()
    if secret_env:
        secret = os.environ.get(secret_env)
        if not secret:
            return HookSendResult(False, "configuration_error", False, error="missing hook secret")
        headers["X-RAG-Hook-Secret"] = secret

    try:
        timeout = float(hook.get("timeout_seconds") or 5)
    except (TypeError, ValueError):
        return HookSendResult(False, "configuration_error", False, error="invalid timeout")

    configured_statuses = hook.get("accepted_statuses")
    if configured_statuses is not None:
        if not isinstance(configured_statuses, list) or not all(isinstance(status, str) and status for status in configured_statuses):
            return HookSendResult(False, "configuration_error", False, error="invalid accepted statuses")
        accepted_statuses = set(configured_statuses)
    else:
        accepted_statuses = None

    try:
        body = json.dumps(event, default=str).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 0)
            if status < 200 or status >= 300:
                return HookSendResult(False, "http_error", status >= 500, http_status=status, error="HTTP request failed")
            if accepted_statuses is None:
                return HookSendResult(True, "accepted", False, http_status=status)
            try:
                response_body = json.loads(response.read().decode("utf-8"))
            except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
                return HookSendResult(False, "malformed_response", True, http_status=status, error="invalid semantic response")
    except Exception:
        return HookSendResult(False, "transport_error", True, error="HTTP request failed")

    outcome = response_body.get("status") if isinstance(response_body, dict) else None
    if outcome in accepted_statuses:
        return HookSendResult(True, outcome, False, http_status=status)
    if outcome in {"no_match", "ambiguous", "correlation_mismatch"}:
        return HookSendResult(False, outcome, False, http_status=status)
    return HookSendResult(False, "unexpected_semantic_status", True, http_status=status, error="unexpected semantic response")
