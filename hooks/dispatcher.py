"""Config-driven event hook dispatcher."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from hooks.http import send_http_event

Sender = Callable[[dict[str, Any], dict[str, Any]], str | None]


def _event_matches(hook: dict[str, Any], event_name: str) -> bool:
    events = hook.get("events")
    if not events:
        return True
    if not isinstance(events, list):
        return False
    return event_name in {str(item) for item in events}


def dispatch_event(
    config: dict[str, Any] | None,
    event: dict[str, Any],
    *,
    sender: Sender = send_http_event,
) -> list[str]:
    """Dispatch an event according to event_hooks config.

    Delivery failures are returned as warning strings instead of raised, so
    indexing remains independent from downstream systems.
    """
    if not config or not config.get("enabled", False):
        return []

    hooks = config.get("hooks") or []
    if not isinstance(hooks, list):
        return ["event_hooks.hooks ignored: expected list"]

    event_name = str(event.get("event") or "")
    warnings: list[str] = []
    for hook in hooks:
        if not isinstance(hook, dict):
            warnings.append("event hook ignored: expected mapping")
            continue
        if str(hook.get("type") or "http") != "http":
            warnings.append(f"hook {hook.get('name') or 'unnamed'} ignored: unsupported type")
            continue
        if not _event_matches(hook, event_name):
            continue
        warning = sender(hook, event)
        if warning:
            warnings.append(warning)
    return warnings
