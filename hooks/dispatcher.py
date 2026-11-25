"""Config-driven event hook dispatcher."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from hooks.http import HookSendResult, send_http_event

Sender = Callable[[dict[str, Any], dict[str, Any]], HookSendResult]


def _event_matches(hook: dict[str, Any], event_name: str) -> bool:
    events = hook.get("events")
    if not events:
        return True
    if not isinstance(events, list):
        return False
    return event_name in {str(item) for item in events}


def _origin_matches(hook: dict[str, Any], event: dict[str, Any] | None) -> bool:
    """A hook that names ``origins`` hears only about documents those systems deposited."""
    origins = hook.get("origins")
    if not origins:
        return True
    if not isinstance(origins, list):
        return False
    origin = (event or {}).get("origin")
    system = origin.get("system") if isinstance(origin, dict) else None
    return system in {str(item) for item in origins}


def matching_hooks(
    config: dict[str, Any] | None, event_name: str, event: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return configured HTTP hooks whose event and origin filters match."""
    if not config or not config.get("enabled", False):
        return []
    hooks = config.get("hooks")
    if not isinstance(hooks, list):
        return []
    return [
        hook
        for hook in hooks
        if isinstance(hook, dict)
        and str(hook.get("type") or "http") == "http"
        and _event_matches(hook, event_name)
        and _origin_matches(hook, event)
    ]


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

    warnings: list[str] = []
    for hook in matching_hooks(config, str(event.get("event") or ""), event):
        try:
            result = sender(hook, event)
        except Exception:
            result = HookSendResult(False, "transport_error", True, error="hook sender failed")
        if not result.accepted:
            warnings.append(f"hook {hook.get('name') or 'unnamed'} failed: {result.outcome}")
    return warnings
