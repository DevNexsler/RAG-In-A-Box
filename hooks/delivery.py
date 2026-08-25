"""Durable delivery orchestration for configured event hooks."""

from __future__ import annotations

import logging
from typing import Any

from core.hook_outbox import HookOutbox
from hooks.dispatcher import matching_hooks
from hooks.http import HookSendResult, send_http_event


def queue_event(config: dict[str, Any] | None, event: dict[str, Any], outbox: HookOutbox) -> int:
    """Persist one delivery for every matching HTTP hook before sending."""
    hooks = matching_hooks(config, str(event.get("event") or ""))
    for hook in hooks:
        outbox.enqueue(event, hook)
    return len(hooks)


def drain_due(
    outbox: HookOutbox,
    *,
    limit: int,
    sender=send_http_event,
    logger: logging.Logger | None = None,
    now: float | None = None,
) -> dict[str, int]:
    """Send due deliveries and persist their accepted, retry, or redrive state.

    Each delivery is claimed before it is sent, so drains that overlap — one
    per index worker, plus the scheduler tick — never send the same event to
    the same hook twice.
    """
    log = logger or logging.getLogger(__name__)
    counts = {"accepted": 0, "retry_pending": 0, "redrive_required": 0}
    for pending in outbox.due(limit=limit, now=now):
        delivery = outbox.claim(pending, now=now)
        if delivery is None:
            # Another drain already owns this delivery and is sending it.
            log.debug("hook delivery skipped event_id=%s: claimed elsewhere", pending.event_id)
            continue
        try:
            result = sender(delivery.hook, delivery.event)
        except Exception:
            result = HookSendResult(False, "transport_error", True, error="sender_failed")

        if result.accepted:
            transitioned = outbox.complete(delivery)
            retry_state = "accepted" if transitioned is not None else "stale"
            if transitioned is not None:
                counts["accepted"] += 1
        elif result.retryable:
            transitioned = outbox.retry(delivery, result.outcome, result.error, now=now)
            retry_state = "redrive_required" if transitioned and transitioned.status == "redrive_required" else "retry_pending"
            if transitioned is not None:
                counts[retry_state] += 1
            elif retry_state == "retry_pending":
                retry_state = "stale"
        else:
            transitioned = outbox.redrive_required(delivery, result.outcome, result.error)
            retry_state = "redrive_required" if transitioned is not None else "stale"
            if transitioned is not None:
                counts["redrive_required"] += 1

        log.info(
            "hook delivery event_id=%s doc_id=%s hook_name=%s attempt=%s outcome=%s http_status=%s retry_state=%s",
            delivery.event_id,
            delivery.event.get("doc_id"),
            delivery.hook_name,
            delivery.attempts + 1,
            result.outcome,
            result.http_status,
            retry_state,
        )
    return counts
