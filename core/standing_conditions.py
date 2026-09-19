"""Announce standing conditions on transition, not once per run.

A *standing condition* is an operator-actionable state that no future run can
change by itself — docs parked at the degraded ledger's terminal cap, docs
skipped as permanently corrupt. Reporting one at ERROR/WARNING every run turns
the level into noise: in a 24h production window, 117 of doc-organizer's 118
ERROR lines were the byte-identical "2 degraded docs parked at terminal cap"
for a set last changed 25 days earlier (#2101). An operator or alert rule keyed
on ERROR then cannot see the one real event in the same window.

The condition itself still has to be *visible*, so this is not a mute: the
condition is announced when it appears, whenever its membership changes, when
it clears, and — so it never disappears entirely — on a periodic digest. The
per-run INFO run summaries keep carrying the counts in between.

Membership is a mapping (``{doc_id: [reasons]}`` or ``{reason: [doc_ids]}``);
the signature covers both keys and values, so a doc whose reasons changed
counts as a change even though the key set did not.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

# Re-announce an unchanged standing condition this often, so it stays visible
# to anything watching the ERROR channel without dominating it. At the 15-min
# production sweep cadence this is ~1 line/day instead of ~116.
DIGEST_INTERVAL_SECONDS = 24 * 60 * 60

STATE_VERSION = 1


@dataclass(frozen=True)
class Announcement:
    """What to do about one standing condition this run."""

    key: str
    announce: bool
    cleared: bool
    first_seen_at: float | None
    last_announced_at: float | None
    #: "entered" | "changed" | "digest" | "cleared" | "standing" | "absent"
    transition: str

    def __bool__(self) -> bool:  # `if announcement:` reads as "report it"
        return self.announce


def _signature(members: Mapping[str, Iterable[str]]) -> str:
    normalized = sorted(
        (str(key), sorted(str(value) for value in values))
        for key, values in members.items()
    )
    payload = json.dumps(normalized, separators=(",", ":"))
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def observe(
    state: dict,
    key: str,
    members: Mapping[str, Iterable[str]],
    *,
    now: float | None = None,
    digest_interval_s: float = DIGEST_INTERVAL_SECONDS,
) -> tuple[dict, Announcement]:
    """Fold one observation of a standing condition into ``state``.

    Pure: returns the next state and what the caller should report. The state
    is only rebuilt when something actually changed, so an unchanged condition
    costs no write.
    """
    now = time.time() if now is None else now
    conditions = dict(state.get("conditions") or {})
    previous = conditions.get(key) if isinstance(conditions.get(key), dict) else None

    if not members:
        if previous is None:
            return state, Announcement(key, False, False, None, None, "absent")
        conditions.pop(key, None)
        return (
            {"version": STATE_VERSION, "conditions": conditions},
            Announcement(
                key, False, True,
                previous.get("first_seen_at"), previous.get("last_announced_at"),
                "cleared",
            ),
        )

    signature = _signature(members)
    if previous is None:
        transition = "entered"
    elif str(previous.get("signature") or "") != signature:
        transition = "changed"
    elif (now - float(previous.get("last_announced_at") or 0.0)) >= digest_interval_s:
        transition = "digest"
    else:
        transition = "standing"

    first_seen_at = (
        float(previous["first_seen_at"])
        if previous and previous.get("first_seen_at") is not None
        else now
    )
    if transition == "standing":
        return state, Announcement(
            key, False, False, first_seen_at,
            float(previous.get("last_announced_at") or 0.0), transition,
        )

    conditions[key] = {
        "signature": signature,
        "member_count": len(members),
        "first_seen_at": first_seen_at,
        "last_announced_at": now,
    }
    return (
        {"version": STATE_VERSION, "conditions": conditions},
        Announcement(key, True, False, first_seen_at, now, transition),
    )


def state_path(index_root: Path) -> Path:
    return Path(index_root) / "standing_conditions.json"


def load(index_root: Path) -> dict:
    try:
        payload = json.loads(state_path(index_root).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = None
    if not (isinstance(payload, dict) and isinstance(payload.get("conditions"), dict)):
        return {"version": STATE_VERSION, "conditions": {}}
    return payload


def save(index_root: Path, state: dict) -> bool:
    try:
        state_path(index_root).write_text(
            json.dumps(state, indent=2, sort_keys=True), encoding="utf-8"
        )
        return True
    except OSError as exc:
        logging.getLogger(__name__).warning(
            "Failed to save standing conditions: %s", exc
        )
        return False


def announce(
    index_root: Path,
    key: str,
    members: Mapping[str, Iterable[str]],
    *,
    now: float | None = None,
) -> Announcement:
    """Load, observe and persist one standing condition.

    A state file that cannot be read or written must never silence a standing
    condition, so both failures fall back to announcing it.
    """
    try:
        state = load(index_root)
        next_state, announcement = observe(state, key, members, now=now)
    except Exception:  # pragma: no cover - defensive; reporting must not break a run
        logging.getLogger(__name__).warning(
            "Standing condition %r could not be evaluated; reporting it", key,
            exc_info=True,
        )
        return Announcement(key, bool(members), False, None, None, "entered")
    if next_state is not state and not save(index_root, next_state):
        # The decision did not persist, so the next run re-derives it from the
        # stale file and reports again. Loud beats silent.
        return Announcement(
            key, bool(members), announcement.cleared,
            announcement.first_seen_at, announcement.last_announced_at,
            announcement.transition,
        )
    return announcement
