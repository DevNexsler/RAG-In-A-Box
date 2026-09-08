"""Bounded exact-identifier chronology shared by context-builder consumers.

Pure cursor queries: connection ownership/read-only enforcement stays in cds_live.
Never broaden an identity via name, channel, semantic rank, or an inferred alias.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json

BODY_LIMIT = 4000
DEFAULT_LIMIT = 50
MAX_LIMIT = 100


def timestamp(value: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError
        return parsed.astimezone(dt.timezone.utc)
    except (AttributeError, TypeError, ValueError):
        raise ValueError("history timestamps must be timezone-aware ISO-8601") from None


def history_request(since=None, limit=DEFAULT_LIMIT, cursor=None) -> dict:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"history_limit must be an integer between 1 and {MAX_LIMIT}")
    since = timestamp(since).isoformat() if since is not None else None
    if since and timestamp(since) > dt.datetime.now(dt.timezone.utc):
        raise ValueError("history_since must not be in the future")
    if cursor is not None and (not isinstance(cursor, str) or not cursor or len(cursor) > 2048):
        raise ValueError("invalid history_cursor")
    return {"since": since, "limit": limit, "cursor": cursor}


def unavailable_history(status="unavailable") -> dict:
    return {"status": status, "messages": [], "coverage_complete": False,
            "has_more": None, "next_cursor": None, "window_exhausted": False}


def _scope(email, phone, since):
    return hashlib.sha256(json.dumps([email, phone, since]).encode()).hexdigest()


def _decode(cursor, scope, now):
    try:
        value = json.loads(base64.urlsafe_b64decode(cursor.encode()))
        if value["v"] != 1 or value["scope"] != scope:
            raise ValueError
        through = timestamp(value["through"])
        before = timestamp(value["before_at"])
        row_id = value["before_id"]
        if through > now or before > through or isinstance(row_id, bool) or not isinstance(row_id, int) or not 0 < row_id < 2**63:
            raise ValueError
        return through, before, row_id
    except (ValueError, TypeError, KeyError, UnicodeError):
        raise ValueError("invalid history_cursor for contact/window") from None


def fetch_conversation(cur, contact: dict) -> dict:
    email = (contact.get("email") or "").strip().lower()
    phone = contact.get("phone_e164")
    if not (email or phone):
        return unavailable_history("no_identifiers")
    request = history_request(**contact.get("history", {}))
    since, limit, cursor = request["since"], request["limit"], request["cursor"]
    scope = _scope(email, phone, since)
    through = dt.datetime.now(dt.timezone.utc)
    before, before_id = None, None
    if cursor:
        through, before, before_id = _decode(cursor, scope, through)
    participants, recipients, identity_params, recipient_params = [], [], [], []
    if email:
        participants.append("lower(p.email)=lower(%s)")
        identity_params.append(email)
        # Same verified raw-mail lane as cds_live, guarded against malformed JSON.
        recipients.append("""(m.source='zoho_mail' AND EXISTS (
            SELECT 1 FROM jsonb_array_elements(CASE
                WHEN jsonb_typeof(r.payload->'participants')='array'
                THEN r.payload->'participants' ELSE '[]'::jsonb END) pt
            WHERE pt->>'kind' IN ('to','cc','bcc') AND lower(pt->>'address')=%s))""")
        recipient_params.append(email)
    if phone:
        participants.append("(p.phone=%s OR p.phone_number=%s)")
        identity_params.extend([phone, phone])
        recipients.append("(m.source='quo' AND r.payload->'data'->'object'->>'to'=%s)")
        recipient_params.append(phone)
    where = """EXISTS (SELECT 1 FROM message_participants mp
        JOIN participants p ON p.id=mp.participant_id
        WHERE mp.message_id=m.id AND (""" + " OR ".join(participants) + "))"
    where += " OR (m.direction='outbound' AND (" + " OR ".join(recipients) + "))"
    params = [*identity_params, *recipient_params, through]
    bounds = " AND m.sent_at <= %s"
    if since:
        bounds += " AND m.sent_at >= %s"
        params.append(timestamp(since))
    if before:
        bounds += " AND (m.sent_at,m.id) < (%s,%s)"
        params.extend([before, before_id])
    params.append(limit + 1)
    cur.execute("""/* contact_history */
        SELECT m.id,m.source,m.source_message_id,m.sent_at,m.direction,
               m.sender_name,m.subject,
               left(coalesce(nullif(m.body,''),nullif(m.body_text,''),m.content,''),4000),
               length(coalesce(nullif(m.body,''),nullif(m.body_text,''),m.content,''))>4000
        FROM messages m LEFT JOIN raw_events r ON r.id=m.raw_event_id
        WHERE (""" + where + ")" + bounds + " ORDER BY m.sent_at DESC,m.id DESC LIMIT %s", tuple(params))
    rows = cur.fetchall()
    has_more = len(rows) > limit
    rows = rows[:limit]
    messages = [dict(zip(("id", "source", "source_message_id", "sent_at", "direction", "sender_name", "subject", "body", "body_truncated"), row)) for row in rows]
    for message in messages:
        message["id"] = str(message["id"])
        message["sent_at"] = message["sent_at"].isoformat()
    clipped = [m["id"] for m in messages if m["body_truncated"]]
    next_cursor = None
    if has_more:
        last = messages[-1]
        payload = {"v": 1, "scope": scope, "through": through.isoformat(),
                   "before_at": last["sent_at"], "before_id": int(last["id"])}
        next_cursor = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    return {"status": "degraded" if clipped else "ok", "messages": messages,
            "order": "sent_at_desc_id_desc", "since": since, "through": through.isoformat(),
            "limit": limit, "body_limit": BODY_LIMIT, "body_truncated_ids": clipped,
            "has_more": has_more, "next_cursor": next_cursor,
            "window_exhausted": not has_more,
            "coverage_complete": not cursor and not has_more and not clipped,
            "scope": "Exact supplied identifiers in CDS only; no inferred aliases. Continuation pages must be accumulated and checked for truncation. Event-time upper bound is not a transactional snapshot."}
