"""Bounded exact-identifier chronology shared by context-builder consumers.

Pure cursor queries: connection ownership/read-only enforcement stays in cds_live.
Never broaden an identity via name, channel, semantic rank, or an inferred alias.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import re

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


def history_request(since=None, limit=DEFAULT_LIMIT, cursor=None, kind="messages") -> dict:
    if kind not in ("messages", "calls"):
        raise ValueError("history_kind must be messages or calls")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"history_limit must be an integer between 1 and {MAX_LIMIT}")
    since = timestamp(since).isoformat() if since is not None else None
    if since and timestamp(since) > dt.datetime.now(dt.timezone.utc):
        raise ValueError("history_since must not be in the future")
    if cursor is not None and (not isinstance(cursor, str) or not cursor or len(cursor) > 2048):
        raise ValueError("invalid history_cursor")
    return {"since": since, "limit": limit, "cursor": cursor,
            **({"kind": kind} if kind != "messages" else {})}


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


# The body Agent-Email-Server stores when an email's content is only inline
# images (inlineImageBodyMarker): no text of the sender's own.
_INLINE_IMAGE_MARKER = re.compile(
    r"\[Message body is \d+ inline images? and no text\. .{0,120}media attachments\.\]")
# One message's media text, the vision/OCR text before the conversation context
# the media pipeline appends; bounded like a body.
_MEDIA_TEXT = ("nullif(btrim(split_part(coalesce(mm.enrichment->>'text',''),"
               "'[Conversation context]',1),E' \\t\\r\\n'),'')")
MEDIA_SQL = f"""/* chronology_media_text */
    SELECT mm.message_id, count(*),
           left(string_agg({_MEDIA_TEXT}, E'\n\n' ORDER BY mm.id), %s),
           coalesce(length(string_agg({_MEDIA_TEXT}, E'\n\n' ORDER BY mm.id)), 0) > %s
    FROM message_media mm WHERE mm.message_id = ANY(%s) GROUP BY mm.message_id"""


def media_only_body(body) -> bool:
    """No text of the sender's own: blank, or the inline-image marker."""
    text = (body or "").strip()
    return not text or _INLINE_IMAGE_MARKER.fullmatch(text) is not None


def attach_media(cur, messages) -> None:
    """Every message on the page with media says so (media_count) and carries
    the media's extracted text (media_text, BODY_LIMIT chars). A media-only body
    also says whether its content was read: content_status extracted (it is in
    media_text) or not_extracted. Unflagged, readers reported a letter pasted as
    images as an empty email (CDS 839634), and a text with a photo hid the photo.
    One query for the page."""
    ids = [int(m["id"]) for m in messages]
    if not ids:
        return
    cur.execute(MEDIA_SQL, (BODY_LIMIT, BODY_LIMIT, ids))
    media = {int(row[0]): row[1:] for row in cur.fetchall()}
    for message in messages:
        count, text, truncated = media.get(int(message["id"]), (0, None, False))
        if not count:
            continue
        message["media_count"] = count
        if text:
            message["media_text"] = text
            message["media_text_truncated"] = bool(truncated)
        if media_only_body(message["body"]):
            message["content_status"] = "extracted" if text else "not_extracted"


def fetch_conversation(cur, contact: dict) -> dict:
    email = (contact.get("email") or "").strip().lower()
    phone = contact.get("phone_e164")
    if not (email or phone):
        return unavailable_history("no_identifiers")
    request = history_request(**contact.get("history", {}))
    since, limit, cursor = request["since"], request["limit"], request["cursor"]
    scope = _scope(email, phone, since)
    calls = request.get("kind") == "calls"
    if calls:
        scope = hashlib.sha256((scope + ":calls").encode()).hexdigest()
    through = dt.datetime.now(dt.timezone.utc)
    before, before_id = None, None
    if cursor:
        through, before, before_id = _decode(cursor, scope, through)
    participants, recipients, identity_params, recipient_params = [], [], [], []
    mail_recipient = None
    if email:
        participants.append("lower(p.email)=lower(%s)")
        identity_params.append(email)
        # Same verified raw-mail lane as cds_live, guarded against malformed JSON.
        # Any direction: staff whose own mailbox is not ingested reach CDS as
        # 'inbound' copies in an archived mailbox, yet the contact is a recipient.
        mail_recipient = """(m.source='zoho_mail' AND EXISTS (
            SELECT 1 FROM jsonb_array_elements(CASE
                WHEN jsonb_typeof(r.payload->'participants')='array'
                THEN r.payload->'participants' ELSE '[]'::jsonb END) pt
            WHERE pt->>'kind' IN ('to','cc','bcc') AND lower(pt->>'address')=%s))"""
    if phone:
        participants.append("(p.phone=%s OR p.phone_number=%s)")
        identity_params.extend([phone, phone])
        # A Quo inbound's 'to' is our own line, so this lane stays outbound-only.
        recipients.append("(m.source='quo' AND r.payload->'data'->'object'->>'to'=%s)")
        recipient_params.append(phone)
    where = """EXISTS (SELECT 1 FROM message_participants mp
        JOIN participants p ON p.id=mp.participant_id
        WHERE mp.message_id=m.id AND (""" + " OR ".join(participants) + "))"
    if mail_recipient:
        where += " OR " + mail_recipient
        recipient_params.insert(0, email)  # Precedes the Quo lane in the SQL.
    if recipients:
        where += " OR (m.direction='outbound' AND (" + " OR ".join(recipients) + "))"
    if email and not calls:
        # Cliq raw sender identity survives absent normalized participant links.
        # Channel recipients and body mentions do not establish sender ownership.
        where += """ OR (m.source='zoho_cliq' AND EXISTS (
            SELECT 1 FROM jsonb_array_elements(CASE
                WHEN jsonb_typeof(r.payload->'participants')='array'
                THEN r.payload->'participants' ELSE '[]'::jsonb END) pt
            WHERE pt->>'kind'='sender' AND lower(pt->>'address')=%s))"""
        recipient_params.append(email)
    table, event_time = "m", "m.sent_at"
    if calls:
        where = "EXISTS (SELECT 1 FROM participants p WHERE p.id=c.host_participant_id AND (" + " OR ".join(participants) + "))"
        recipient_params = []
        if phone:
            where += " OR c.from_number=%s OR c.to_number=%s"
            recipient_params = [phone, phone]
        table, event_time = "c", "c.started_at"
    params = [*identity_params, *recipient_params, through]
    bounds = f" AND {event_time} <= %s"
    if since:
        bounds += f" AND {event_time} >= %s"
        params.append(timestamp(since))
    if before:
        bounds += f" AND ({event_time},{table}.id) < (%s,%s)"
        params.extend([before, before_id])
    params.append(limit + 1)
    select = """/* contact_history */
        SELECT m.id,m.source,m.source_message_id,m.sent_at,m.direction,
               m.sender_name,m.subject,
               left(coalesce(nullif(m.body,''),nullif(m.body_text,''),m.content,''),4000),
               length(coalesce(nullif(m.body,''),nullif(m.body_text,''),m.content,''))>4000
        FROM messages m LEFT JOIN raw_events r ON r.id=m.raw_event_id"""
    if calls:
        # Discovery only. Exact-event retrieval owns transcript/metadata semantics.
        select = """/* contact_call_history */
            SELECT c.id,c.source,c.source_call_id,c.started_at,
                   CASE coalesce(c.direction, r.payload #>> '{data,object,direction}') WHEN 'incoming' THEN 'inbound'
                        WHEN 'outgoing' THEN 'outbound' ELSE coalesce(c.direction, r.payload #>> '{data,object,direction}') END,
                   NULL,NULL,'',false FROM calls c LEFT JOIN raw_events r ON r.id=c.raw_event_id"""
    cur.execute(select + " WHERE (" + where + ")" + bounds
                + f" ORDER BY {event_time} DESC,{table}.id DESC LIMIT %s", tuple(params))
    rows = cur.fetchall()
    has_more = len(rows) > limit
    rows = rows[:limit]
    messages = [dict(zip(("id", "source", "source_message_id", "sent_at", "direction", "sender_name", "subject", "body", "body_truncated"), row)) for row in rows]
    if not calls:
        attach_media(cur, messages)
    for message in messages:
        message["id"] = str(message["id"])
        if calls:
            message["id"] = "call:" + message["id"]
            message["event_kind"] = "call_reference"
        message["sent_at"] = message["sent_at"].isoformat()
    clipped = [m["id"] for m in messages if m["body_truncated"]]
    next_cursor = None
    if has_more:
        last = messages[-1]
        payload = {"v": 1, "scope": scope, "through": through.isoformat(),
                   "before_at": last["sent_at"], "before_id": int(last["id"].removeprefix("call:"))}
        next_cursor = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    return {"status": "degraded" if clipped else "ok", "messages": messages,
            "order": "sent_at_desc_id_desc", "since": since, "through": through.isoformat(),
            "limit": limit, "body_limit": BODY_LIMIT, "body_truncated_ids": clipped,
            "has_more": has_more, "next_cursor": next_cursor,
            "window_exhausted": not has_more,
            "coverage_complete": not cursor and not has_more and not clipped,
            "scope": ("Call references only; retrieve transcripts/metadata through exact-event mode. " if calls else "") + "Exact supplied identifiers in CDS only; no inferred aliases. A message with media_count has attachments; media_text is their extracted (OCR/vision) text, bounded. A media-only body (blank, or the inline-image marker) is not an empty message: content_status extracted means its content is media_text, not_extracted that its attachments are not read yet. Continuation pages must be accumulated and checked for truncation. Event-time upper bound is not a transactional snapshot."}
