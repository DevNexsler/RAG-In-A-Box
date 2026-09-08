"""Read-only exact-event pages for Context Builder, independent of contact search.

One reference/body fragment per page. Cursors bind references and source version;
callers must accumulate pages and validate offsets/hashes before using full bodies.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re

BODY_PAGE_CHARS = 12000


def event_request(refs, cursor=None):
    if not isinstance(refs, list) or not 1 <= len(refs) <= 20 or any(
        not isinstance(ref, str) or not ref.strip() or len(ref) > 512 for ref in refs
    ) or len(set(refs)) != len(refs):
        raise ValueError("event_refs must contain 1–20 unique nonempty source IDs")
    scope = hashlib.sha256(json.dumps(refs).encode()).hexdigest()
    state = {"v": 1, "scope": scope, "index": 0, "offset": 0, "version": None}
    if cursor is not None:
        try:
            if not isinstance(cursor, str) or not cursor or len(cursor) > 2048:
                raise ValueError
            state = json.loads(base64.b64decode(cursor, altchars=b"-_", validate=True))
            if state["v"] != 1 or state["scope"] != scope:
                raise ValueError
            if type(state["index"]) is not int or not 0 <= state["index"] < len(refs):
                raise ValueError
            if type(state["offset"]) is not int or not 0 <= state["offset"] <= 10_000_000:
                raise ValueError
            if state["offset"] and (not isinstance(state["version"], str) or len(state["version"]) != 64):
                raise ValueError
        except (ValueError, TypeError, KeyError, UnicodeError):
            raise ValueError("invalid event_cursor for references") from None
    return state


def fetch_event_page(cur, refs, state):
    index, offset = state["index"], state["offset"]
    ref = refs[index]
    body = "coalesce(nullif(body,''),nullif(body_text,''),content,'')"
    cur.execute(
        "/* exact_event_page */ SELECT id,source,source_message_id,sent_at,direction,sender_name,subject,"
        f"substring({body} from %s for %s),length({body}),"
        f"encode(sha256(convert_to({body},'UTF8')),'hex') "
        "FROM messages WHERE source_message_id=%s ORDER BY id LIMIT 2",
        (offset + 1, BODY_PAGE_CHARS, ref),
    )
    rows = cur.fetchall()
    if re.fullmatch(r'AC[0-9a-fA-F]{32}', ref):
        # Quo uses AC IDs for both SMS and calls. Check both typed stores;
        # collisions stay ambiguous, never silently prefer one source kind.
        transcript = 't.body'
        cur.execute(
            "/* exact_call_event_page */ SELECT 'call:' || c.id::text,c.source,c.source_call_id,c.started_at,"
            "CASE c.direction WHEN 'incoming' THEN 'inbound' WHEN 'outgoing' THEN 'outbound' ELSE c.direction END,"
            "NULL::text,'Call transcript',"
            f"substring({transcript} from %s for %s),length({transcript}),"
            f"encode(sha256(convert_to({transcript},'UTF8')),'hex') "
            "FROM calls c CROSS JOIN LATERAL (SELECT nullif(c.transcript,'') AS body UNION "
            "SELECT nullif(transcript_text,'') FROM transcripts WHERE call_id=c.id) t "
            "WHERE c.source_call_id=%s AND t.body IS NOT NULL ORDER BY c.id,t.body LIMIT 2",
            (offset + 1, BODY_PAGE_CHARS, ref),
        )
        rows += cur.fetchall()
    messages, missing, ambiguous = [], [], []
    next_state = {**state, "index": index + 1, "offset": 0, "version": None}
    if not rows:
        missing.append(ref)
    elif len(rows) != 1:
        ambiguous.append(ref)
    else:
        message = dict(zip(("id", "source", "source_message_id", "sent_at", "direction",
                            "sender_name", "subject", "body", "body_total_chars", "body_sha256"), rows[0]))
        message["id"] = str(message["id"])
        if message['id'].startswith('call:'):
            message['event_kind'] = 'call_transcript'
            message['body_authority'] = 'Stored call transcript, not a verbatim audio verification or summary.'
        message["sent_at"] = message["sent_at"].isoformat() if message["sent_at"] else None
        metadata = {k: v for k, v in message.items() if k != "body"}
        version = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        if offset > message["body_total_chars"] or (state["version"] and state["version"] != version):
            raise ValueError("event changed during pagination; restart retrieval")
        message.update(body_offset=offset, body_truncated=offset + len(message["body"]) < message["body_total_chars"],
                       source_version=version)
        messages.append(message)
        if message["body_truncated"]:
            next_state = {**state, "offset": offset + len(message["body"]), "version": version}
    if offset and (missing or ambiguous):
        raise ValueError("event changed during pagination; restart retrieval")
    has_more = next_state["index"] < len(refs)
    cursor = base64.urlsafe_b64encode(json.dumps(next_state).encode()).decode() if has_more else None
    return {"status": "degraded" if missing or ambiguous else "ok", "messages": messages,
            "missing_refs": missing, "ambiguous_refs": ambiguous, "requested_refs": refs,
            "has_more": has_more, "next_cursor": cursor, "window_exhausted": not has_more,
            "body_page_chars": BODY_PAGE_CHARS,
            "scope": "Exact source IDs only, not proof of contact identity. Duplicate IDs are ambiguous. Accumulate body fragments and check offsets, length and SHA-256. No transactional snapshot across events."}


def exact_event_context(refs, cursor=None):
    state = event_request(refs, cursor)
    import cds_live
    try:
        conn = cds_live._get_readonly_conn()
        try:
            with conn.cursor() as cur:
                page = fetch_event_page(cur, refs, state)
        finally:
            conn.rollback()
        return {"cds": {"status": "ok", "events": page}}
    except ValueError:
        raise
    except Exception:
        return {"cds": {"status": "unavailable", "events": {"status": "unavailable", "messages": [],
                "has_more": None, "next_cursor": None, "window_exhausted": False}}}
