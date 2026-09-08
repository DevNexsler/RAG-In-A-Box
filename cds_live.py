"""cds_live: hardened read-only CDS (comm-data-store) queries.

Mirrors the connection discipline of ``sor_query.py:109-146`` exactly (lazy,
recovering, read-only connection; ``default_transaction_read_only=on`` and a
15s statement timeout enforced at the libpq level so they survive reconnects)
but against ``COMM_DATA_STORE_DSN`` instead of the SOR DSN, and never reuses
the indexing connection in ``sources/postgres.py``.

Query functions take an open cursor (or a fake standing in for one in tests)
so they are testable without a live database; ``cds_source`` owns the
connection lifecycle, wraps any failure into a per-call ``status: "error:..."``
result (never raises for a source failure), and calls ``_reset_conn()`` on
failure so the next call opens a fresh connection.

zoho_mail outbound recipient path (live-probed 2026-08-27 against the runtime
CDS on host 127.0.0.1:5433, comm_data_store db, read-only SELECT only): every
zoho_mail outbound row's ``raw_events.payload`` carries a ``participants``
array (verified 3957/3957 rows), each element shaped like
``{"kind": "to"|"cc"|"bcc"|"from", "address": "...", "name": "..."}``. The
recipient(s) are the elements with ``kind = 'to'``. (``identity_evidence`` is
present on only ~12% of rows, so it is not a reliable path; ``participants``
is universal and is what lane 4 below uses.) ``RAW_EMAIL_SQL`` guards with
``jsonb_typeof(...) = 'array'`` before the lateral unnest so a future
non-array ``participants`` value on some row is skipped rather than raising
and degrading the whole ``cds_source`` call.
"""

from __future__ import annotations

import os

import psycopg

from core.logging_setup import MAX_ERROR_CHARS, collapse
from cds_history import fetch_conversation, unavailable_history

CDS_ENV_VAR = "COMM_DATA_STORE_DSN"

# Predicates are assembled dynamically per-call from these FIXED strings --
# never by interpolating a value into the SQL text, only the query's own
# placeholders are ever value-bearing. `coalesce(%s, '')` on an absent
# identifier used to match rows storing an empty-string column (an
# email-only contact spuriously matching p.phone = ''), so an identifier
# that isn't present on the contact contributes no predicate at all.
_INBOUND_BASE_SQL = """
select count(distinct m.id) filter (where m.sent_at > now() - interval '30 days'), max(m.sent_at)
from messages m
join message_participants mp on mp.message_id = m.id
join participants p on p.id = mp.participant_id
where m.direction = 'inbound'
  and m.sent_at <= now()
  and ({predicate})
"""
_INBOUND_EMAIL_PRED = "lower(p.email) = lower(%s)"
_INBOUND_PHONE_PRED = "(p.phone = %s or p.phone_number = %s)"

OUTBOUND_ACTIONS_SQL = """
select created_at, operation,
       coalesce(provider_message_id, action_uid::text, id::text)
from outbound_actions
where status = 'completed' and channel = any(%s)
order by created_at desc limit 5
"""

RAW_QUO_SQL = """
select m.sent_at, r.payload->'data'->'object'->>'id'
from messages m join raw_events r on r.id = m.raw_event_id
where m.direction = 'outbound' and m.source = 'quo'
  and r.payload->'data'->'object'->>'to' = %s
order by m.sent_at desc limit 5
"""

# Recipient path pinned by the Step 1 live probe: participants[] elements with
# kind='to', matched case-insensitively against the contact's email. The
# jsonb_typeof guard sits on the raw_events JOIN (not the trailing WHERE) so
# it gates which rows ever reach `jsonb_array_elements` in the LATERAL clause
# below — a WHERE-only guard would run too late, after a non-array
# `participants` value had already raised inside the unnest. This keeps a
# future malformed payload from taking down the whole cds_source call.
RAW_EMAIL_SQL = """
select m.sent_at, r.payload->'data'->'object'->>'id'
from messages m
join raw_events r on r.id = m.raw_event_id
  and jsonb_typeof(r.payload->'participants') = 'array'
cross join lateral jsonb_array_elements(r.payload->'participants') as pt
where m.direction = 'outbound' and m.source = 'zoho_mail'
  and pt->>'kind' = 'to'
  and lower(pt->>'address') = lower(%s)
order by m.sent_at desc limit 5
"""


def _ts(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return value.isoformat()


def fetch_inbound_summary(cur, email, phone) -> dict:
    """Recent count plus all-time latest inbound, including quiet contacts.

    Only identifiers actually present on the contact get a predicate (see
    `_INBOUND_BASE_SQL`'s docstring note above) -- an email-only contact
    never runs the phone predicate at all, so it can't match a row with an
    empty-string phone/phone_number column.
    """
    preds, params = [], []
    if email:
        preds.append(_INBOUND_EMAIL_PRED)
        params.append(email)
    if phone:
        preds.append(_INBOUND_PHONE_PRED)
        params.extend([phone, phone])
    if not preds:
        return {"inbound_count_30d": 0, "latest_inbound_at": None}

    sql = _INBOUND_BASE_SQL.format(predicate=" or ".join(preds))
    cur.execute(sql, tuple(params))
    row = cur.fetchone()
    count, latest = (row[0], row[1]) if row else (0, None)
    return {"inbound_count_30d": count or 0, "latest_inbound_at": _ts(latest)}


def fetch_outbound_evidence(cur, email, phone, lead_id) -> list[dict]:
    """Our-outbound evidence across every lane we can positively identify,
    merged newest first. Every item has the same shape:
    {"lane", "at", "operation", "ref"}. outbound_actions carries its own
    `operation` column value; the raw lanes don't have one in their payload,
    so they're stamped with the operation implied by the lane itself
    ("quo.sms.send" for raw_quo, "email.send" for raw_email).
    """
    channels = [v for v in (email, phone, lead_id) if v]
    items: list[dict] = []
    seen: set[tuple] = set()

    def _add(at, lane, operation, ref):
        at_iso = _ts(at)
        key = (at_iso, ref)
        if key in seen:
            return  # same evidence row surfaced by more than one lane
        seen.add(key)
        items.append({"lane": lane, "at": at_iso, "operation": operation, "ref": ref})

    cur.execute(OUTBOUND_ACTIONS_SQL, (channels,))
    for at, operation, ref in cur.fetchall():
        _add(at, "outbound_actions", operation, ref)

    if phone:
        cur.execute(RAW_QUO_SQL, (phone,))
        for at, ref in cur.fetchall():
            _add(at, "raw_quo", "quo.sms.send", ref)

    if email:
        cur.execute(RAW_EMAIL_SQL, (email,))
        for at, ref in cur.fetchall():
            _add(at, "raw_email", "email.send", ref)

    items.sort(key=lambda item: item["at"] or "", reverse=True)
    return items


_CONN: "psycopg.Connection | None" = None


def _get_readonly_conn() -> "psycopg.Connection":
    """Lazy, recovering, read-only connection in the MCP process.

    Read-only + a 15s statement timeout are enforced at the libpq level via
    connection options, so they apply to every query and survive reconnects.
    Mirrors ``sor_query.py:_get_readonly_conn`` but targets
    ``COMM_DATA_STORE_DSN`` and never touches the indexing connection in
    ``sources/postgres.py``. Uses the default (tuple) row factory rather than
    ``sor_query``'s ``dict_row`` — the fetch functions above unpack rows
    positionally so they stay testable with plain-tuple fakes.
    """
    global _CONN
    from psycopg.pq import TransactionStatus
    stale = (
        _CONN is None
        or _CONN.closed
        or _CONN.pgconn.transaction_status == TransactionStatus.INERROR
    )
    if stale:
        if _CONN is not None and not _CONN.closed:
            try:
                _CONN.close()
            except Exception:
                pass
        dsn = os.environ.get(CDS_ENV_VAR)
        if not dsn:
            raise RuntimeError(f"{CDS_ENV_VAR} is not set")
        _CONN = psycopg.connect(
            dsn,
            options="-c default_transaction_read_only=on -c statement_timeout=15000",
        )
    return _CONN


def _reset_conn() -> None:
    """Drop the cached connection so the next call opens a fresh one."""
    global _CONN
    if _CONN is not None and not _CONN.closed:
        try:
            _CONN.close()
        except Exception:
            pass
    _CONN = None


_EMPTY = {"inbound_count_30d": 0, "latest_inbound_at": None,
          "latest_outbound_at": None, "outbound_evidence": []}


def cds_source(contact: dict) -> dict:
    """Injectable ``cds`` dep for ``context_builder.build_context``.

    Never raises: source failures come back as ``status: "error:<detail>"``
    with empty fields, degrading loud instead of blowing up the whole dossier.
    """
    email = contact.get("email")
    phone = contact.get("phone_e164")
    lead_id = contact.get("lead_id")
    if not (email or phone or lead_id):
        return {"status": "no_identifiers", **_EMPTY, "conversation": unavailable_history("no_identifiers")}

    try:
        conn = _get_readonly_conn()
        with conn.cursor() as cur:
            inbound = fetch_inbound_summary(cur, email, phone)
            outbound_evidence = fetch_outbound_evidence(cur, email, phone, lead_id)
            conversation = fetch_conversation(cur, contact)
        conn.rollback()  # close the read-only txn cleanly
    except Exception as exc:
        _reset_conn()
        # collapse (not bare str(exc)): a DSN-parse or auth failure can echo
        # credentials into the exception text (same discipline as
        # factbook_client._call_tool's error path).
        return {"status": f"error:{collapse(exc, MAX_ERROR_CHARS)}", **_EMPTY,
                "conversation": unavailable_history("error")}

    latest_outbound_at = outbound_evidence[0]["at"] if outbound_evidence else None
    return {
        "status": "ok",
        "inbound_count_30d": inbound["inbound_count_30d"],
        "latest_inbound_at": inbound["latest_inbound_at"],
        "latest_outbound_at": latest_outbound_at,
        "outbound_evidence": outbound_evidence,
        "conversation": conversation,
    }
