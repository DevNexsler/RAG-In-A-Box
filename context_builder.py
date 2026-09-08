"""context_builder: deterministic contact dossier (pure orchestration).

Sources are injected callables so this module has zero I/O. See
docs (spec in hermes repo): evidence-only, degrade-loud per source.
"""
from __future__ import annotations

import datetime as _dt
import re
import time

_DIGITS = re.compile(r"\D+")


def _phone_e164(value):
    digits = _DIGITS.sub("", str(value or ""))
    if len(digits) == 10:
        return "+1" + digits
    if len(digits) == 11 and digits.startswith("1"):
        return "+" + digits
    return None


def normalize_contact(email=None, phone=None, name=None, lead_id=None,
                      latest_inbound_at=None) -> dict:
    contact = {
        "email": (str(email).strip().lower() or None) if email else None,
        "phone_e164": _phone_e164(phone),
        "name": (str(name).strip() or None) if name else None,
        "lead_id": str(lead_id) if lead_id not in (None, "") else None,
        "latest_inbound_at": latest_inbound_at or None,
    }
    if not (contact["email"] or contact["phone_e164"] or contact["name"]
            or contact["lead_id"]):
        raise ValueError("at least one identifier required")
    return contact


def exact_hit(hit: dict, contact: dict) -> bool:
    """True if `hit` carries an exact identifier match for `contact`.

    Matches per-field (sender/channel/snippet individually), never against a
    concatenated haystack: joining the three fields before extracting digits
    lets an unrelated digit run in one field (e.g. a partial number in
    `channel`) butt up against an unrelated digit run in another (e.g. an
    extension in `snippet`) and accidentally spell out the contact's full
    phone number across the seam -- a false positive no single field
    actually contains.
    """
    fields = [str(hit.get(k) or "") for k in ("sender", "channel", "snippet")]
    lowered = [f.lower() for f in fields]
    digit_fields = [_DIGITS.sub("", f) for f in fields]

    email = contact.get("email")
    if email and any(email in f for f in lowered):
        return True

    phone_digits = _DIGITS.sub("", contact.get("phone_e164") or "")
    if phone_digits and any(phone_digits in d for d in digit_fields):
        return True

    name = contact.get("name")
    if name and any(name.lower() in f for f in lowered):
        return True

    return False


def _ts(value):
    if not value:
        return None
    try:
        parsed = _dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=_dt.timezone.utc)


def derive_flags(contact: dict, cds_result: dict) -> dict:
    if cds_result.get("status") != "ok":
        return {"our_outbound_after_latest_inbound": "unknown"}
    candidates = [_ts(contact.get("latest_inbound_at")), _ts(cds_result.get("latest_inbound_at"))]
    inbound = max((value for value in candidates if value is not None), default=None)
    if inbound is None:
        return {"our_outbound_after_latest_inbound": "unknown"}
    for item in cds_result.get("outbound_evidence") or []:
        out_ts = _ts(item.get("at"))
        if out_ts and out_ts > inbound:
            return {"our_outbound_after_latest_inbound": True}
    return {"our_outbound_after_latest_inbound": False}


# Single source of truth for each source's empty/skipped shape -- shared with
# mcp_server.py (`_CTX_EMPTY_BY_SOURCE = context_builder.EMPTY_BY_SOURCE`) so
# the two never drift out of sync.
EMPTY_BY_SOURCE = {
    "factbook": {"entities": [], "flags": {}},
    "cds": {"inbound_count_30d": 0, "latest_inbound_at": None,
            "latest_outbound_at": None, "outbound_evidence": []},
    "comm": {"hits": []},
}


def build_context(contact: dict, deps: dict) -> dict:
    started = time.monotonic()
    result: dict = {"contact": contact}
    for source, empty in EMPTY_BY_SOURCE.items():
        try:
            result["comm_context" if source == "comm" else source] = deps[source](contact)
        except Exception as exc:
            result["comm_context" if source == "comm" else source] = {
                "status": f"error:{exc}", **empty}
    result["derived"] = derive_flags(contact, result["cds"])
    result["elapsed_ms"] = int((time.monotonic() - started) * 1000)
    return result
