"""Controlled vocabulary for ``enr_doc_type``.

``enr_doc_type`` is a published search facet. Free-text LLM labels made it
unusable: thousands of one-off synonyms and a fresh spelling on every re-index
(#0233, #1251, #3050). This module owns the vocabulary seed, write-time
constraint, reclassification stickiness, and enrichment-input hashing so a
document's type is a stable membership in the taxonomy, not a coin flip.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.enrichment_postprocess import canonicalize_doc_type


def _csv_values(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]

if TYPE_CHECKING:
    from taxonomy_store import TaxonomyStore

logger = logging.getLogger(__name__)

UNCLASSIFIED_DOC_TYPE = "unclassified"

# Canonical name, short description, comma-separated aliases (pre-canonical form).
# Head of the production facet plus synonym clusters that churned document types
# across re-indexes. Aliases are folded through canonicalize_doc_type before match.
DEFAULT_DOC_TYPE_ENTRIES: tuple[tuple[str, str, str], ...] = (
    ("email", "Email message", "e_mail,emails,email_message"),
    ("message", "Generic message or note", "msg,msgs"),
    ("pg_message", "Postgres-sourced message row", ""),
    ("notification", "Generic notification", "notices,notice"),
    ("system_notification", "Automated system notification", "system_notice"),
    ("system_alert", "System alert or error signal", "alert"),
    ("system_email", "Automated system email", ""),
    ("system_message", "Automated system message", ""),
    ("correspondence", "General correspondence", "letter"),
    ("delivery_notification", "Package or delivery status notice", "delivery_notice,delivery_confirmation"),
    ("newsletter", "Newsletter or digest", "digest"),
    ("marketing_email", "Marketing or promotional email", "promotional_email,marketing"),
    ("email_notification", "Email-delivered notification", ""),
    ("automated_notification", "Automated notification", "automated_message,automated_email"),
    ("calendar_notification", "Calendar notification", "calendar_reminder,calendar_invitation"),
    ("order_confirmation", "Order confirmation", "confirmation"),
    ("payment_notification", "Payment status notification", "payment_confirmation,payment_notice"),
    ("payment_log", "Payment history or ledger entry", "payment_history_log,financial_log,financial_ledger,payment_receipt"),
    ("collection_record", "Rent or debt collection record", "collections_record,collection_log"),
    ("invoice", "Invoice", "bill"),
    ("receipt", "Receipt", ""),
    ("financial_document", "Other financial document", "financial_statement"),
    ("legal", "Legal document or matter", "legal_document,legal_notice,legal_correspondence"),
    ("contract", "Contract or agreement", "agreement,operating_agreement"),
    ("lease_administration", "Lease administration", "lease,lease_renewal"),
    ("tax_return", "Tax return or filing", "tax,tax_document"),
    ("rental_inquiry", "Rental or prospect inquiry", "inquiry,rental_enquiry"),
    ("listing_report", "Listing or market report", "listing"),
    ("property_showing", "Property showing", "showing"),
    ("maintenance_request", "Maintenance request", "maintenance,maintenance_log"),
    ("maintenance_dispatch", "Maintenance dispatch", "maintenance_dispatch_log,maintenance_dispatch_record"),
    ("service_request", "Service request", "service_request_log,service_request_record"),
    ("incident_report", "Incident report", ""),
    ("work_order", "Work order", ""),
    ("estimate", "Estimate, quote, or proposal", "quote,proposal"),
    ("status_update", "Status update", ""),
    ("complaint", "Complaint", ""),
    ("acknowledgment", "Acknowledgment", ""),
    ("form", "Form", ""),
    ("template", "Template", ""),
    ("note", "Note", "notes,staff_notes"),
    ("task", "Task", "sor_task"),
    ("request", "Generic request", ""),
    ("scheduling", "Scheduling message", ""),
    ("chat_message", "Chat message", "chat,instant_message"),
    ("text_message", "SMS or text message", "sms"),
    ("phone_call", "Phone call", ""),
    ("transcript", "Call or meeting transcript", "pg_transcript,phone_call_transcript,voicemail_transcript"),
    ("voicemail", "Voicemail", ""),
    ("tenant_communication", "Tenant communication", "tenant_communication_log,tenant_account_summary,internal_communication,internal_message,communication,communication_log"),
    ("reply", "Reply message", ""),
    ("image", "Image", "photo,photograph,img,screenshot"),
    ("utility_notification", "Utility notification", ""),
    ("error_log", "Error or log entry", "log_entry"),
    ("document", "Generic document", ""),
    ("report", "General report", "engineering_report,geotechnical_report,financial_report"),
    ("recipe", "Recipe or how-to", ""),
    ("memo", "Memo", "memorandum"),
    ("engineering", "Engineering document", ""),
    ("follow_up", "Follow-up action or note", "followup"),
    ("leasing_follow_up", "Leasing follow-up", "leasing_followup"),
    ("prospect_follow_up", "Prospect follow-up", "prospect_followup"),
    ("maintenance_follow_up", "Maintenance follow-up", "maintenance_followup"),
    ("rental_application_follow_up", "Rental application follow-up", "rental_application_followup"),
    ("lead_follow_up", "Lead follow-up", "lead_followup"),
    ("application_follow_up", "Application follow-up", "application_followup"),
    ("rental_follow_up", "Rental follow-up", "rental_followup"),
    ("showing_follow_up", "Showing follow-up", "showing_followup"),
    ("payment_follow_up", "Payment follow-up", "payment_followup"),
    ("property_showing_follow_up", "Property showing follow-up", "property_showing_followup"),
    ("rental_lead_follow_up", "Rental lead follow-up", "rental_lead_followup"),
    ("rental_inquiry_follow_up", "Rental inquiry follow-up", "rental_inquiry_followup"),
    ("pay_stub", "Pay stub", "paystub"),
    ("health_check", "Health check", "healthcheck"),
    ("section_8", "Section 8 housing document", "section8"),
    ("w_9", "IRS W-9 form", "w9"),
    (UNCLASSIFIED_DOC_TYPE, "No taxonomy match for the model label", "other,unknown"),
)


@dataclass(frozen=True)
class DocTypeConstraintResult:
    """Outcome of mapping a free-text ``enr_doc_type`` onto the vocabulary."""

    value: str
    rejected: tuple[str, ...]
    unknown_count: int


def default_alias_map() -> dict[str, str]:
    """Build ``canonical_or_alias -> canonical`` from the shipped seed."""
    return _alias_map_from_entries(DEFAULT_DOC_TYPE_ENTRIES)


def vocabulary_alias_map(store: "TaxonomyStore | None") -> dict[str, str]:
    """Prefer live taxonomy ``doc_type`` rows; fall back to the shipped seed.

    Operators can extend the facet by adding taxonomy entries (with aliases);
    the seed keeps constraint fail-closed when the store has no doc_type rows yet.
    """
    if store is None:
        return default_alias_map()
    try:
        entries = store.list_by_kind("doc_type", status="active")
    except Exception as exc:
        logger.warning("Failed to load doc_type taxonomy for constraint: %s", exc)
        return default_alias_map()
    if not entries:
        return default_alias_map()
    mapped: list[tuple[str, str, str]] = []
    for entry in entries:
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        mapped.append(
            (
                name,
                str(entry.get("description") or ""),
                str(entry.get("aliases") or ""),
            )
        )
    if not any(canonicalize_doc_type(name) == UNCLASSIFIED_DOC_TYPE for name, _, _ in mapped):
        mapped.append((UNCLASSIFIED_DOC_TYPE, "No taxonomy match", "other,unknown"))
    return _alias_map_from_entries(mapped)


def _alias_map_from_entries(entries: list[tuple[str, str, str]] | tuple[tuple[str, str, str], ...]) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for name, _description, aliases in entries:
        canonical = canonicalize_doc_type(name)
        if not canonical:
            continue
        # Multi-label names are not valid vocabulary keys.
        if "," in canonical:
            canonical = canonical.split(",", 1)[0].strip()
        alias_map[canonical] = canonical
        for alias in _csv_values(aliases):
            folded = canonicalize_doc_type(alias)
            if folded:
                alias_map[folded] = canonical
    return alias_map


def constrain_doc_type(
    value: str,
    alias_map: Mapping[str, str] | None = None,
) -> DocTypeConstraintResult:
    """Map a CSV of labels onto the vocabulary.

    Known labels resolve through aliases to their canonical name. Unknown
    labels are dropped and counted; if the model emitted labels but none
    survive, the explicit ``unclassified`` sentinel is stored so the facet
    stays filterable. An empty input stays empty so required-field checks
    can still detect a missing ``doc_type``.
    """
    mapping = dict(alias_map) if alias_map is not None else default_alias_map()
    labels = _csv_values(canonicalize_doc_type(value))
    if not labels:
        return DocTypeConstraintResult(value="", rejected=(), unknown_count=0)

    kept: list[str] = []
    rejected: list[str] = []
    seen: set[str] = set()
    for label in labels:
        canonical = mapping.get(label)
        if canonical is None:
            rejected.append(label)
            continue
        if canonical not in seen:
            kept.append(canonical)
            seen.add(canonical)
    if not kept:
        return DocTypeConstraintResult(
            value=UNCLASSIFIED_DOC_TYPE,
            rejected=tuple(rejected),
            unknown_count=len(rejected),
        )
    return DocTypeConstraintResult(
        value=", ".join(kept),
        rejected=tuple(rejected),
        unknown_count=len(rejected),
    )


def reconcile_doc_type(*, existing: str, proposed: str) -> tuple[str, bool]:
    """Keep a stored label when a fresh enrichment disagrees.

    Reclassification is a deliberate event, not a side effect of re-indexing.
    Returns ``(final_value, disagreed)``.
    """
    existing_norm = canonicalize_doc_type(existing or "")
    proposed_norm = canonicalize_doc_type(proposed or "")
    if not existing_norm:
        return proposed_norm, False
    if not proposed_norm:
        return existing_norm, False
    if existing_norm == proposed_norm:
        return existing_norm, False
    return existing_norm, True


def enrichment_input_hash(
    *,
    text: str,
    title: str,
    source_type: str,
    context_text: str = "",
    max_input_chars: int = 4000,
) -> str:
    """Hash the exact inputs that feed the enrichment LLM call.

    Uses the same head+tail truncation as ``enrich_document`` so a cache hit
    means the model would have seen identical bytes.
    """
    body = text or ""
    if len(body) > max_input_chars:
        half = max_input_chars // 2
        body = body[:half] + "\n\n[...]\n\n" + body[-half:]
    payload = "\0".join(
        (
            (title or "").strip(),
            (source_type or "").strip(),
            body,
            (context_text or "").strip(),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sync_doc_type_taxonomy(store: "TaxonomyStore | None") -> dict[str, int]:
    """Idempotently seed the shipped doc_type vocabulary into the taxonomy store."""
    if store is None:
        return {"discovered": 0, "added": 0, "existing": 0}
    added = 0
    existing = 0
    for name, description, aliases in DEFAULT_DOC_TYPE_ENTRIES:
        entry_id = f"doc_type:{name}"
        try:
            current = store.get(entry_id)
        except Exception as exc:
            logger.warning("doc_type taxonomy get failed for %s: %s", entry_id, exc)
            continue
        if current is not None:
            existing += 1
            continue
        try:
            store.add(
                "doc_type",
                name,
                description,
                aliases=aliases,
                ai_managed=0,
                created_by="indexer",
            )
            added += 1
        except Exception as exc:
            logger.warning("doc_type taxonomy add failed for %s: %s", entry_id, exc)
    return {
        "discovered": len(DEFAULT_DOC_TYPE_ENTRIES),
        "added": added,
        "existing": existing,
    }


def is_vocab_member(label: str, alias_map: Mapping[str, str] | None = None) -> bool:
    """Whether an atomic label is in the vocabulary (or the unclassified sentinel)."""
    mapping = alias_map if alias_map is not None else default_alias_map()
    folded = canonicalize_doc_type(label)
    if not folded:
        return False
    if folded == UNCLASSIFIED_DOC_TYPE:
        return True
    return folded in mapping
