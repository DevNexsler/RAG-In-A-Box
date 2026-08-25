import json

from core.enrichment_postprocess import canonicalize_doc_type, repair_enrichment


def test_importance_raises_actionable_payment_documents_above_default():
    enrichment = {
        "enr_importance": "0.5",
        "enr_doc_type": "message",
        "enr_key_facts": json.dumps(["TenantCloud sent a notice."]),
    }

    repaired = repair_enrichment(
        enrichment,
        text=(
            "TenantCloud payment failed for rent at Unit E. "
            "Balance due is $1,250 and payment is overdue."
        ),
        title="TenantCloud failed rent payment",
        source_type="email",
        enabled=True,
    )

    assert float(repaired["enr_importance"]) >= 0.8


def test_doc_type_adds_stable_classification_without_dropping_model_values():
    enrichment = {
        "enr_importance": "0.5",
        "enr_doc_type": "email",
        "enr_key_facts": "[]",
    }

    repaired = repair_enrichment(
        enrichment,
        text="Zillow Rental Manager sent a new renter inquiry requesting a tour.",
        title="New message from Zillow Rental Manager",
        source_type="email",
        enabled=True,
    )

    doc_types = {item.strip() for item in repaired["enr_doc_type"].split(",")}
    assert "email" in doc_types
    assert "rental_inquiry" in doc_types


def test_key_facts_drop_unsupported_generic_items_and_add_source_evidence():
    enrichment = {
        "enr_importance": "0.5",
        "enr_doc_type": "message",
        "enr_key_facts": json.dumps(
            [
                "The document contains important information.",
                "Tenant must sign the renewal by 2026-03-01.",
                "A pet deposit was invented.",
            ]
        ),
    }

    repaired = repair_enrichment(
        enrichment,
        text=(
            "Lease renewal reminder: tenant must sign the renewal by 2026-03-01. "
            "Monthly rent remains $1,250."
        ),
        title="Lease renewal reminder",
        source_type="email",
        enabled=True,
    )

    facts = json.loads(repaired["enr_key_facts"])
    joined = " ".join(facts).lower()
    assert "important information" not in joined
    assert "pet deposit" not in joined
    assert "tenant must sign" in joined
    assert "2026-03-01" in joined


def test_repair_disabled_returns_equal_copy():
    enrichment = {
        "enr_importance": "0.5",
        "enr_doc_type": "message",
        "enr_key_facts": "[]",
    }

    repaired = repair_enrichment(
        enrichment,
        text="Rent payment failed.",
        title="Payment failed",
        source_type="email",
        enabled=False,
    )

    assert repaired == enrichment
    assert repaired is not enrichment


def test_enabled_rules_can_limit_repair_to_importance_only():
    enrichment = {
        "enr_importance": "0.5",
        "enr_doc_type": "message",
        "enr_key_facts": json.dumps(["The document contains important information."]),
    }

    repaired = repair_enrichment(
        enrichment,
        text="TenantCloud rent payment failed. Balance due is $1,250 and overdue.",
        title="TenantCloud failed rent payment",
        source_type="email",
        enabled=True,
        enabled_rules=("importance",),
    )

    assert float(repaired["enr_importance"]) >= 0.8
    assert repaired["enr_doc_type"] == "message"
    assert repaired["enr_key_facts"] == enrichment["enr_key_facts"]


def test_doc_type_folds_word_segmentation_variants_to_one_spelling():
    """#1330: a compound spelled as one word must fold to the canonical segmentation.

    #1251 made the separators canonical; it cannot reconcile a model that writes
    ``followup`` where another run writes ``follow_up``, because those differ in
    word count, not in punctuation. Every compound in the vocabulary folds to its
    multi-word spelling wherever it appears in a label.
    """
    assert canonicalize_doc_type("followup") == "follow_up"
    assert canonicalize_doc_type("paystub") == "pay_stub"
    assert canonicalize_doc_type("healthcheck") == "health_check"
    assert canonicalize_doc_type("w9") == "w_9"
    assert canonicalize_doc_type("section8") == "section_8"


def test_doc_type_segmentation_applies_to_qualified_compounds():
    """The fold is on the compound, not on the whole label.

    13 of the 17 production clusters are a prefix in front of one compound
    (``leasing_followup``, ``prospect_followup``, ...), so matching whole labels
    would need a new entry per prefix and would still miss the next one.
    """
    assert canonicalize_doc_type("leasing_followup") == "leasing_follow_up"
    assert canonicalize_doc_type("Rental Application FollowUp") == "rental_application_follow_up"
    assert canonicalize_doc_type("followup_email") == "follow_up_email"
    # A prefix the corpus has never carried folds the same way.
    assert canonicalize_doc_type("vendor-followup") == "vendor_follow_up"


def test_doc_type_segmentation_is_idempotent_and_leaves_other_labels_alone():
    assert canonicalize_doc_type("follow_up") == "follow_up"
    assert canonicalize_doc_type("follow_up, followup") == "follow_up"
    assert canonicalize_doc_type("rental_inquiry, mp3") == "rental_inquiry, mp3"
