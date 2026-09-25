import json

import pytest

from core.enrichment_postprocess import (
    CardSuffixCorrection,
    canonicalize_doc_type,
    ground_card_suffixes,
    repair_enrichment,
)


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


def _ground_summary(summary: str, source_text: str):
    return ground_card_suffixes({"enr_summary": summary}, source_text=source_text)


@pytest.mark.parametrize(
    "phone_shaped",
    [
        "###-###-7305",
        "***-***-7305",
        "XXX.XXX.7305",
        "610-555-7305",
        "1-610-555-7305",
        "(610) 555-7305",
        "(###)###-7305",
    ],
)
def test_a_phone_shaped_tail_never_grounds_a_card_suffix(phone_shaped):
    """#2526: the Pro Xtra member ID is printed like a phone number."""
    repaired, corrections = _ground_summary(
        "Paid by Visa ending in 7305.",
        f"TOTAL $42.40 XXXXXXXXXXXX4821 VISA Member {phone_shaped} Thanks",
    )

    assert repaired["enr_summary"] == "Paid by Visa ending in 4821."
    assert corrections == [CardSuffixCorrection("enr_summary", "7305", "4821")]


@pytest.mark.parametrize(
    "masked",
    [
        "XXXXXXXXXXXX4821",
        "xxxx-xxxx-xxxx-4821",
        "**** **** **** 4821",
        "############4821",
        "Card #: *4821",
        "Visa\u00a0\u00b7\u00b7\u00b7\u00b7\u00a04821",
        "Checking (...4821)",
    ],
)
def test_a_wrong_suffix_is_rewritten_to_the_sole_masked_number(masked):
    repaired, _ = _ground_summary(
        "Paid by credit card ending in 7305.",
        f"Charged {masked}. Pro Xtra ###-###-7305",
    )

    assert repaired["enr_summary"] == "Paid by credit card ending in 4821."


@pytest.mark.parametrize(
    ("summary", "dropped"),
    [
        ("Paid by Visa ending in 7305.", "Paid by Visa."),
        ("Paid by Visa (ending in 7305) in store.", "Paid by Visa in store."),
        ("Paid by Mastercard, last four digits 7305.", "Paid by Mastercard."),
        ("Paid by debit card ****7305.", "Paid by debit card."),
    ],
)
def test_a_wrong_suffix_is_dropped_when_no_single_masked_number_exists(summary, dropped):
    repaired, corrections = _ground_summary(summary, "Visa 4821 Pro Xtra ###-###-7305")

    assert repaired["enr_summary"] == dropped
    assert corrections == [CardSuffixCorrection("enr_summary", "7305", "")]


@pytest.mark.parametrize(
    "summary",
    [
        # Not a card claim: the member ID and the store phone really end so.
        "The Pro Xtra member number ends in 7305.",
        "Paid by credit card at the store whose phone number ends in 7305.",
        "Paid by Visa. The store phone ends in 7305.",
        # A card claim whose suffix the source does print, unmasked.
        "Paid by Visa ending in 4821.",
        "Paid by Visa ending in 1111.",
    ],
)
def test_supported_and_non_card_claims_are_left_alone(summary):
    repaired, corrections = _ground_summary(
        summary, "Visa 4821 Ref 4111111111111111 Pro Xtra ###-###-7305"
    )

    assert repaired["enr_summary"] == summary
    assert corrections == []


def test_card_suffix_grounding_rewrites_list_items_in_place():
    enrichment = {
        "enr_key_facts": json.dumps(["Total is $42.40.", "Paid by Visa ending in 7305."]),
        "enr_keywords": "Home Depot, Visa ending in 7305",
        "enr_importance": "0.5",
    }

    repaired, corrections = ground_card_suffixes(
        enrichment, source_text="XXXXXXXXXXXX4821 VISA PRO XTRA ###-###-7305"
    )

    assert json.loads(repaired["enr_key_facts"]) == [
        "Total is $42.40.",
        "Paid by Visa ending in 4821.",
    ]
    assert repaired["enr_keywords"] == "Home Depot, Visa ending in 4821"
    assert repaired["enr_importance"] == "0.5"
    assert [c.field for c in corrections] == ["enr_key_facts", "enr_keywords"]
