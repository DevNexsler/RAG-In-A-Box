import json

from core.enrichment_postprocess import repair_enrichment


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
    assert "rental inquiry" in doc_types


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
