import json

import pytest

from core.benchmarking.scoring import score_audit_case, score_case, score_failed_case


def test_score_folder_accepts_configured_alternate():
    gold = {
        "canonical": {"suggested_folder": "2-Housing/Applications"},
        "alternates": {"suggested_folder": ["2-Housing/Rentals"]},
    }
    pred = {"enr_suggested_folder": "2-Housing/Rentals"}

    assert score_case(pred, gold).field_scores["suggested_folder"] == 1.0


def test_score_folder_normalizes_equivalent_trailing_slash():
    gold = {
        "canonical": {"suggested_folder": "2-Housing/Applications"},
        "alternates": {"suggested_folder": []},
    }
    pred = {"enr_suggested_folder": "2-Housing/Applications/"}

    assert score_case(pred, gold).field_scores["suggested_folder"] == 1.0


def test_score_folder_parent_branch_gets_partial_credit():
    gold = {
        "canonical": {"suggested_folder": "2-Housing/Applications"},
        "alternates": {"suggested_folder": []},
    }
    pred = {"enr_suggested_folder": "2-Housing"}

    assert score_case(pred, gold).field_scores["suggested_folder"] == 0.5


def test_score_folder_deeper_child_does_not_get_partial_credit():
    gold = {
        "canonical": {"suggested_folder": "Housing/Leases"},
        "alternates": {"suggested_folder": []},
    }
    pred = {"enr_suggested_folder": "Housing/Leases/2026"}

    assert score_case(pred, gold).field_scores["suggested_folder"] == 0.0


def test_score_importance_penalizes_numeric_distance():
    gold = {
        "canonical": {"importance": "0.8"},
        "alternates": {},
    }
    pred = {"enr_importance": "0.6"}

    assert score_case(pred, gold).field_scores["importance"] == 0.8


def test_score_entities_dates_canonicalizes_common_formats():
    gold = {
        "canonical": {"entities_dates": ["2026-03-01"]},
        "alternates": {},
    }
    pred = {"enr_entities_dates": "03/01/2026"}

    assert score_case(pred, gold).field_scores["entities_dates"] == 1.0


def test_score_case_uses_weighted_total_across_fields():
    gold = {
        "canonical": {
            "doc_type": ["lease"],
            "topics": ["rent"],
            "keywords": ["parking addendum"],
            "suggested_tags": ["renewal"],
        },
        "alternates": {"suggested_tags": []},
    }
    pred = {
        "enr_doc_type": "lease",
        "enr_topics": "rent",
        "enr_keywords": "different",
        "enr_suggested_tags": "renewal",
    }

    result = score_case(pred, gold)

    expected = 0.18 + 0.18 + 0.0 + 0.10
    assert result.total_score == expected


def test_score_topics_allows_wording_drift():
    gold = {
        "canonical": {"topics": ["lease renewal request"]},
        "alternates": {},
    }
    pred = {"enr_topics": "renewal of lease"}

    score = score_case(pred, gold).field_scores["topics"]

    assert 0.0 < score < 1.0


def test_score_keywords_allows_wording_drift():
    gold = {
        "canonical": {"keywords": ["parking addendum update"]},
        "alternates": {},
    }
    pred = {"enr_keywords": "parking addendum revised"}

    score = score_case(pred, gold).field_scores["keywords"]

    assert 0.0 < score < 1.0


def test_score_suggested_tags_alternates_extend_allowed_tag_set():
    gold = {
        "canonical": {
            "suggested_tags": ["lease"],
        },
        "alternates": {"suggested_tags": ["housing"]},
    }
    pred = {"enr_suggested_tags": "lease, housing"}

    assert score_case(pred, gold).field_scores["suggested_tags"] == 1.0


def test_score_key_facts_handles_normalized_json_array():
    gold = {
        "canonical": {"key_facts": ["Tenant requested renewal.", "Parking changed."]},
        "alternates": {},
    }
    pred = {"enr_key_facts": json.dumps(["Tenant requested renewal.", "Parking changed."])}

    assert score_case(pred, gold).field_scores["key_facts"] == 1.0


def test_score_key_facts_awards_partial_credit_for_paraphrase_and_penalizes_extra():
    gold = {
        "canonical": {
            "key_facts": [
                "Tenant requested renewal.",
                "Parking addendum changed.",
            ]
        },
        "alternates": {},
    }
    pred = {
        "enr_key_facts": json.dumps(
            [
                "Tenant requested lease renewal.",
                "Invented pet deposit increase.",
            ]
        )
    }

    score = score_case(pred, gold).field_scores["key_facts"]
    assert 0.2 < score < 0.5


def test_parse_failure_scores_zero():
    result = score_failed_case(error="json_parse_error")

    assert result.total_score == 0.0
    assert result.reliability["parse_failed"] is True


def test_score_audit_case_returns_subscores_and_composite():
    gold = {
        "label_source": "manual_audit",
        "canonical": {
            "summary": "Sender waived fees and gave a rent contact number.",
            "doc_type": ["message"],
            "entities_people": ["Luis"],
            "entities_places": [],
            "entities_orgs": ["Pinefield Group LLC"],
            "entities_dates": [],
            "topics": ["fee waiver", "rent inquiries"],
            "keywords": ["waive fees", "contact number"],
            "key_facts": ["Fees were waived.", "Rent questions should be texted to the provided number."],
            "suggested_tags": ["customer communication", "fee waiver"],
            "suggested_folder": "Housing/Tenant Payments",
            "importance": "0.6",
        },
        "alternates": {"suggested_tags": ["rental management"], "suggested_folder": ["Finance/Receipts and Payments"]},
        "summary_rubric": {
            "coverage": ["fees were waived", "contact number"],
            "brevity": {"max_sentences": 2, "max_words": 25},
            "hallucination": ["late payment warning"],
        },
    }
    pred = {
        "enr_summary": "Fees were waived. Contact number provided.",
        "enr_doc_type": "message",
        "enr_entities_people": "Luis",
        "enr_entities_orgs": "Pinefield Group LLC",
        "enr_topics": "fee waiver, rent inquiries",
        "enr_keywords": "waive fees, contact number",
        "enr_key_facts": json.dumps(["Fees were waived.", "Rent questions should be texted to the provided number."]),
        "enr_suggested_tags": "customer communication, fee waiver",
        "enr_suggested_folder": "Housing/Tenant Payments",
        "enr_importance": "0.6",
    }

    result = score_audit_case(pred, gold)

    assert result.subscores["extraction_core"] == 1.0
    assert result.subscores["filing_taxonomy"] == 1.0
    assert result.subscores["summary_quality"] == 1.0
    assert result.total_score == 1.0


def test_score_audit_case_does_not_penalize_missing_alternate_only_tags():
    gold = {
        "label_source": "manual_audit",
        "canonical": {
            "summary": "",
            "doc_type": [],
            "entities_people": [],
            "entities_places": [],
            "entities_orgs": [],
            "entities_dates": [],
            "topics": [],
            "keywords": [],
            "key_facts": [],
            "suggested_tags": ["customer communication", "fee waiver"],
            "suggested_folder": "Housing/Tenant Payments",
            "importance": "",
        },
        "alternates": {"suggested_tags": ["rental management"], "suggested_folder": []},
        "summary_rubric": {
            "coverage": [],
            "brevity": {"max_sentences": 0, "max_words": 0},
            "hallucination": [],
        },
    }
    pred = {
        "enr_suggested_tags": "customer communication, fee waiver",
        "enr_suggested_folder": "Housing/Tenant Payments",
    }

    result = score_audit_case(pred, gold)

    assert result.subscores["filing_taxonomy"] == 1.0


def test_score_audit_case_penalizes_summary_brevity_and_hallucination():
    gold = {
        "label_source": "manual_audit",
        "canonical": {
            "summary": "Tenant reported blocked fire escape and rent escrow.",
            "doc_type": ["tenant communication"],
            "entities_people": [],
            "entities_places": [],
            "entities_orgs": [],
            "entities_dates": [],
            "topics": ["maintenance issues", "rent escrow"],
            "keywords": ["fire escape", "rent escrow"],
            "key_facts": ["Fire escape is blocked.", "Tenant plans rent escrow."],
            "suggested_tags": ["maintenance", "fire safety"],
            "suggested_folder": "Housing/Maintenance Issues",
            "importance": "0.9",
        },
        "alternates": {"suggested_tags": [], "suggested_folder": []},
        "summary_rubric": {
            "coverage": ["blocked fire escape", "rent escrow"],
            "brevity": {"max_sentences": 1, "max_words": 10},
            "hallucination": ["landlord already repaired unit"],
        },
    }
    pred = {
        "enr_summary": (
            "The tenant says the blocked fire escape still has not been fixed and plans rent escrow. "
            "The landlord already repaired most of the unit."
        ),
    }

    result = score_audit_case(pred, gold)

    assert result.subscores["summary_quality"] < 1.0
    assert result.subscores["summary_quality"] > 0.0
    assert result.total_score < 1.0


def test_score_audit_case_rejects_missing_summary_rubric():
    gold = {
        "label_source": "manual_audit",
        "canonical": {
            "summary": "",
            "doc_type": [],
            "entities_people": [],
            "entities_places": [],
            "entities_orgs": [],
            "entities_dates": [],
            "topics": [],
            "keywords": [],
            "key_facts": [],
            "suggested_tags": [],
            "suggested_folder": "",
            "importance": "",
        },
        "alternates": {"suggested_tags": [], "suggested_folder": []},
    }

    with pytest.raises(ValueError, match="summary_rubric"):
        score_audit_case({}, gold)
