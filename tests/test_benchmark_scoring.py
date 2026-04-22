import json

from core.benchmarking.scoring import score_case, score_failed_case


def test_score_folder_accepts_configured_alternate():
    gold = {
        "canonical": {"suggested_folder": "2-Housing/Applications"},
        "alternates": {"suggested_folder": ["2-Housing/Rentals"]},
    }
    pred = {"enr_suggested_folder": "2-Housing/Rentals"}

    assert score_case(pred, gold).field_scores["suggested_folder"] == 1.0


def test_score_folder_parent_branch_gets_partial_credit():
    gold = {
        "canonical": {"suggested_folder": "2-Housing/Applications"},
        "alternates": {"suggested_folder": []},
    }
    pred = {"enr_suggested_folder": "2-Housing"}

    assert score_case(pred, gold).field_scores["suggested_folder"] == 0.5


def test_score_importance_penalizes_numeric_distance():
    gold = {
        "canonical": {"importance": "0.8"},
        "alternates": {},
    }
    pred = {"enr_importance": "0.6"}

    assert score_case(pred, gold).field_scores["importance"] == 0.8


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


def test_score_key_facts_handles_normalized_json_array():
    gold = {
        "canonical": {"key_facts": ["Tenant requested renewal.", "Parking changed."]},
        "alternates": {},
    }
    pred = {"enr_key_facts": json.dumps(["Tenant requested renewal.", "Parking changed."])}

    assert score_case(pred, gold).field_scores["key_facts"] == 1.0


def test_parse_failure_scores_zero():
    result = score_failed_case(error="json_parse_error")

    assert result.total_score == 0.0
    assert result.reliability["parse_failed"] is True
