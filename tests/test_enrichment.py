"""Tests for doc_enrichment.py — LLM document enrichment parsing and normalization.

Unit tests mock the LLM generator.  Integration test uses the real MiniMax M2.5 via OpenRouter.
"""

import json
import logging
import re
from unittest.mock import MagicMock

import pytest

from doc_enrichment import (
    ENRICHMENT_FIELDS,
    _extract_json,
    _normalize_list,
    _normalize_enrichment,
    empty_enrichment,
    failed_enrichment,
    enrich_document,
    parse_enrichment_response,
)


def _complete_enrichment_json(**overrides) -> str:
    payload = {
        "summary": "test",
        "doc_type": ["note"],
        "entities_people": [],
        "entities_places": [],
        "entities_orgs": [],
        "entities_dates": [],
        "topics": [],
        "keywords": [],
        "key_facts": [],
        "suggested_tags": [],
        "suggested_folder": "",
        "importance": 0.5,
    }
    payload.update(overrides)
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# Unit tests — parsing, normalization, and error handling (no LLM needed)
# ---------------------------------------------------------------------------


class TestEmptyEnrichment:
    def test_has_all_fields(self):
        result = empty_enrichment()
        for field in ENRICHMENT_FIELDS:
            assert field in result
            assert result[field] == ""

    def test_returns_new_dict_each_time(self):
        a = empty_enrichment()
        b = empty_enrichment()
        a["enr_summary"] = "modified"
        assert b["enr_summary"] == ""


class TestExtractJson:
    def test_plain_json(self):
        text = '{"summary": "A document", "doc_type": ["report"]}'
        result = _extract_json(text)
        assert result["summary"] == "A document"
        assert result["doc_type"] == ["report"]

    def test_json_with_markdown_fences(self):
        text = '```json\n{"summary": "A doc", "doc_type": ["report"]}\n```'
        result = _extract_json(text)
        assert result["summary"] == "A doc"

    def test_json_with_plain_fences(self):
        text = '```\n{"summary": "A doc"}\n```'
        result = _extract_json(text)
        assert result["summary"] == "A doc"

    def test_json_with_leading_whitespace(self):
        text = '\n  \n{"summary": "A doc"}\n'
        result = _extract_json(text)
        assert result["summary"] == "A doc"

    def test_json_with_thinking_tags(self):
        text = '<think>Let me analyze this document...</think>\n{"summary": "A doc"}'
        result = _extract_json(text)
        assert result["summary"] == "A doc"

    def test_json_with_trailing_text(self):
        text = '{"summary": "A doc", "doc_type": ["note"]} Here is my analysis...'
        result = _extract_json(text)
        assert result["summary"] == "A doc"

    def test_truncated_json_salvaged(self):
        """Token-limit truncation should salvage completed fields."""
        text = '{"summary": "A tax doc", "doc_type": ["tax", "finan'
        result = _extract_json(text)
        assert result["summary"] == "A tax doc"
        assert "tax" in result["doc_type"]

    def test_truncated_json_mid_array(self):
        text = '{"summary": "Hello", "topics": ["ai", "ml", "deep'
        result = _extract_json(text)
        assert result["summary"] == "Hello"
        assert "ai" in result["topics"]

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json("not json at all")


class TestNormalizeList:
    def test_list_to_csv(self):
        assert _normalize_list(["a", "b", "c"]) == "a, b, c"

    def test_empty_list(self):
        assert _normalize_list([]) == ""

    def test_string_passthrough(self):
        assert _normalize_list("already a string") == "already a string"

    def test_list_strips_whitespace(self):
        assert _normalize_list(["  a  ", "  b "]) == "a, b"

    def test_filters_empty_items(self):
        assert _normalize_list(["a", "", "  ", "b"]) == "a, b"


class TestNormalizeEnrichment:
    def test_full_valid_response(self):
        raw = {
            "summary": "A geotechnical report.",
            "doc_type": ["report", "engineering"],
            "entities_people": ["John Smith"],
            "entities_places": ["12100 Ganesh Lane"],
            "entities_orgs": ["ABC Engineering"],
            "entities_dates": ["2024-03-15"],
            "topics": ["soil analysis", "foundation design"],
            "keywords": ["geotechnical", "boring logs"],
            "key_facts": ["Foundation type: spread footings", "Bearing capacity: 2500 psf"],
        }
        result = _normalize_enrichment(raw)
        assert result["enr_summary"] == "A geotechnical report."
        assert result["enr_doc_type"] == "report, engineering"
        assert result["enr_entities_people"] == "John Smith"
        assert result["enr_entities_places"] == "12100 Ganesh Lane"
        assert result["enr_entities_orgs"] == "ABC Engineering"
        assert result["enr_entities_dates"] == "2024-03-15"
        assert result["enr_topics"] == "soil analysis, foundation design"
        assert result["enr_keywords"] == "geotechnical, boring logs"
        facts = json.loads(result["enr_key_facts"])
        assert "Foundation type: spread footings" in facts

    def test_doc_type_folds_separator_and_case_variants(self):
        """#1251: enr_doc_type is a filter key, so its spelling must be canonical."""
        raw = {
            "doc_type": [
                "Property Showing",
                "property-showing",
                "property_showing",
                "Follow-Up",
            ]
        }
        result = _normalize_enrichment(raw)
        assert result["enr_doc_type"] == "property_showing, follow_up"

    def test_missing_fields_default_to_empty(self):
        result = _normalize_enrichment({"summary": "Hello"})
        assert result["enr_summary"] == "Hello"
        assert result["enr_doc_type"] == ""
        assert result["enr_topics"] == ""
        assert result["enr_key_facts"] == ""

    def test_none_values_become_empty(self):
        raw = {f: None for f in ENRICHMENT_FIELDS}
        result = _normalize_enrichment(raw)
        for f in ENRICHMENT_FIELDS:
            assert result[f] == ""

    def test_key_facts_string_passthrough(self):
        raw = {"key_facts": '["already serialized"]'}
        result = _normalize_enrichment(raw)
        assert result["enr_key_facts"] == '["already serialized"]'


def test_parse_enrichment_response_normalizes_valid_json():
    raw = '{"summary":"x","doc_type":["memo"],"entities_people":[],"entities_places":[],"entities_orgs":[],"entities_dates":[],"topics":["ops"],"keywords":["lease"],"key_facts":["rent due"],"suggested_tags":["housing"],"suggested_folder":"2-Housing","importance":0.7}'
    parsed = parse_enrichment_response(raw)
    assert parsed["enr_doc_type"] == "memo"
    assert parsed["enr_topics"] == "ops"
    assert parsed["enr_importance"] == "0.7"


def test_parse_enrichment_response_handles_fenced_json():
    raw = '```json\n{"summary":"x","doc_type":["memo"],"topics":["ops"],"importance":0.7}\n```'
    parsed = parse_enrichment_response(raw)
    assert parsed["enr_doc_type"] == "memo"
    assert parsed["enr_topics"] == "ops"
    assert parsed["enr_importance"] == "0.7"


def test_parse_enrichment_response_defaults_missing_optional_context_fields():
    result = parse_enrichment_response('{"summary":"ok","doc_type":["email"]}')

    assert result["enr_summary"] == "ok"
    assert result["enr_doc_type"] == "email"
    assert result["enr_entities_people"] == ""
    assert result["enr_context_warning"] == ""


def test_parse_context_enrichment_fields():
    parsed = parse_enrichment_response(
        """
        {
          "summary": "Photo of vehicle parts.",
          "doc_type": ["image"],
          "entities_people": [],
          "entities_places": [],
          "entities_orgs": [],
          "entities_dates": [],
          "topics": ["vehicle"],
          "keywords": ["car parts"],
          "key_facts": [],
          "suggested_tags": ["maintenance"],
          "suggested_folder": "Housing/Maintenance",
          "importance": 0.5,
          "atomic_entities_places": [],
          "context_entities_places": ["54 S Broad Main Unit E"],
          "context_topics": ["basement storage"],
          "context_key_facts": ["Nearby message says the photos are from Unit E."],
          "context_relationship": "batch_label",
          "context_confidence": "high",
          "context_source_message_ids": ["4434"],
          "context_warning": ""
        }
        """
    )

    assert parsed["enr_context_entities_places"] == "54 S Broad Main Unit E"
    assert parsed["enr_context_relationship"] == "batch_label"
    assert parsed["enr_context_confidence"] == "high"
    assert parsed["enr_context_source_message_ids"] == "4434"


def test_parse_context_placeholder_values_as_empty():
    parsed = parse_enrichment_response(
        json.dumps(
            {
                "summary": "Photo of a receipt.",
                "doc_type": ["image"],
                "context_entities_places": ["places inferred from relevant nearby context"],
                "context_topics": ["topics inferred from relevant nearby context"],
                "context_source_message_ids": [],
            }
        )
    )

    assert parsed["enr_context_entities_places"] == ""
    assert parsed["enr_context_topics"] == ""


# A store's "items were delivered" email and the earlier same-order message that
# carried the payment summary, in the layout production indexes (#2562). Only
# the earlier message has the totals and the card; every value is sanitized.
DELIVERY_EMAIL = (
    "Placed March 3, 2026\nOrder # 300000000000000001\nInvoice # 12345\n"
    "Your Item(s) Were Delivered\nDelivered Tuesday, Mar 3, 2026\n"
    "Address: 100 Example Lane, Anytown, PA 18000\n"
    "Delivered Items: Tile Adhesive (25 Pound(s)) QTY1\n"
    "Item #: 1000001|Model #: 2000002 Unit Price: $42.50|Subtotal: $42.50\n"
    "Need to make a return? Start a Return/Replacement Online"
)
DELIVERY_CONTEXT = (
    "BEFORE MESSAGES\n"
    "[BEFORE 2026-03-03T09:12:00+00:00 message_id=1001 "
    "source_message_id=<prepared@example.test> sender=orders@example.test] "
    "Your order 300000000000000001 is being prepared. Subtotal $75.59 "
    "Savings -$18.00 Tax $3.61 Total $61.20 Loyalty Discount applied. "
    "Paid with Visa ending in 4242."
)
DELIVERY_PRIMARY_FACTS = [
    "Delivered Tuesday, Mar 3, 2026 to 100 Example Lane, Anytown, PA 18000.",
    "Invoice #12345; order #300000000000000001.",
    "Return/replacement and billing/help links are provided.",
]
# What the model wrote from the earlier message, including a fact that mixes a
# primary amount with the context-only card.
DELIVERY_CONTEXT_FACTS = [
    "Payment summary shows subtotal $75.59, savings $18.00, tax $3.61, and total $61.20.",
    "Loyalty discount applied; card ending in 4242 was used.",
    "Unit price $42.50 was charged to the card ending in 4242.",
]

LOWES_DELIVERY_EMAIL = (
    "Your Item(s) Were Delivered\nDelivered Tuesday, Mar 3, 2026\n"
    "Address: 100 Example Lane, Anytown, PA 18000\n"
    "Delivered Items: Tile Adhesive QTY1\n"
    "Order # 300000000000000001"
)
LOWES_DELIVERY_CONTEXT = (
    "BEFORE MESSAGES\n"
    "[BEFORE 2026-03-03T09:12:00+00:00 message_id=1001 "
    "source_message_id=<prepared@example.test>] "
    "Your order 300000000000000001 is being prepared. "
    "Military discount applied. Card ending in 5531."
)


class TestContextOnlyFactsLeavePrimaryFields:
    """Facts only a nearby message supports are stored as context facts (#2562)."""

    def _enrich(self, response: dict, text: str, context_text: str) -> dict:
        gen = MagicMock()
        gen.generate.return_value = json.dumps(response)
        return enrich_document(text, "Your items were delivered", "pg_message", gen,
                               context_text=context_text)

    def test_context_payment_details_move_out_of_primary_facts_and_keywords(self):
        response = {
            "summary": "Delivery confirmation for order 300000000000000001.",
            "doc_type": ["delivery confirmation"],
            "keywords": [
                "Order #300000000000000001",
                "Invoice #12345",
                "Delivered Tuesday, Mar 3, 2026",
                "Subtotal $42.50",
                "Total $61.20",
                "Total Tax $3.61",
                "Card ending 4242",
                "Loyalty Discount",
            ],
            "key_facts": [
                DELIVERY_PRIMARY_FACTS[0],
                DELIVERY_PRIMARY_FACTS[1],
                *DELIVERY_CONTEXT_FACTS,
                DELIVERY_PRIMARY_FACTS[2],
            ],
            "context_key_facts": ["The earlier message shows the same order being prepared."],
            "context_confidence": "high",
            "context_relationship": "Earlier message about the same order.",
            "context_source_message_ids": ["<prepared@example.test>"],
        }

        result = self._enrich(response, DELIVERY_EMAIL, DELIVERY_CONTEXT)

        assert json.loads(result["enr_key_facts"]) == DELIVERY_PRIMARY_FACTS
        assert json.loads(result["enr_context_key_facts"]) == [
            "The earlier message shows the same order being prepared.",
            *DELIVERY_CONTEXT_FACTS,
        ]
        # Keywords keep their stored spelling; the moved ones are already
        # carried by the moved facts, so they are not repeated.
        assert result["enr_keywords"] == (
            "Order #300000000000000001, Invoice #12345, Delivered Tuesday, Mar 3, 2026, "
            "Subtotal $42.50"
        )
        assert "Loyalty Discount" not in result["enr_keywords"]
        # The model's own provenance is kept as written.
        assert result["enr_context_confidence"] == "high"
        assert result["enr_context_source_message_ids"] == "<prepared@example.test>"
        assert result["enr_context_warning"] == ""

    def test_facts_the_primary_text_supports_are_unchanged(self):
        # The context repeats every number, and the model reformatted each one:
        # thousands separator and cents, phone punctuation, a masked card, an
        # ISO date that only the context's timestamp header spells that way, and
        # a total it added up from the receipt's own line amounts.
        text = (
            "Receipt. Placed March 3, 2026. Order Total $1,041.00 "
            "VISA XXXXXXXXXXXX4242. Questions? Call (804) 555-0142. "
            "Payment received: Rent - $1,250, Water fee - $45."
        )
        context_text = (
            "BEFORE MESSAGES\n"
            "[BEFORE 2026-03-03T09:12:00+00:00 message_id=1001 "
            "source_message_id=<confirm@example.test>] Order confirmed. "
            "Total $1041.00 on Visa ending in 4242. Store phone 804-555-0142. "
            "Transfer amount $1,295.00."
        )
        response = {
            "summary": "Receipt for an order.",
            "doc_type": ["receipt"],
            "keywords": ["Total $1041", "Visa 4242", "804-555-0142", "2026-03-03"],
            "key_facts": [
                "Total was $1041.",
                "Paid with Visa ending in 4242.",
                "Store phone is 804-555-0142.",
                "Order placed 2026-03-03.",
                "Total payment amount was $1,295.",
            ],
            "importance": 0.5,
        }

        result = self._enrich(response, text, context_text)

        assert result == parse_enrichment_response(json.dumps(response))

    def test_text_only_context_keyword_moves_without_numbers(self):
        """#2611: number-only guards left phrases like Military Discount in keywords."""
        response = {
            "summary": "Delivery confirmation for order 300000000000000001.",
            "doc_type": ["delivery confirmation"],
            "keywords": [
                "Order #300000000000000001",
                "Delivered Tuesday, Mar 3, 2026",
                "Military Discount",
            ],
            "key_facts": [
                "Delivered Tuesday, Mar 3, 2026 to 100 Example Lane, Anytown, PA 18000.",
                "Military discount applied; card ending in 5531.",
            ],
            "context_key_facts": [
                "Earlier message shows military discount and card ending in 5531.",
            ],
            "context_confidence": "high",
            "context_relationship": "Earlier message about the same order.",
            "context_source_message_ids": ["<prepared@example.test>"],
        }

        result = self._enrich(response, LOWES_DELIVERY_EMAIL, LOWES_DELIVERY_CONTEXT)

        assert json.loads(result["enr_key_facts"]) == [
            "Delivered Tuesday, Mar 3, 2026 to 100 Example Lane, Anytown, PA 18000.",
        ]
        assert "Military Discount" not in result["enr_keywords"]
        context_facts = json.loads(result["enr_context_key_facts"])
        assert "Military discount applied; card ending in 5531." in context_facts
        assert any("military discount" in fact.lower() for fact in context_facts)

    def test_context_only_keyword_is_kept_as_context_fact_with_provenance(self):
        context_text = (
            "BEFORE MESSAGES\n"
            "[BEFORE source_message_id=<confirm@example.test>] Order confirmation. "
            "Order total $88.14.\n"
            "AFTER MESSAGES\n"
            "[AFTER source_message_id=<survey@example.test>] Tell us how we did."
        )
        response = {
            "summary": "Order is ready for pickup.",
            "doc_type": ["pickup notification"],
            "keywords": ["ready for pickup", "order total $88.14"],
            "key_facts": ["The order is ready for pickup."],
            "importance": 0.5,
        }

        result = self._enrich(response, "Your order is ready for pickup.", context_text)

        assert result["enr_keywords"] == "ready for pickup"
        assert json.loads(result["enr_key_facts"]) == ["The order is ready for pickup."]
        assert json.loads(result["enr_context_key_facts"]) == ["order total $88.14"]
        assert result["enr_context_source_message_ids"] == "<confirm@example.test>"
        assert result["enr_context_confidence"] == "medium"
        assert "omitted structured context fields" in result["enr_context_warning"]


def test_context_provenance_eval_sample_fixtures():
    """Sanitized production-shaped comm fixtures: fewer context facts in primary fields."""
    fixtures = [
        (
            DELIVERY_EMAIL,
            DELIVERY_CONTEXT,
            {
                "summary": "Delivery confirmation.",
                "doc_type": ["delivery confirmation"],
                "keywords": ["Order #300000000000000001", "Card ending 4242"],
                "key_facts": [
                    DELIVERY_PRIMARY_FACTS[0],
                    DELIVERY_CONTEXT_FACTS[1],
                ],
            },
            1,
        ),
        (
            LOWES_DELIVERY_EMAIL,
            LOWES_DELIVERY_CONTEXT,
            {
                "summary": "Delivery confirmation.",
                "doc_type": ["delivery confirmation"],
                "keywords": ["Order #300000000000000001", "Military Discount"],
                "key_facts": [
                    "Delivered Tuesday, Mar 3, 2026 to 100 Example Lane, Anytown, PA 18000.",
                    "Military discount applied; card ending in 5531.",
                ],
            },
            1,
        ),
        (
            "Your order is ready for pickup.",
            (
                "BEFORE MESSAGES\n"
                "[BEFORE source_message_id=<confirm@example.test>] "
                "Order total $88.14."
            ),
            {
                "summary": "Pickup notice.",
                "doc_type": ["pickup notification"],
                "keywords": ["ready for pickup", "order total $88.14"],
                "key_facts": ["The order is ready for pickup."],
            },
            1,
        ),
    ]

    context_leaks_removed = 0
    primary_facts_preserved = 0

    for text, context_text, response, expected_primary in fixtures:
        gen = MagicMock()
        gen.generate.return_value = json.dumps(response)
        result = enrich_document(text, "fixture", "pg_message", gen, context_text=context_text)

        primary_facts = json.loads(result["enr_key_facts"] or "[]")
        assert len(primary_facts) == expected_primary
        primary_facts_preserved += expected_primary

        for keyword in response.get("keywords", []):
            if keyword.lower() not in text.lower() and keyword.lower() in context_text.lower():
                assert keyword not in (result["enr_keywords"] or "")
                context_leaks_removed += 1

        for fact in response.get("key_facts", [])[expected_primary:]:
            assert fact not in primary_facts

    assert context_leaks_removed >= 2
    assert primary_facts_preserved == 3


class TestEnrichDocument:
    """Test enrich_document with mocked LLM generator."""

    def _make_generator(self, response: str) -> MagicMock:
        try:
            overrides = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            pass
        else:
            if isinstance(overrides, dict):
                response = _complete_enrichment_json(**overrides)
        gen = MagicMock()
        gen.generate.return_value = response
        return gen

    def test_successful_enrichment(self):
        llm_response = json.dumps({
            "summary": "Tax return filing for 2022.",
            "doc_type": ["tax", "financial"],
            "entities_people": ["John Doe"],
            "entities_places": ["Maryland"],
            "entities_orgs": ["IRS"],
            "entities_dates": ["2022-04-15"],
            "topics": ["tax filing", "deductions"],
            "keywords": ["Form 1040", "W-2"],
            "key_facts": ["Total income: $85,000"],
            "suggested_tags": ["finance", "tax"],
            "suggested_folder": "Financial/",
        })
        gen = self._make_generator(llm_response)
        result = enrich_document("Some tax document text...", "TaxReturn.pdf", "pdf", gen)

        assert result["enr_summary"] == "Tax return filing for 2022."
        assert "tax" in result["enr_doc_type"]
        assert result["enr_entities_people"] == "John Doe"
        assert "finance" in result["enr_suggested_tags"]
        assert result["enr_suggested_folder"] == "Financial/"
        gen.generate.assert_called_once()

    def test_new_rows_carry_no_separator_variant_clusters(self):
        """#1251 acceptance: separator/case variants leave no cluster behind.

        The ticket's index-wide scan buckets labels by
        ``re.sub(r"[-_ ]", "", s.lower())`` and reports a cluster whenever one
        bucket holds more than one spelling. Enriching one concept under every
        separator and case spelling the model has been seen to emit must
        therefore write exactly one value per bucket. (Word-segmentation
        differences such as ``follow_up`` vs ``followup`` land in the same
        bucket but are a different defect — see the ticket.)
        """
        spellings = [
            "Property Showing",
            "property-showing",
            "property_showing",
            "rental inquiry",
            "Rental-Inquiry",
            "rental_inquiry",
        ]
        written = set()
        for spelling in spellings:
            gen = self._make_generator(
                json.dumps({"summary": "A viewing was booked.", "doc_type": [spelling]})
            )
            result = enrich_document("Viewing booked for the unit.", "Viewing", "email", gen)
            written.add(result["enr_doc_type"])

        clusters: dict[str, set[str]] = {}
        for label in written:
            clusters.setdefault(re.sub(r"[-_ ]", "", label.lower()), set()).add(label)

        assert [bucket for bucket, labels in clusters.items() if len(labels) > 1] == []
        assert written == {"property_showing", "rental_inquiry"}

    def test_new_rows_carry_no_word_segmentation_clusters(self):
        """#1330 acceptance: the index-wide scan reports 0 clusters of any kind.

        These are the exact spelling pairs the 2026-08-19 read-only scan of the
        production index still reported after #1251's separator fold — the 17
        residual clusters over 588 label-doc instances. Enriching both spellings
        of each pair must leave one value per scan bucket, i.e. no cluster of
        any kind, separator or word-segmentation.
        """
        residual_clusters = [
            ("follow_up", "followup"),
            ("leasing_follow_up", "leasing_followup"),
            ("pay_stub", "paystub"),
            ("prospect_follow_up", "prospect_followup"),
            ("maintenance_followup", "maintenance_follow_up"),
            ("rental_application_follow_up", "rental_application_followup"),
            ("lead_follow_up", "lead_followup"),
            ("application_follow_up", "application_followup"),
            ("rental_follow_up", "rental_followup"),
            ("showing_follow_up", "showing_followup"),
            ("payment_follow_up", "payment_followup"),
            ("property_showing_follow_up", "property_showing_followup"),
            ("w9", "w_9"),
            ("rental_lead_follow_up", "rental_lead_followup"),
            ("rental_inquiry_followup", "rental_inquiry_follow_up"),
            ("health_check", "healthcheck"),
            ("section_8", "section8"),
        ]

        written: set[str] = set()
        for spelling in (s for pair in residual_clusters for s in pair):
            gen = self._make_generator(
                json.dumps({"summary": "A lead was contacted.", "doc_type": [spelling]})
            )
            result = enrich_document("Contacted the lead again.", "Lead", "email", gen)
            written.add(result["enr_doc_type"])

        clusters: dict[str, set[str]] = {}
        for label in written:
            clusters.setdefault(re.sub(r"[-_ ]", "", label.lower()), set()).add(label)

        assert [sorted(labels) for labels in clusters.values() if len(labels) > 1] == []
        assert len(written) == len(residual_clusters)

    def test_postprocess_flag_repairs_importance(self):
        llm_response = json.dumps({
            "summary": "TenantCloud sent a failed payment notice.",
            "doc_type": ["message"],
            "entities_people": [],
            "entities_places": [],
            "entities_orgs": ["TenantCloud"],
            "entities_dates": [],
            "topics": ["rent"],
            "keywords": ["payment"],
            "key_facts": ["TenantCloud sent a notice."],
            "suggested_tags": ["housing"],
            "suggested_folder": "Housing/Tenant Payments",
            "importance": 0.5,
        })
        gen = self._make_generator(llm_response)

        result = enrich_document(
            "TenantCloud rent payment failed. Balance due is $1,250 and overdue.",
            "TenantCloud failed rent payment",
            "email",
            gen,
            postprocess_enrichment=True,
        )

        assert float(result["enr_importance"]) >= 0.8

    def test_empty_text_returns_empty(self):
        gen = self._make_generator("")
        result = enrich_document("", "empty.md", "md", gen)
        assert result == empty_enrichment()
        gen.generate.assert_not_called()

    def test_malformed_json_returns_failed(self):
        gen = self._make_generator("This is not valid JSON at all!")
        result = enrich_document("Some text", "doc.md", "md", gen)
        assert result["_enrichment_failed"]
        assert "json_parse_error" in result["_enrichment_failed"]
        # All enrichment fields should still be present (as empty strings)
        for field in ENRICHMENT_FIELDS:
            assert field in result

    def test_salvaged_response_missing_required_fields_is_failed_without_raw_log(
        self,
        caplog,
    ):
        private_marker = "private-customer-response-marker"
        gen = self._make_generator(
            '{"context_entities_people":["'
            + private_marker
            + '"],"context_entities_orgs":['
        )
        taxonomy = MagicMock()
        taxonomy.format_for_prompt.return_value = ""

        with caplog.at_level(logging.DEBUG):
            result = enrich_document(
                "A real email body with substantive property-management content.",
                "mail.eml",
                "email",
                gen,
                taxonomy_store=taxonomy,
            )

        assert result["_enrichment_failed"].startswith(
            "structured_output_contract_violation:"
        )
        assert result["_enrichment_transient"] is False
        taxonomy.increment_usage.assert_not_called()
        assert private_marker not in caplog.text

    def test_exception_in_generate_returns_failed(self):
        gen = MagicMock()
        gen.generate.side_effect = RuntimeError("model unavailable")
        result = enrich_document("Some text", "doc.md", "md", gen)
        assert result["_enrichment_failed"]
        assert "RuntimeError" in result["_enrichment_failed"]

    def test_text_truncation_short_doc(self):
        """Documents shorter than max_input_chars are passed through entirely."""
        gen = self._make_generator('{"summary": "short", "doc_type": ["note"]}')
        short_text = "hello world"
        enrich_document(short_text, "small.md", "md", gen, max_input_chars=500)
        call_args = gen.generate.call_args[0][0]
        assert "hello world" in call_args
        assert "[...]" not in call_args

    def test_text_truncation_head_tail(self):
        """Long documents use head+tail sampling with [...] separator."""
        gen = self._make_generator('{"summary": "short", "doc_type": ["note"]}')
        head = "HEAD_MARKER " + "a" * 5000
        tail = "b" * 5000 + " TAIL_MARKER"
        long_text = head + "c" * 5000 + tail
        enrich_document(long_text, "big.md", "md", gen, max_input_chars=500)
        call_args = gen.generate.call_args[0][0]
        # Should contain head content, tail content, and the separator
        assert "HEAD_MARKER" in call_args
        assert "TAIL_MARKER" in call_args
        assert "[...]" in call_args
        # Should NOT contain the full middle section
        assert len(call_args) < len(long_text)

    def test_markdown_fences_in_response(self):
        response = "```json\n" + _complete_enrichment_json(
            summary="A doc", doc_type=["note"]
        ) + "\n```"
        gen = self._make_generator(response)
        result = enrich_document("Some text", "note.md", "md", gen)
        assert result["enr_summary"] == "A doc"
        assert result["enr_doc_type"] == "note"

    def test_thinking_tags_in_response(self):
        response = (
            '<think>Let me analyze...</think>\n'
            + _complete_enrichment_json(
                summary="Analyzed", doc_type=["note"], topics=["AI"]
            )
        )
        gen = self._make_generator(response)
        result = enrich_document("Some text", "doc.md", "md", gen)
        assert result["enr_summary"] == "Analyzed"
        assert result["enr_topics"] == "AI"


    def test_taxonomy_block_in_prompt(self):
        """When taxonomy_store is provided, its format_for_prompt output appears in the LLM prompt."""
        gen = self._make_generator('{"summary": "test", "doc_type": ["note"]}')
        mock_taxonomy = MagicMock()
        mock_taxonomy.format_for_prompt.return_value = "## Available Tags\n- work: Work stuff"
        mock_taxonomy.increment_usage = MagicMock()

        enrich_document("Some text", "doc.md", "md", gen, taxonomy_store=mock_taxonomy)
        call_args = gen.generate.call_args[0][0]
        assert "## Available Tags" in call_args
        assert "work: Work stuff" in call_args
        mock_taxonomy.format_for_prompt.assert_called_once_with(
            query="doc.md\nSome text",
            max_chars=96000,
        )

    def test_taxonomy_usage_writes_can_be_disabled_for_index_workers(self):
        """Concurrent index workers should read taxonomy for prompts without writing usage."""
        gen = self._make_generator(json.dumps({
            "summary": "test",
            "doc_type": ["note"],
            "suggested_tags": ["work", "urgent"],
            "suggested_folder": "Projects/Renovation",
        }))
        mock_taxonomy = MagicMock()
        mock_taxonomy.format_for_prompt.return_value = "## Available Tags\n- work: Work stuff"
        mock_taxonomy.increment_usage = MagicMock()

        result = enrich_document(
            "Some text",
            "doc.md",
            "md",
            gen,
            taxonomy_store=mock_taxonomy,
            record_taxonomy_usage=False,
        )

        assert result["enr_suggested_tags"] == "work, urgent"
        assert result["enr_suggested_folder"] == "Projects/Renovation"
        mock_taxonomy.increment_usage.assert_not_called()

    def test_no_taxonomy_no_block(self):
        """Without taxonomy_store, no taxonomy block in prompt."""
        gen = self._make_generator('{"summary": "test", "doc_type": ["note"]}')
        enrich_document("Some text", "doc.md", "md", gen, taxonomy_store=None)
        call_args = gen.generate.call_args[0][0]
        assert "## Available Tags" not in call_args

    def test_context_prompt_labels_nearby_candidates(self):
        response = json.dumps({
            "summary": "Photo context.",
            "doc_type": ["image"],
            "entities_people": [],
            "entities_places": [],
            "entities_orgs": [],
            "entities_dates": [],
            "topics": [],
            "keywords": [],
            "key_facts": [],
            "suggested_tags": [],
            "suggested_folder": "",
            "importance": 0.5,
            "context_relationship": "nearby_ambiguous",
            "context_confidence": "ambiguous",
        })
        gen = self._make_generator(response)

        enrich_document(
            "image notes",
            "photo.jpg",
            "img",
            gen,
            context_text="[before] Unit E",
        )

        prompt = gen.generate.call_args[0][0]
        assert "PRIMARY ITEM" in prompt
        assert "NEARBY SAME-CHANNEL CONTEXT CANDIDATES" in prompt
        assert "may or may not describe the primary item" in prompt

    def test_context_prompt_keeps_context_facts_out_of_primary_facts_and_keywords(self):
        """The prompt sends context facts where storage keeps them (#2611).

        Storage moves key facts and keywords that only nearby context supports
        to context_key_facts (#2562). A prompt that allows a context fact in
        key_facts or keywords once it is also in context_* contradicts that.
        """
        response = json.dumps({
            "summary": "Photo context.",
            "doc_type": ["image"],
        })
        gen = self._make_generator(response)

        enrich_document(
            "image notes",
            "photo.jpg",
            "img",
            gen,
            context_text="[BEFORE source_message_id=m1] Unit E",
        )

        prompt = gen.generate.call_args[0][0]
        assert (
            "key_facts and keywords describe the PRIMARY ITEM only. Put facts and terms taken\n"
            "from nearby context in context_key_facts only, never in key_facts or keywords."
        ) in prompt
        assert "summary describes what the PRIMARY ITEM itself says." in prompt
        shared = re.search(
            r"If you use nearby context in (.*?), you\s+MUST also fill the matching "
            r"context_\* fields",
            prompt,
            re.DOTALL,
        )
        assert shared is not None
        shared_fields = set(re.split(r",\s*(?:or\s+)?", shared.group(1)))
        assert shared_fields == {"entities", "topics", "tags", "folder", "importance"}

    def test_context_prompt_prioritizes_context_fields_before_summary(self):
        gen = self._make_generator('{"summary": "test", "doc_type": ["note"]}')

        enrich_document(
            "image notes",
            "photo.jpg",
            "img",
            gen,
            context_text="[BEFORE source_message_id=m1] Unit E",
        )

        prompt = gen.generate.call_args[0][0]
        assert "The context_* fields are required output keys; never omit them." in prompt
        assert prompt.index('"context_entities_places"') < prompt.index('"summary"')

    def test_context_prompt_rejects_placeholders_and_conflict_confidence(self):
        gen = self._make_generator('{"summary": "test", "doc_type": ["note"]}')

        enrich_document(
            "image notes",
            "photo.jpg",
            "img",
            gen,
            context_text="[BEFORE source_message_id=m1] Unit E",
        )

        prompt = gen.generate.call_args[0][0]
        assert "Never copy placeholder/example/schema description text into values" in prompt
        assert "If candidates conflict, set context_confidence to ambiguous" in prompt

    def test_repairs_context_used_when_model_omits_context_fields(self):
        response = json.dumps({
            "summary": "Photo is related to Unit E, as indicated by nearby context.",
            "doc_type": ["image"],
            "entities_places": ["54 S Broad Main Unit E"],
            "topics": ["storage"],
            "key_facts": ["Nearby context identifies this photo as Unit E."],
            "importance": 0.5,
        })
        gen = self._make_generator(response)

        result = enrich_document(
            "Blurry photo of shelves. No address visible.",
            "photo.jpg",
            "img",
            gen,
            context_text=(
                "[BEFORE source_message_id=m-unit-e] "
                "These next photos are for 54 S Broad Main Unit E."
            ),
        )

        assert result["enr_context_entities_places"] == "54 S Broad Main Unit E"
        assert result["enr_context_confidence"] == "medium"
        assert result["enr_context_relationship"] == "llm_used_nearby_context"
        assert result["enr_context_source_message_ids"] == "m-unit-e"
        assert "omitted structured context fields" in result["enr_context_warning"]

    def test_context_repair_does_not_warn_when_context_fields_present(self):
        response = json.dumps({
            "summary": "Photo location comes from nearby context.",
            "doc_type": ["image"],
            "entities_places": ["12 Oak Ave"],
            "context_entities_places": ["12 Oak Ave"],
            "context_confidence": "high",
            "context_relationship": (
                "Nearby context provides the specific location shown in the photo."
            ),
            "context_source_message_ids": ["m-after-b"],
            "importance": 0.5,
        })
        gen = self._make_generator(response)

        result = enrich_document(
            "Photo of exterior brick wall. No readable address.",
            "photo.jpg",
            "img",
            gen,
            context_text=(
                "[AFTER source_message_id=m-after-b] "
                "That exterior stairwell photo is 12 Oak Ave rear entrance."
            ),
        )

        assert result["enr_context_entities_places"] == "12 Oak Ave"
        assert result["enr_context_confidence"] == "high"
        assert result["enr_context_source_message_ids"] == "m-after-b"
        assert result["enr_context_warning"] == ""

    def test_context_conflict_downgrades_high_confidence_to_ambiguous(self):
        response = json.dumps({
            "summary": "Closet photo may be Unit E or Unit F.",
            "doc_type": ["image"],
            "context_entities_places": ["Unit E", "Unit F"],
            "context_confidence": "high",
            "context_relationship": (
                "Nearby messages provide conflicting information about the unit."
            ),
            "context_source_message_ids": ["m-before-e", "m-after-f"],
            "importance": 0.5,
        })
        gen = self._make_generator(response)

        result = enrich_document(
            "Photo of a closet with no visible label.",
            "photo.jpg",
            "img",
            gen,
            context_text=(
                "[BEFORE source_message_id=m-before-e] These photos are for Unit E.\n"
                "[AFTER source_message_id=m-after-f] Correction: might be Unit F."
            ),
        )

        assert result["enr_context_confidence"] == "ambiguous"
        assert "conflicting" in result["enr_context_warning"]

    def test_context_repair_does_not_copy_primary_item_entities(self):
        response = json.dumps({
            "summary": "Photo of 54 S Broad Main Unit E.",
            "doc_type": ["image"],
            "entities_places": ["54 S Broad Main Unit E"],
            "importance": 0.5,
        })
        gen = self._make_generator(response)

        result = enrich_document(
            "Photo label says 54 S Broad Main Unit E.",
            "photo.jpg",
            "img",
            gen,
            context_text="[BEFORE source_message_id=m1] Lunch tomorrow?",
        )

        assert result["enr_context_entities_places"] == ""
        assert result["enr_context_confidence"] == ""

    def test_none_context_text_uses_no_context_prompt(self):
        gen = self._make_generator('{"summary": "test", "doc_type": ["note"]}')

        result = enrich_document(
            "Some text",
            "doc.md",
            "md",
            gen,
            context_text=None,
        )

        prompt = gen.generate.call_args[0][0]
        assert result["enr_summary"] == "test"
        assert "PRIMARY ITEM" not in prompt
        assert "NEARBY SAME-CHANNEL CONTEXT CANDIDATES" not in prompt
        gen.generate.assert_called_once()


# A Home Depot e-receipt body in the layout production indexes (#2526), with the
# store, people, codes and every number sanitized. Two four-digit tails sit side
# by side: the masked payment card (4821) and the phone-shaped Pro Xtra loyalty
# member ID (7305), which is not a card.
HOME_DEPOT_RECEIPT = (
    "Subject: Your Electronic Receipt\n\nBody:\n"
    "The Home Depot 96 96 Please keep this mail for your records. Thank you for "
    "shopping with The Home Depot. 100 EXAMPLE PIKE ANYTOWN, PA 18000 STORE MGR "
    "610-555-0142 1000 00002 00001 01/15/26 10:30 AM SALE 045242540389 "
    "SCRWSETTER4P <A> MKE DRYWALL SCREW SETTER SET 4PC 2@5.00 10.00 019442146849 "
    "3/4 CAP BLAC <A> 3.00 843382100551 HVYDTY100PK <A> 15.00 049057104934 "
    "TANK LEVER <A> 12.00 SUBTOTAL 40.00 SALES TAX 2.40 TOTAL $42.40 "
    "XXXXXXXXXXXX4821 VISA USD$ 42.40 AUTH CODE S00000/0000000 TA AUTH MODE - "
    "ISSUER Contactless AID A0000000031010 VISA CREDIT PRO XTRA MEMBER STATEMENT "
    "PRO XTRA ###-###-7305 SUMMARY THIS RECEIPT PO/JOB NAME: 101 2026 PRO XTRA "
    "SPEND 09/15: $1,234.56 Get the CREDIT LINE your business needs when you "
    "join Pro Xtra, register, & use your Pro Xtra Credit Card. RETURN POLICY "
    "DEFINITIONS POLICY ID DAYS POLICY EXPIRES ON A 1 90 04/15/2026"
)

# The other facts the model wrote for the production receipt, which were right.
HOME_DEPOT_TRUE_FACTS = [
    "Total transaction amount is $42.40.",
    "Purchase occurred on 2026-01-15 at 10:30 AM.",
    "Items purchased include a 4pc drywall screw setter set, 3/4\" black cap, "
    "Husky heavy-duty utility blades, and a Strongarm wave tank lever.",
    "The transaction is associated with PO/Job Name 101 2026.",
    "Return policy expires on 2026-04-15.",
]


def _receipt_response(card_fact: str, summary: str = "Home Depot receipt for $42.40.") -> str:
    facts = list(HOME_DEPOT_TRUE_FACTS)
    facts.insert(3, card_fact)
    return json.dumps({
        "summary": summary,
        "doc_type": ["receipt"],
        "keywords": ["Home Depot", "receipt", "Pro Xtra", "Visa", "PO/JOB NAME 101"],
        "key_facts": facts,
        "importance": 0.5,
    })


class TestCardSuffixGrounding:
    """#2526: a stored card suffix must be a card the source actually shows."""

    def _enrich(self, response: str, text: str = HOME_DEPOT_RECEIPT, **kwargs) -> dict:
        gen = MagicMock()
        gen.generate.return_value = response
        return enrich_document(text, "Your Electronic Receipt", "email", gen, **kwargs)

    def test_loyalty_member_number_is_not_stored_as_the_payment_card(self):
        """The exact production failure: the member ID's tail named as the card."""
        result = self._enrich(
            _receipt_response("Payment was made via Visa Pro Xtra credit card ending in 7305.")
        )

        facts = json.loads(result["enr_key_facts"])
        assert "7305" not in result["enr_key_facts"]
        assert facts[3] == "Payment was made via Visa Pro Xtra credit card ending in 4821."
        # Nothing else in the receipt's enrichment moves.
        assert facts[:3] + facts[4:] == HOME_DEPOT_TRUE_FACTS
        assert result["enr_summary"] == "Home Depot receipt for $42.40."
        assert result["enr_keywords"] == "Home Depot, receipt, Pro Xtra, Visa, PO/JOB NAME 101"

    def test_summary_claim_is_grounded_too(self):
        result = self._enrich(
            _receipt_response(
                "Paid by Visa ending in 4821.",
                summary="Home Depot receipt for $42.40 paid with a Visa card ending in 7305.",
            )
        )

        assert result["enr_summary"] == (
            "Home Depot receipt for $42.40 paid with a Visa card ending in 4821."
        )

    def test_claim_matching_the_masked_card_is_unchanged(self):
        response = _receipt_response("Paid by Visa ending in 4821 using contactless payment.")

        result = self._enrich(response)

        assert result["enr_key_facts"] == parse_enrichment_response(response)["enr_key_facts"]

    @pytest.mark.parametrize(
        ("text", "card_fact"),
        [
            # Home Depot's other e-receipt layout: an icon, a dash, the bare tail.
            (
                "Order Total: $36.50 Payment: [credit_card_icon_20x13.png]\n— 4821\n"
                "Pro Xtra ###-###-7305",
                "Order total is $36.50, paid by credit card ending in 4821.",
            ),
            # Payment-processor layout: the tail after a label, sometimes starred.
            (
                "Account Number with Aqua: *1357\nBank Account or Card #: *2468\n",
                "Bank account or card ending in 2468.",
            ),
            # Card statement layout: an elided account number.
            (
                "Account Chase Credit Card (...9753)\nDue date 09/20/2026",
                "The credit card statement is for the card ending in 9753.",
            ),
        ],
    )
    def test_correct_claims_in_other_layouts_are_unchanged(self, text, card_fact):
        response = _receipt_response(card_fact)

        result = self._enrich(response, text=text)

        assert result["enr_key_facts"] == parse_enrichment_response(response)["enr_key_facts"]

    def test_claim_is_dropped_when_the_real_card_is_ambiguous(self):
        text = HOME_DEPOT_RECEIPT.replace(
            "VISA CREDIT", "XXXXXXXXXXXX6650 MASTERCARD USD$ 10.00 VISA CREDIT"
        )

        result = self._enrich(
            _receipt_response("Payment was made via Visa Pro Xtra credit card ending in 7305."),
            text=text,
        )

        facts = json.loads(result["enr_key_facts"])
        assert facts[3] == "Payment was made via Visa Pro Xtra credit card."
        assert facts[:3] + facts[4:] == HOME_DEPOT_TRUE_FACTS

    def test_non_card_identifiers_keep_their_phone_shaped_tail(self):
        """The member ID really does end in 7305; only card claims are guarded."""
        response = _receipt_response("The Pro Xtra member number ends in 7305.")

        result = self._enrich(response)

        assert result["enr_key_facts"] == parse_enrichment_response(response)["enr_key_facts"]

    def test_card_seen_only_in_nearby_context_is_grounded(self):
        response = json.dumps({
            "summary": "Delivery checklist for a dryer order.",
            "doc_type": ["delivery_notification"],
            "key_facts": ["Order total is $500.00."],
            "context_key_facts": ["The nearby receipt was paid by credit card ending in 4821."],
            "context_confidence": "medium",
            "context_relationship": "same order",
        })

        result = self._enrich(
            response,
            text="Your dryer delivery is scheduled. Questions? Call 866-555-4821.",
            context_text="[BEFORE source_message_id=m1] TOTAL $500.00 XXXXXXXXXXXX4821 VISA",
        )

        assert json.loads(result["enr_context_key_facts"]) == [
            "The nearby receipt was paid by credit card ending in 4821."
        ]

    def test_correction_is_logged_for_counting(self, caplog):
        with caplog.at_level(logging.WARNING, logger="doc_enrichment"):
            self._enrich(
                _receipt_response("Payment was made via Visa Pro Xtra credit card ending in 7305.")
            )

        corrections = [r.getMessage() for r in caplog.records if "card suffix" in r.getMessage()]
        assert corrections == [
            "Ungrounded card suffix in enrichment for 'Your Electronic Receipt': "
            "enr_key_facts 7305 -> 4821"
        ]


class TestFailedEnrichment:
    """Test the failed_enrichment() helper."""

    def test_has_all_fields_plus_reason(self):
        result = failed_enrichment("timeout")
        for field in ENRICHMENT_FIELDS:
            assert field in result
            assert result[field] == ""
        assert result["_enrichment_failed"] == "timeout"

    def test_different_reasons(self):
        r1 = failed_enrichment("json_parse_error: expecting value")
        r2 = failed_enrichment("RuntimeError: connection refused")
        assert r1["_enrichment_failed"] != r2["_enrichment_failed"]

    def test_carries_transient_classification(self):
        # Provider-level failures are marked transient so the degraded ledger
        # doesn't charge its attempts cap for an LLM-provider outage (#0251).
        assert failed_enrichment("boom", transient=True)["_enrichment_transient"] is True
        assert failed_enrichment("boom")["_enrichment_transient"] is False

    def test_enrich_document_classifies_provider_down_as_transient(self):
        import httpx

        gen = MagicMock()
        gen.generate.side_effect = httpx.ConnectError("[Errno 111] Connection refused")
        result = enrich_document("Some text", "doc.md", "md", gen)
        assert result["_enrichment_failed"]
        assert result["_enrichment_transient"] is True

    def test_enrich_document_classifies_bad_json_as_doc_specific(self):
        gen = MagicMock()
        gen.generate.return_value = "This is not valid JSON at all!"
        result = enrich_document("Some text", "doc.md", "md", gen)
        assert "json_parse_error" in result["_enrichment_failed"]
        assert result["_enrichment_transient"] is False


# -----------------------------------------------------------------------
# Live integration tests — real OpenRouter enrichment
# -----------------------------------------------------------------------

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))


@pytest.mark.live
@pytest.mark.skipif(not _has_openrouter, reason="OPENROUTER_API_KEY not set")
class TestEnrichmentLiveOpenRouter:
    """Live enrichment tests against real OpenRouter using production config."""

    @pytest.fixture(scope="class")
    def enrichment_config(self):
        from core.config import load_config
        from providers.llm import build_llm_provider
        config = load_config()
        generator = build_llm_provider(config)
        enr_cfg = config.get("enrichment", {})
        return {
            "generator": generator,
            "max_input_chars": enr_cfg.get("max_input_chars", 20000),
            "max_output_tokens": enr_cfg.get("max_output_tokens", 5000),
        }

    def test_enrichment_returns_valid_json(self, enrichment_config):
        """Real LLM should return parseable enrichment JSON."""
        result = enrich_document(
            text=(
                "# Insurance Claim Report\n\n"
                "Claim #2024-5678 for roof damage at 123 Main St, filed by John Smith "
                "on 2024-03-15. Adjuster Sarah Johnson inspected for ABC Insurance. "
                "Estimated repair cost: $12,500."
            ),
            title="claim_report.pdf",
            source_type="pdf",
            generator=enrichment_config["generator"],
            max_input_chars=enrichment_config["max_input_chars"],
            max_output_tokens=enrichment_config["max_output_tokens"],
        )
        assert result["enr_summary"], "Summary should not be empty"
        assert "_enrichment_failed" not in result

    def test_enrichment_entities_extracted(self, enrichment_config):
        """Real LLM should extract entities from structured text."""
        result = enrich_document(
            text=(
                "Meeting at Google headquarters in Mountain View on 2024-06-15. "
                "Attendees: Sundar Pichai, Tim Cook from Apple. "
                "Topics: AI strategy, cloud computing."
            ),
            title="meeting.md",
            source_type="md",
            generator=enrichment_config["generator"],
            max_input_chars=enrichment_config["max_input_chars"],
            max_output_tokens=enrichment_config["max_output_tokens"],
        )
        has_people = bool(result["enr_entities_people"])
        has_orgs = bool(result["enr_entities_orgs"])
        assert has_people or has_orgs, f"Expected entities, got: {result}"

    def test_enrichment_with_taxonomy_suggests_tags(self, enrichment_config):
        """Real LLM + semantic inference should suggest tags from the taxonomy."""
        import tempfile
        from taxonomy_store import TaxonomyStore
        from providers.embed import build_embed_provider
        from core.config import load_config

        config = load_config()
        embed_provider = build_embed_provider(config)

        def embed_fn(text):
            return embed_provider.embed_texts([text])[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            tax_store = TaxonomyStore(tmpdir, "taxonomy", embed_fn=embed_fn)
            tax_store.add("tag", "insurance", "Insurance policies, claims, and coverage")
            tax_store.add("tag", "property", "Real estate and property matters")
            tax_store.add("tag", "finance", "Financial documents, invoices, budgets")
            tax_store.add("folder", "Insurance/Claims/", "Insurance claim documents")

            result = enrich_document(
                text=(
                    "# Insurance Claim Report\n\n"
                    "Claim #2024-5678 for roof damage at 123 Main St, "
                    "filed by John Smith on 2024-03-15. "
                    "Adjuster Sarah Johnson inspected for ABC Insurance. "
                    "Estimated repair cost: $12,500."
                ),
                title="claim_report.pdf",
                source_type="pdf",
                generator=enrichment_config["generator"],
                max_input_chars=enrichment_config["max_input_chars"],
                max_output_tokens=enrichment_config["max_output_tokens"],
                taxonomy_store=tax_store,
            )
            assert result["enr_summary"], "Summary should not be empty"
            assert "_enrichment_failed" not in result
            # Semantic inference should suggest at least one tag
            suggested = result.get("enr_suggested_tags", "")
            assert suggested, (
                f"Expected taxonomy tag suggestions (via LLM or semantic inference), "
                f"got empty. Full result: {result}"
            )
            # Insurance tag should be the best match
            assert "insurance" in suggested.lower(), (
                f"Expected 'insurance' in suggestions, got: '{suggested}'"
            )

    def test_enrichment_with_taxonomy_suggests_folder(self, enrichment_config):
        """Real LLM + semantic inference should suggest a folder from the taxonomy."""
        import tempfile
        from taxonomy_store import TaxonomyStore
        from providers.embed import build_embed_provider
        from core.config import load_config

        config = load_config()
        embed_provider = build_embed_provider(config)

        def embed_fn(text):
            return embed_provider.embed_texts([text])[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            tax_store = TaxonomyStore(tmpdir, "taxonomy", embed_fn=embed_fn)
            tax_store.add("tag", "insurance", "Insurance policies, claims, and coverage")
            tax_store.add("tag", "property", "Real estate and property matters")
            tax_store.add("folder", "Insurance/Claims/", "Insurance claim documents and reports")
            tax_store.add("folder", "Recipes/", "Cooking recipes and food preparation guides")
            tax_store.add("folder", "Work/Meetings/", "Meeting notes and minutes")

            result = enrich_document(
                text=(
                    "# Insurance Claim Report\n\n"
                    "Claim #2024-5678 for roof damage at 123 Main St, "
                    "filed by John Smith on 2024-03-15. "
                    "Adjuster Sarah Johnson inspected for ABC Insurance. "
                    "Estimated repair cost: $12,500."
                ),
                title="claim_report.pdf",
                source_type="pdf",
                generator=enrichment_config["generator"],
                max_input_chars=enrichment_config["max_input_chars"],
                max_output_tokens=enrichment_config["max_output_tokens"],
                taxonomy_store=tax_store,
            )
            assert result["enr_summary"], "Summary should not be empty"
            assert "_enrichment_failed" not in result
            folder = result.get("enr_suggested_folder", "")
            assert folder, (
                f"Expected folder suggestion (via LLM or semantic inference), "
                f"got empty. Full result: {result}"
            )
            assert "insurance" in folder.lower() or "claims" in folder.lower(), (
                f"Expected 'Insurance/Claims/' folder suggestion, got: '{folder}'"
            )
