"""Tests for doc_enrichment.py — LLM document enrichment parsing and normalization.

Unit tests mock the LLM generator.  Integration test uses the real MiniMax M2.5 via OpenRouter.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from doc_enrichment import (
    ENRICHMENT_FIELDS,
    _ENRICHMENT_KEYS_RAW,
    _extract_json,
    _normalize_list,
    _normalize_enrichment,
    empty_enrichment,
    failed_enrichment,
    enrich_document,
    parse_enrichment_response,
)


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


class TestEnrichDocument:
    """Test enrich_document with mocked LLM generator."""

    def _make_generator(self, response: str) -> MagicMock:
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

    def test_exception_in_generate_returns_failed(self):
        gen = MagicMock()
        gen.generate.side_effect = RuntimeError("model unavailable")
        result = enrich_document("Some text", "doc.md", "md", gen)
        assert result["_enrichment_failed"]
        assert "RuntimeError" in result["_enrichment_failed"]

    def test_text_truncation_short_doc(self):
        """Documents shorter than max_input_chars are passed through entirely."""
        gen = self._make_generator('{"summary": "short"}')
        short_text = "hello world"
        enrich_document(short_text, "small.md", "md", gen, max_input_chars=500)
        call_args = gen.generate.call_args[0][0]
        assert "hello world" in call_args
        assert "[...]" not in call_args

    def test_text_truncation_head_tail(self):
        """Long documents use head+tail sampling with [...] separator."""
        gen = self._make_generator('{"summary": "short"}')
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
        response = '```json\n{"summary": "A doc", "doc_type": ["note"]}\n```'
        gen = self._make_generator(response)
        result = enrich_document("Some text", "note.md", "md", gen)
        assert result["enr_summary"] == "A doc"
        assert result["enr_doc_type"] == "note"

    def test_thinking_tags_in_response(self):
        response = '<think>Let me analyze...</think>\n{"summary": "Analyzed", "topics": ["AI"]}'
        gen = self._make_generator(response)
        result = enrich_document("Some text", "doc.md", "md", gen)
        assert result["enr_summary"] == "Analyzed"
        assert result["enr_topics"] == "AI"


    def test_taxonomy_block_in_prompt(self):
        """When taxonomy_store is provided, its format_for_prompt output appears in the LLM prompt."""
        gen = self._make_generator('{"summary": "test"}')
        mock_taxonomy = MagicMock()
        mock_taxonomy.format_for_prompt.return_value = "## Available Tags\n- work: Work stuff"
        mock_taxonomy.increment_usage = MagicMock()

        enrich_document("Some text", "doc.md", "md", gen, taxonomy_store=mock_taxonomy)
        call_args = gen.generate.call_args[0][0]
        assert "## Available Tags" in call_args
        assert "work: Work stuff" in call_args

    def test_no_taxonomy_no_block(self):
        """Without taxonomy_store, no taxonomy block in prompt."""
        gen = self._make_generator('{"summary": "test"}')
        enrich_document("Some text", "doc.md", "md", gen, taxonomy_store=None)
        call_args = gen.generate.call_args[0][0]
        assert "## Available Tags" not in call_args


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
