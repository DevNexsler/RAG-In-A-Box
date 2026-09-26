"""Regression tests for controlled enr_doc_type vocabulary (#3050).

These fail against the pre-fix free-text path for the same reasons as the
production defect: unbounded labels, re-index churn, and silent overwrite.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from core.doc_type_vocabulary import (
    UNCLASSIFIED_DOC_TYPE,
    constrain_doc_type,
    default_alias_map,
    enrichment_input_hash,
    reconcile_doc_type,
)
from doc_enrichment import (
    ENRICHMENT_INPUT_HASH_FIELD,
    apply_doc_type_vocabulary,
    enrich_document,
    parse_enrichment_response,
)


def test_constrain_doc_type_maps_unknown_labels_to_unclassified():
    """Free-text synonyms outside the vocabulary must not mint new facet values."""
    result = constrain_doc_type(
        "collection_record, payment_history_log, totally_made_up_type"
    )
    assert "collection_record" in result.value
    assert "payment_log" in result.value  # alias of payment_history_log
    assert "totally_made_up_type" not in result.value
    assert result.unknown_count >= 1


def test_constrain_doc_type_all_unknown_becomes_unclassified_sentinel():
    result = constrain_doc_type("zzz_not_a_real_type, another_invention")
    assert result.value == UNCLASSIFIED_DOC_TYPE
    assert result.unknown_count == 2


def test_reconcile_doc_type_keeps_existing_on_disagreement():
    final, disagreed = reconcile_doc_type(
        existing="collection_record, payment_log",
        proposed="collection_record, financial_ledger",
    )
    assert final == "collection_record, payment_log"
    assert disagreed is True


def test_reconcile_doc_type_accepts_first_write():
    final, disagreed = reconcile_doc_type(existing="", proposed="email, notification")
    assert final == "email, notification"
    assert disagreed is False


def test_enrichment_input_hash_stable_for_identical_inputs():
    kwargs = dict(
        text="Tenant paid rent on 2026-09-20.",
        title="Mejia-Miguel",
        source_type="sor_collection",
        context_text="",
        max_input_chars=4000,
    )
    assert enrichment_input_hash(**kwargs) == enrichment_input_hash(**kwargs)


def test_enrichment_input_hash_changes_when_body_changes():
    base = dict(
        title="Mejia-Miguel",
        source_type="sor_collection",
        context_text="",
        max_input_chars=4000,
    )
    a = enrichment_input_hash(text="balance 100", **base)
    b = enrichment_input_hash(text="balance 200", **base)
    assert a != b


def test_apply_doc_type_vocabulary_records_disagreement_counter():
    enrichment = {"enr_doc_type": "message"}
    updated, counters = apply_doc_type_vocabulary(
        enrichment,
        taxonomy_store=None,
        existing_doc_type="email",
    )
    assert updated["enr_doc_type"] == "email"
    assert counters["disagreement"] == 1


def test_enrich_document_rejects_free_text_outside_vocabulary(monkeypatch):
    """Same defect shape as production: model invents a one-off synonym."""
    generator = MagicMock()
    generator.generate.return_value = json.dumps(
        {
            "summary": "Rent collection notes for Mejia-Miguel.",
            "doc_type": [
                "collection_record",
                "payment_history_log",
                "invented_one_off_label",
            ],
            "topics": ["collections"],
        }
    )

    result = enrich_document(
        text="Collection note: balance due $400.",
        title="Mejia-Miguel",
        source_type="sor_collection",
        generator=generator,
    )

    assert generator.generate.call_count == 1
    assert "collection_record" in result["enr_doc_type"]
    assert "payment_log" in result["enr_doc_type"]
    assert "invented_one_off_label" not in result["enr_doc_type"]
    assert result[ENRICHMENT_INPUT_HASH_FIELD]
    assert int(result.get("_doc_type_unknown", "0")) >= 1


def test_enrich_document_keeps_existing_label_when_model_disagrees():
    generator = MagicMock()
    generator.generate.return_value = json.dumps(
        {
            "summary": "Same collection record, different secondary label.",
            "doc_type": ["message"],
            "topics": ["collections"],
        }
    )

    result = enrich_document(
        text="Collection note: balance due $400.",
        title="Mejia-Miguel",
        source_type="sor_collection",
        generator=generator,
        existing_doc_type="email",
    )

    assert result["enr_doc_type"] == "email"
    assert result.get("_doc_type_disagreement") == "1"


def test_parse_still_canonicalizes_before_vocabulary_is_applied():
    """#1251/#1330 fold remains; vocabulary constraint is a later seam."""
    parsed = parse_enrichment_response(
        '{"summary":"x","doc_type":["Property Showing","followup"]}'
    )
    assert parsed["enr_doc_type"] == "property_showing, follow_up"


def test_default_alias_map_includes_unclassified_sentinel():
    assert UNCLASSIFIED_DOC_TYPE in default_alias_map()


def test_process_doc_skips_llm_when_enrichment_input_hash_matches(tmp_path):
    """#3050: re-index with unchanged enrichment input must not call the LLM."""
    from unittest.mock import MagicMock, patch

    import flow_index_vault as fiv
    from doc_enrichment import ENRICHMENT_INPUT_HASH_FIELD
    from llama_index.core.node_parser import SentenceSplitter
    from lancedb_store import LanceDBStore

    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    path = docs_root / "note.md"
    body = "Collection note: balance due $400 for Mejia-Miguel."
    path.write_text(body)

    store = LanceDBStore(tmp_path / "index", "chunks")

    class _MockEmbed:
        def embed_texts(self, texts):
            return [[0.1] * 768 for _ in texts]

        def embed_query(self, q):
            return [0.1] * 768

    generator = MagicMock()
    generator.generate.return_value = json.dumps(
        {
            "summary": "Rent collection notes.",
            "doc_type": ["collection_record", "payment_log"],
            "topics": ["collections"],
        }
    )

    logger_patch = patch("flow_index_vault.get_run_logger", return_value=MagicMock())
    logger_patch.start()
    fiv._RUNTIME.clear()
    fiv._RUNTIME.update(
        {
            "store": store,
            "embed_provider": _MockEmbed(),
            "splitter": SentenceSplitter(chunk_size=512, chunk_overlap=20),
            "llm_generator": generator,
            "taxonomy_store": None,
            "config": {"enrichment": {"enabled": True, "max_input_chars": 4000}},
        }
    )
    try:
        doc = {
            "doc_id": "documents::cachehit",
            "rel_path": "note.md",
            "abs_path": str(path),
            "mtime": path.stat().st_mtime,
            "size": path.stat().st_size,
            "ext": "md",
            "source_name": "documents",
        }
        fiv.process_doc_task.fn(doc)
        assert generator.generate.call_count == 1
        first_type = store.get_doc_chunks(doc["doc_id"])[0].enr_doc_type
        assert "collection_record" in first_type

        # Bump mtime/size metadata without changing enrichment input text.
        path.touch()
        doc["mtime"] = path.stat().st_mtime
        doc["size"] = path.stat().st_size
        fiv.process_doc_task.fn(doc)
        assert generator.generate.call_count == 1, "second pass must reuse enrichment"
        second = store.get_doc_chunks(doc["doc_id"])[0]
        assert second.enr_doc_type == first_type
        assert second.extra_metadata.get(ENRICHMENT_INPUT_HASH_FIELD) or getattr(
            second, ENRICHMENT_INPUT_HASH_FIELD, ""
        )
        counters = fiv._RUNTIME.get("_enrichment_counters") or {}
        assert counters.get("cache_hit", 0) >= 1
    finally:
        logger_patch.stop()
        fiv._RUNTIME.clear()
