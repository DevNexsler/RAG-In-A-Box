"""Cross-PR regressions found while reviewing the September backlog."""

from core.enrichment_postprocess import ground_card_suffixes, repair_enrichment
from doc_enrichment import apply_doc_type_vocabulary
from extractors import CONTENT_MISSING, ExtractionResult


def test_metadata_only_extraction_is_missing_without_provider_exception():
    result = ExtractionResult.from_text("EXIF camera metadata", primary_content=False)
    assert result.content_status([]) == CONTENT_MISSING


def test_invalid_stored_doc_type_cannot_override_current_vocabulary():
    result, _ = apply_doc_type_vocabulary(
        {"enr_doc_type": "email"}, existing_doc_type="invented_type"
    )
    assert result["enr_doc_type"] == "email"


def test_route_arrow_is_not_an_identity_correction():
    original = {"enr_summary": "Route goes from A to B, not B to A."}
    assert repair_enrichment(
        original, text="A -> B", title="Route", source_type="message", enabled=False
    ) == original


def test_correction_preserves_unrelated_summary_sentences():
    result = repair_enrichment(
        {"enr_summary": "Invoice due Friday. The name is Shawn, not Sean."},
        text="Correction: changed from Shawn to Sean.",
        title="Correction", source_type="message", enabled=False,
    )
    assert result["enr_summary"] == "Invoice due Friday. Correction: Sean (not Shawn)."


def test_invoice_number_cannot_support_card_suffix():
    result, _ = ground_card_suffixes(
        {"enr_summary": "Paid by Visa ending in 1234."},
        source_text="Invoice 1234; Visa ****5678",
    )
    assert result["enr_summary"] == "Paid by Visa ending in 5678."


def test_masked_member_account_cannot_become_card_suffix():
    result, _ = ground_card_suffixes(
        {"enr_summary": "Paid by Visa ending in 1234."},
        source_text="Member account ****9999; paid cash",
    )
    assert result["enr_summary"] == "Paid by Visa."


def test_folder_sync_repairs_deleted_taxonomy_entry(tmp_path):
    from core.taxonomy import sync_folder_taxonomy_from_filesystem

    class Store:
        def __init__(self):
            self.rows = {}

        def list_by_kind(self, kind):
            return list(self.rows.values())

        def add(self, kind, name, description, **entry):
            self.rows[f"folder:{name}"] = {
                **entry, "id": f"folder:{name}", "name": name, "description": description
            }

    (tmp_path / "Invoices").mkdir()
    store = Store()
    assert sync_folder_taxonomy_from_filesystem(store, tmp_path)["added"] == 1
    store.rows.clear()
    assert sync_folder_taxonomy_from_filesystem(store, tmp_path)["added"] == 1


def test_unrelated_correction_cannot_turn_route_arrow_into_correction():
    original = {"enr_summary": "Route goes from A to B, not B to A."}
    assert repair_enrichment(
        original, text="Correction: changed from Shawn to Sean.\nRoute: A -> B",
        title="Message", source_type="message", enabled=False,
    ) == original
