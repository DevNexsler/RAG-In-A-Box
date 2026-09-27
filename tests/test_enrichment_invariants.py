"""Negative evidence and idempotence: enrichment must not invent or erase facts."""

import pytest

from core.enrichment_postprocess import ground_card_suffixes, repair_enrichment
from sources.text_normalization import normalize_source_text


@pytest.mark.parametrize("label", ["Invoice", "Order", "Member account", "Room", "Reference", "PIN"])
def test_unrelated_identifier_cannot_ground_a_card_suffix(label):
    source = f"{label} 1234. Paid with Visa ****5678."
    grounded, _ = ground_card_suffixes({"enr_summary": "Paid by Visa ending in 1234."}, source_text=source)
    assert grounded["enr_summary"] == "Paid by Visa ending in 5678."
    again, _ = ground_card_suffixes(grounded, source_text=source)
    assert again == grounded, "reprocessing grounded facts must be idempotent"


@pytest.mark.parametrize("arrow", ["->", "→", "=>"])
def test_correction_in_another_paragraph_cannot_rewrite_route(arrow):
    summary = {"enr_summary": "Invoice due Friday. Route goes from A to B, not B to A."}
    source = f"Correction: changed from Shawn to Sean.\nRoute: A {arrow} B."
    assert repair_enrichment(summary, text=source, title="Route", source_type="message", enabled=False) == summary


@pytest.mark.parametrize("hidden", [
    '<div hidden/>', '<span style="display:none"/>',
    '<div hidden><br></br>secret</div>',
    '<div hidden><img src="x"/></div>',
    '<script>secret</script>',
])
def test_hidden_html_cannot_swallow_following_visible_content(hidden):
    normalized = normalize_source_text("pg_message", hidden + "<p>Invoice 42 due Friday.</p>")
    assert normalized.text == "Invoice 42 due Friday."
    assert normalize_source_text("pg_message", normalized.text).text == normalized.text
