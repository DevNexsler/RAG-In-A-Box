import json

from hooks.events import build_document_indexed_event


def test_build_document_indexed_event_includes_document_text_and_chunks():
    event = build_document_indexed_event(
        doc_id="documents::000hF",
        source_name="documents",
        source_type="img",
        rel_path="email-attachments/gunther/photo@000hF@.jpg",
        abs_path="/data/documents/email-attachments/gunther/photo@000hF@.jpg",
        text="OCR text from image",
        metadata={"enr_summary": "White wall.", "enr_topics": "visual description"},
        chunks=[
            {"loc": "img:c:0", "snippet": "White wall.", "text": "OCR text from image"},
        ],
        occurred_at="2026-04-30T00:00:00+00:00",
    )

    assert event["event"] == "document.indexed"
    assert event["version"] == 1
    assert event["occurred_at"] == "2026-04-30T00:00:00+00:00"
    assert event["doc_id"] == "documents::000hF"
    assert event["source_name"] == "documents"
    assert event["source_type"] == "img"
    assert event["rel_path"] == "email-attachments/gunther/photo@000hF@.jpg"
    assert event["abs_path"] == "/data/documents/email-attachments/gunther/photo@000hF@.jpg"
    assert event["text"] == "OCR text from image"
    assert event["metadata"]["enr_summary"] == "White wall."
    assert event["chunks"] == [{"loc": "img:c:0", "snippet": "White wall.", "text": "OCR text from image"}]
    json.dumps(event)


def test_build_document_indexed_event_drops_internal_metadata_keys():
    event = build_document_indexed_event(
        doc_id="documents::000hF",
        source_name="documents",
        source_type="img",
        rel_path="photo@000hF@.jpg",
        abs_path="/data/documents/photo@000hF@.jpg",
        text="OCR text",
        metadata={
            "_node_content": "internal",
            "_node_type": "TextNode",
            "document_id": "documents::000hF",
            "ref_doc_id": "documents::000hF",
            "enr_summary": "safe",
        },
        chunks=[],
        occurred_at="2026-04-30T00:00:00+00:00",
    )

    assert event["metadata"] == {"enr_summary": "safe"}
