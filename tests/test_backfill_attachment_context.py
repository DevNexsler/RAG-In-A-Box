from communication_context import CommunicationItem
from scripts.backfill_attachment_context import (
    candidates_from_rows,
    context_provider_from_rows,
)


def test_candidates_from_rows_filters_deduplicates_sorts_and_limits():
    rows = [
        {
            "doc_id": "documents::b",
            "metadata": {
                "source_type": "video",
                "sidecar_path": "/data/b.json",
            },
        },
        {
            "doc_id": "documents::a",
            "metadata": {
                "source_type": "img",
                "sidecar_path": "/data/a.json",
            },
        },
        {
            "doc_id": "documents::a",
            "metadata": {
                "source_type": "img",
                "sidecar_path": "/data/a.json",
            },
        },
        {
            "doc_id": "documents::text",
            "metadata": {
                "source_type": "md",
                "sidecar_path": "/data/text.json",
            },
        },
    ]

    candidates = candidates_from_rows(rows, source_types={"img", "video"}, limit=1)

    assert [(c.doc_id, c.source_type, str(c.sidecar_path)) for c in candidates] == [
        ("documents::a", "img", "/data/a.json")
    ]


def test_candidates_from_rows_honors_explicit_document_ids():
    rows = [
        {
            "doc_id": "documents::a",
            "metadata": {"source_type": "img", "sidecar_path": "/data/a.json"},
        },
        {
            "doc_id": "documents::b",
            "metadata": {"source_type": "video", "sidecar_path": "/data/b.json"},
        },
    ]

    candidates = candidates_from_rows(
        rows,
        source_types={"img", "video"},
        doc_ids={"documents::b"},
        limit=5,
    )

    assert [c.doc_id for c in candidates] == ["documents::b"]


def test_context_provider_from_rows_rebuilds_symmetric_message_window():
    rows = [
        {
            "doc_id": "comm::before",
            "metadata": {
                "source_type": "pg_message",
                "source": "zoho_cliq",
                "source_channel_id": "repairs",
                "source_message_id": "before",
                "sent_at": "2026-05-08T18:59:00Z",
                "snippet": "482 #6",
            },
        },
        {
            "doc_id": "comm::after",
            "metadata": {
                "source_type": "pg_message",
                "source": "zoho_cliq",
                "source_channel_id": "repairs",
                "source_message_id": "after",
                "sent_at": "2026-05-08T19:04:00Z",
                "snippet": "163 Washington",
            },
        },
    ]
    target = CommunicationItem(
        doc_id="documents::photo",
        origin_source="zoho_cliq",
        channel_id="repairs",
        sent_at="2026-05-08T19:00:00Z",
    )

    provider = context_provider_from_rows(
        rows,
        {"max_time_window_minutes": 15},
    )
    envelope = provider.get_context_envelope(target)

    assert [message.text for message in envelope.same_channel_before] == ["482 #6"]
    assert [message.text for message in envelope.same_channel_after] == [
        "163 Washington"
    ]
