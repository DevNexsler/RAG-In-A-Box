from scripts.backfill_attachment_context import candidates_from_rows


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
