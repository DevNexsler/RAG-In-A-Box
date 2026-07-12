"""Tests for the #0264 one-time sweep (scripts/backfill_unledgered_stub_docs.py).

Image docs stub-indexed WITHOUT a degraded-ledger entry (empty describe during
the vision outage, dropped notes on the targeted path) have no retry path.
The sweep detects metadata-only chunk text and enqueues those docs.
"""

from scripts.backfill_unledgered_stub_docs import (
    BACKFILL_REASON,
    backfill,
    find_stub_docs,
    is_metadata_stub_text,
)

# Verbatim shape of documents::0028e in production (2026-07-12).
PROD_STUB = (
    "[Document: documents::0028e]\n"
    "Summary: An image file with dimensions 600x800 pixels in JPEG format. "
    "No additional content or context is available.\n\n"
    "Image dimensions: 600x800\nFormat: JPEG"
)

REAL_DESCRIBE = (
    "[Document: documents::0027V | Topics: identity verification]\n"
    "Summary: Driver's license photo for identity verification.\n\n"
    "--- Text ---\nSAMIRA DAVIS\nDL 1234567\n\n--- Description ---\n"
    "A photo of a driver's license held up to the camera.\n\n"
    "Image dimensions: 600x800\nFormat: JPEG"
)


def test_prod_stub_is_detected():
    assert is_metadata_stub_text(PROD_STUB)


def test_real_describe_is_not_a_stub():
    assert not is_metadata_stub_text(REAL_DESCRIBE)


def test_stub_without_context_header_is_detected():
    # Enrichment disabled / empty title: no [Document...] or Summary lines.
    assert is_metadata_stub_text("Image dimensions: 600x800\nFormat: JPEG")


def test_exif_rich_stub_is_detected():
    # EXIF-only summary (the documents::0028a shape): camera/GPS lines but no
    # describe content is still a stub.
    assert is_metadata_stub_text(
        "Summary: Photo created on an Android device.\n\n"
        "Image dimensions: 3024x4032\nFormat: JPEG\n"
        "Camera: samsung SM-G991U\nDate taken: 2026:07:11 15:55:02\n"
        "Software: G991USQU9FXCA\nGPS: 33.749, -84.388"
    )


def test_caption_only_stub_is_detected():
    # A communication caption is copied from message metadata, not extracted
    # from the image — the visual content is still lost.
    assert is_metadata_stub_text(
        "Communication message/caption: here's the receipt\n\n"
        "Image dimensions: 600x800\nFormat: JPEG"
    )


def test_multiline_summary_stub_is_detected():
    assert is_metadata_stub_text(
        "[Document: documents::0026K]\n"
        "Summary: An image file.\nNo further content available.\n\n"
        "Image dimensions: 100x100\nFormat: PNG"
    )


def test_find_stub_docs_requires_every_chunk_to_be_stub():
    rows = [
        ("documents::stub", PROD_STUB),
        ("documents::real", PROD_STUB),  # one stub-looking chunk…
        ("documents::real", REAL_DESCRIBE),  # …but another has content
    ]
    assert find_stub_docs(rows) == ["documents::stub"]


def test_backfill_adds_only_unledgered_docs():
    ledger = {
        "docs": {
            "documents::capped": {"reasons": ["ocr_describe_failed"], "attempts": 5}
        }
    }
    updated, added = backfill(ledger, ["documents::capped", "documents::stub"])

    assert added == ["documents::stub"]
    # existing entry untouched (capped docs are #0251/PR#60 territory)
    assert updated["docs"]["documents::capped"]["attempts"] == 5
    assert updated["docs"]["documents::stub"] == {
        "reasons": [BACKFILL_REASON],
        "attempts": 0,
    }


def test_backfill_preserves_ledger_version_fields():
    # PR #60 stamps the ledger with version: 2 — the sweep must not drop it.
    updated, _ = backfill({"version": 2, "docs": {}}, ["documents::stub"])
    assert updated["version"] == 2
