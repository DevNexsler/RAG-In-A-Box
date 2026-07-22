from types import SimpleNamespace
from unittest.mock import MagicMock

from attachment_context_refresh import (
    build_contextualized_text,
    context_text_from_sidecar,
    refresh_document_context,
)


def test_build_contextualized_text_appends_and_replaces_suffix():
    original = "[Document: photo]\nvisual description"
    first = build_contextualized_text(original, "BEFORE MESSAGES\nBEFORE: 482 #6")
    second = build_contextualized_text(first, "AFTER MESSAGES\nAFTER: 163 Washington")

    assert first == (
        original
        + "\n\n[Conversation context]\nBEFORE MESSAGES\nBEFORE: 482 #6"
    )
    assert second == (
        original
        + "\n\n[Conversation context]\nAFTER MESSAGES\nAFTER: 163 Washington"
    )
    assert second.count("[Conversation context]") == 1


def test_build_contextualized_text_removes_stale_suffix_when_context_empty():
    existing = "body\n\n[Conversation context]\nBEFORE: stale"

    assert build_contextualized_text(existing, "") == "body"


def test_refresh_document_context_updates_existing_marker_chunk_only():
    chunks = [
        SimpleNamespace(loc="video:c:0", text="first transcript chunk"),
        SimpleNamespace(
            loc="video:c:1",
            text="last transcript\n\n[Conversation context]\nBEFORE: old address",
        ),
    ]
    store = MagicMock()
    store.get_doc_chunks.return_value = chunks
    store.replace_chunk_text_and_vector.return_value = True
    embed = MagicMock()
    embed.embed_texts.return_value = [[0.2, 0.3]]

    changed = refresh_document_context(
        store,
        embed,
        "documents::video",
        "BEFORE: 482 #6\nAFTER: 163 Washington",
    )

    assert changed is True
    embed.embed_texts.assert_called_once_with(
        [
            "last transcript\n\n[Conversation context]\n"
            "BEFORE: 482 #6\nAFTER: 163 Washington"
        ]
    )
    store.replace_chunk_text_and_vector.assert_called_once_with(
        "documents::video",
        "video:c:1",
        chunks[1].text,
        (
            "last transcript\n\n[Conversation context]\n"
            "BEFORE: 482 #6\nAFTER: 163 Washington"
        ),
        embed.embed_texts.return_value[0],
    )


def test_refresh_document_context_skips_matching_text_without_embedding():
    text = "photo\n\n[Conversation context]\nBEFORE: 482 #6"
    store = MagicMock()
    store.get_doc_chunks.return_value = [SimpleNamespace(loc="img:c:0", text=text)]
    embed = MagicMock()

    changed = refresh_document_context(
        store, embed, "documents::photo", "BEFORE: 482 #6"
    )

    assert changed is False
    embed.embed_texts.assert_not_called()
    store.replace_chunk_text_and_vector.assert_not_called()


def test_context_text_from_sidecar_includes_explicit_before_and_after(tmp_path):
    sidecar = tmp_path / "photo.json"
    sidecar.write_text(
        """{
          "source": "zoho_cliq",
          "message": {"sent_at": "2026-05-08T19:00:00Z"},
          "channel": {"source_channel_id": "renovations"},
          "media": {"media_index": 0},
          "context": {
            "same_channel_before": [],
            "same_channel_after": [],
            "nearest_nonempty_before": {
              "sent_at": "2026-05-08T18:59:00Z",
              "text": "482 #6"
            },
            "nearest_nonempty_after": {
              "sent_at": "2026-05-08T19:01:00Z",
              "text": "163 Washington"
            }
          }
        }"""
    )

    context = context_text_from_sidecar(sidecar, doc_id="documents::photo")

    assert "[BEFORE 2026-05-08T18:59:00Z] 482 #6" in context
    assert "[AFTER 2026-05-08T19:01:00Z] 163 Washington" in context
