from pathlib import Path

from communication_context import (
    CommunicationItem,
    CommunicationMessage,
    SourceWindowContextProvider,
    communication_item_from_sidecar,
)


def test_sidecar_attachment_becomes_communication_item(tmp_path: Path):
    media = tmp_path / "2026-04-22T14-56-22Z__msg4442__mm0@000kM@.jpg"
    media.write_bytes(b"fake")
    sidecar = tmp_path / "2026-04-22T14-56-22Z__msg4442__mm0@000kD@.json"
    sidecar.write_text(
        """
        {
          "source": "zoho_cliq",
          "message": {
            "message_id": "4442",
            "source_message_id": "1776869782220_21353330717388",
            "sent_at": "2026-04-22T14:56:22.220Z",
            "from": {"name": "Joycelyn Smith"}
          },
          "channel": {
            "source_channel_id": "2242125288797599446",
            "channel_type": "conversation"
          },
          "media": {
            "media_index": 0,
            "media_type": "image/jpeg",
            "original_filename": "IMG_2133.HEIC"
          }
        }
        """
    )

    item = communication_item_from_sidecar(media, sidecar)

    assert item.origin_source == "zoho_cliq"
    assert item.message_id == "4442"
    assert item.source_message_id == "1776869782220_21353330717388"
    assert item.channel_id == "2242125288797599446"
    assert item.sender == "Joycelyn Smith"
    assert item.attachment_index == "0"
    assert item.batch_key == "zoho_cliq:2242125288797599446:2026-04-22T14:56:22Z"


def test_sidecar_attachment_handles_malformed_sender(tmp_path: Path):
    media = tmp_path / "message.jpg"
    media.write_bytes(b"fake")
    sidecar = tmp_path / "message.json"
    sidecar.write_text(
        """
        {
          "source": "zoho_cliq",
          "message": {
            "message_id": "4442",
            "sent_at": "2026-04-22T14:56:22.220Z",
            "from": "Joycelyn Smith"
          },
          "channel": {
            "source_channel_id": "2242125288797599446"
          },
          "media": {
            "media_index": 0
          }
        }
        """
    )

    item = communication_item_from_sidecar(media, sidecar)

    assert item.sender == ""


def test_provider_uses_same_channel_only():
    target = CommunicationItem(
        doc_id="attachment-1",
        origin_source="zoho_cliq",
        channel_id="chan-a",
        sent_at="2026-04-22T14:56:22Z",
    )
    messages = [
        CommunicationMessage(
            message_id="1",
            source_message_id="a1",
            sent_at="2026-04-22T14:55:00Z",
            text="Unit E",
        ),
        CommunicationMessage(
            message_id="2",
            source_message_id="b1",
            sent_at="2026-04-22T14:55:30Z",
            text="Other channel",
        ),
        CommunicationMessage(
            message_id="3",
            source_message_id="a2",
            sent_at="2026-04-22T14:57:00Z",
            text="Also this",
        ),
    ]
    message_channels = {"1": "chan-a", "2": "chan-b", "3": "chan-a"}

    provider = SourceWindowContextProvider.from_messages(
        messages,
        message_channels=message_channels,
        message_sources={"1": "zoho_cliq", "2": "zoho_cliq", "3": "zoho_cliq"},
        window_before=5,
        window_after=5,
    )

    envelope = provider.get_context_envelope(target)

    assert [m.message_id for m in envelope.same_channel_before] == ["1"]
    assert [m.message_id for m in envelope.same_channel_after] == ["3"]


def test_provider_tracks_nearest_nonempty_messages():
    target = CommunicationItem(
        doc_id="a",
        channel_id="c",
        sent_at="2026-01-01T10:00:00Z",
    )
    messages = [
        CommunicationMessage(
            message_id="1",
            sent_at="2026-01-01T09:59:00Z",
            text="Building 54",
        ),
        CommunicationMessage(
            message_id="2",
            sent_at="2026-01-01T09:59:30Z",
            text="",
        ),
        CommunicationMessage(
            message_id="3",
            sent_at="2026-01-01T10:00:30Z",
            text="",
        ),
        CommunicationMessage(
            message_id="4",
            sent_at="2026-01-01T10:01:00Z",
            text="Unit E",
        ),
    ]
    provider = SourceWindowContextProvider.from_messages(
        messages,
        message_channels={m.message_id: "c" for m in messages},
    )

    envelope = provider.get_context_envelope(target)

    assert envelope.nearest_nonempty_before.message_id == "1"
    assert envelope.nearest_nonempty_after.message_id == "4"


def test_provider_orders_same_timestamp_messages_around_target_id():
    target = CommunicationItem(
        doc_id="attachment-2",
        message_id="2",
        source_message_id="src-2",
        channel_id="chan-a",
        sent_at="2026-04-22T14:56:22Z",
    )
    messages = [
        CommunicationMessage(
            message_id="1",
            source_message_id="src-1",
            sent_at="2026-04-22T14:56:22Z",
            text="Before same second",
        ),
        CommunicationMessage(
            message_id="2",
            source_message_id="src-2",
            sent_at="2026-04-22T14:56:22Z",
            text="Target message",
        ),
        CommunicationMessage(
            message_id="3",
            source_message_id="src-3",
            sent_at="2026-04-22T14:56:22Z",
            text="After same second",
        ),
    ]

    provider = SourceWindowContextProvider.from_messages(
        messages,
        message_channels={m.message_id: "chan-a" for m in messages},
    )

    envelope = provider.get_context_envelope(target)

    assert [m.message_id for m in envelope.same_channel_before] == ["1"]
    assert [m.message_id for m in envelope.same_channel_after] == ["3"]


def test_provider_anchors_same_timestamp_target_by_source_message_id():
    target = CommunicationItem(
        doc_id="attachment-3",
        source_message_id="src-2",
        channel_id="chan-a",
        sent_at="2026-04-22T14:56:22Z",
    )
    messages = [
        CommunicationMessage(
            message_id="1",
            source_message_id="src-1",
            sent_at="2026-04-22T14:56:22Z",
            text="Before same second",
        ),
        CommunicationMessage(
            message_id="2",
            source_message_id="src-2",
            sent_at="2026-04-22T14:56:22Z",
            text="Target source",
        ),
        CommunicationMessage(
            message_id="3",
            source_message_id="src-3",
            sent_at="2026-04-22T14:56:22Z",
            text="After same second",
        ),
    ]

    provider = SourceWindowContextProvider.from_messages(
        messages,
        message_channels={m.message_id: "chan-a" for m in messages},
    )

    envelope = provider.get_context_envelope(target)

    assert [m.message_id for m in envelope.same_channel_before] == ["1"]
    assert [m.message_id for m in envelope.same_channel_after] == ["3"]


def test_provider_uses_deterministic_insertion_when_target_has_no_message_ids():
    target = CommunicationItem(
        doc_id="attachment-4",
        channel_id="chan-a",
        sent_at="2026-04-22T14:56:22Z",
    )
    messages = [
        CommunicationMessage(
            message_id="1",
            source_message_id="src-1",
            sent_at="2026-04-22T14:56:22Z",
            text="Same second first",
        ),
        CommunicationMessage(
            message_id="2",
            source_message_id="src-2",
            sent_at="2026-04-22T14:56:22Z",
            text="Same second second",
        ),
    ]

    provider = SourceWindowContextProvider.from_messages(
        messages,
        message_channels={m.message_id: "chan-a" for m in messages},
    )

    envelope = provider.get_context_envelope(target)

    assert envelope.same_channel_before == []
    assert [m.message_id for m in envelope.same_channel_after] == ["1", "2"]


def test_provider_does_not_cross_source_or_thread_boundaries():
    target = CommunicationItem(
        doc_id="attachment-5",
        origin_source="zoho_cliq",
        message_id="2",
        channel_id="chan-a",
        thread_id="thread-a",
        sent_at="2026-04-22T14:56:22Z",
    )
    messages = [
        CommunicationMessage(
            message_id="1",
            sent_at="2026-04-22T14:56:22Z",
            text="Same source and thread before",
        ),
        CommunicationMessage(
            message_id="2",
            sent_at="2026-04-22T14:56:22Z",
            text="Target message",
        ),
        CommunicationMessage(
            message_id="3",
            sent_at="2026-04-22T14:56:22Z",
            text="Same source and thread after",
        ),
        CommunicationMessage(
            message_id="4",
            sent_at="2026-04-22T14:56:22Z",
            text="Other source",
        ),
        CommunicationMessage(
            message_id="5",
            sent_at="2026-04-22T14:56:22Z",
            text="Other thread",
        ),
    ]

    provider = SourceWindowContextProvider.from_messages(
        messages,
        message_channels={m.message_id: "chan-a" for m in messages},
        message_sources={
            "1": "zoho_cliq",
            "2": "zoho_cliq",
            "3": "zoho_cliq",
            "4": "slack",
            "5": "zoho_cliq",
        },
        message_threads={
            "1": "thread-a",
            "2": "thread-a",
            "3": "thread-a",
            "4": "thread-a",
            "5": "thread-b",
        },
    )

    envelope = provider.get_context_envelope(target)

    assert [m.message_id for m in envelope.same_channel_before] == ["1"]
    assert [m.message_id for m in envelope.same_channel_after] == ["3"]
