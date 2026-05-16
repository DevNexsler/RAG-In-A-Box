from pathlib import Path

from communication_context import (
    CommunicationItem,
    CommunicationMessage,
    ContextEnvelope,
    SourceWindowContextProvider,
    communication_item_from_record,
    communication_item_from_sidecar,
    envelope_metadata,
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


def test_record_aliases_become_communication_item():
    item = communication_item_from_record(
        {
            "doc_id": "documents::photo",
            "rel_path": "photo.jpg",
            "source_name": "documents",
            "source_type": "img",
        },
        {
            "origin_source": "chat",
            "source_channel_id": "chan-1",
            "timestamp": "2026-01-01T10:00:01Z",
            "message": {
                "id": "msg-2",
                "source_message_id": "src-msg-2",
            },
            "thread": "thread-1",
            "sender_name": "Joycelyn",
            "batch": "batch-1",
            "attachment": 2,
            "sidecar": "photo.json",
        },
        primary_text="image description",
    )

    assert item is not None
    assert item.origin_source == "chat"
    assert item.channel_id == "chan-1"
    assert item.sent_at == "2026-01-01T10:00:01Z"
    assert item.message_id == "msg-2"
    assert item.source_message_id == "src-msg-2"
    assert item.thread_id == "thread-1"
    assert item.sender == "Joycelyn"
    assert item.batch_key == "batch-1"
    assert item.attachment_index == "2"
    assert item.sidecar_path == "photo.json"
    assert item.primary_text == "image description"


def test_record_message_alias_accepts_scalar_message_id():
    item = communication_item_from_record(
        {"doc_id": "documents::message"},
        {"message": "msg-1"},
    )

    assert item is not None
    assert item.message_id == "msg-1"


def test_record_non_scalar_sender_alone_is_not_communication_item():
    item = communication_item_from_record(
        {"doc_id": "documents::metadata"},
        {"sender": ["x"]},
    )

    assert item is None


def test_record_dict_sender_accepts_scalar_name():
    item = communication_item_from_record(
        {"doc_id": "documents::metadata"},
        {"sender": {"name": "Joycelyn"}},
    )

    assert item is not None
    assert item.sender == "Joycelyn"


def test_record_dict_sender_rejects_non_scalar_name():
    item = communication_item_from_record(
        {"doc_id": "documents::metadata"},
        {"sender": {"name": ["Joycelyn"]}},
    )

    assert item is None


def test_envelope_metadata_preserves_message_and_source_ids():
    envelope = ContextEnvelope(
        primary_item=CommunicationItem(doc_id="documents::photo"),
        same_channel_before=[
            CommunicationMessage(
                message_id="m1",
                source_message_id="src-m1",
                text="Before one",
            ),
            CommunicationMessage(
                message_id="m2",
                source_message_id="src-m2",
                text="Before two",
            ),
        ],
        same_channel_after=[
            CommunicationMessage(
                message_id="m3",
                source_message_id="src-m3",
                text="After one",
            )
        ],
        nearest_nonempty_before=CommunicationMessage(
            message_id="m2",
            source_message_id="src-m2",
            text="Before two",
        ),
        nearest_nonempty_after=CommunicationMessage(
            message_id="m3",
            source_message_id="src-m3",
            text="After one",
        ),
    )

    metadata = envelope_metadata(envelope)

    assert metadata == {
        "context_before_message_ids": "m1,m2",
        "context_before_source_message_ids": "src-m1,src-m2",
        "context_after_message_ids": "m3",
        "context_after_source_message_ids": "src-m3",
        "context_nearest_before_message_id": "m2",
        "context_nearest_before_source_message_id": "src-m2",
        "context_nearest_after_message_id": "m3",
        "context_nearest_after_source_message_id": "src-m3",
    }


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


def test_provider_prunes_farther_context_candidates_by_time_distance():
    target = CommunicationItem(
        doc_id="photo",
        origin_source="zoho_cliq",
        channel_id="renovations",
        sent_at="2026-05-08T19:00:56Z",
    )
    messages = [
        CommunicationMessage(
            message_id="old",
            origin_source="zoho_cliq",
            channel_id="renovations",
            sent_at="2026-05-05T19:33:06Z",
            text="665 sayre",
        ),
        CommunicationMessage(
            message_id="before",
            origin_source="zoho_cliq",
            channel_id="renovations",
            sent_at="2026-05-08T19:00:09Z",
            text="482 # 6 as follow.....",
        ),
        CommunicationMessage(
            message_id="after",
            origin_source="zoho_cliq",
            channel_id="renovations",
            sent_at="2026-05-08T19:04:16Z",
            text="163 washington as follow....",
        ),
    ]

    provider = SourceWindowContextProvider.from_messages(
        messages,
        message_channels={m.message_id: "renovations" for m in messages},
        message_sources={m.message_id: "zoho_cliq" for m in messages},
        window_before=5,
        window_after=5,
    )

    envelope = provider.get_context_envelope(target)

    assert [m.message_id for m in envelope.same_channel_before] == ["before"]
    assert envelope.same_channel_after == []


def test_provider_keeps_balanced_before_and_after_candidates():
    target = CommunicationItem(
        doc_id="photo",
        origin_source="zoho_cliq",
        channel_id="renovations",
        sent_at="2026-05-08T19:00:56Z",
    )
    messages = [
        CommunicationMessage(
            message_id="before",
            origin_source="zoho_cliq",
            channel_id="renovations",
            sent_at="2026-05-08T19:00:50Z",
            text="Building A",
        ),
        CommunicationMessage(
            message_id="after",
            origin_source="zoho_cliq",
            channel_id="renovations",
            sent_at="2026-05-08T19:01:03Z",
            text="Unit 2",
        ),
    ]

    provider = SourceWindowContextProvider.from_messages(
        messages,
        message_channels={m.message_id: "renovations" for m in messages},
        message_sources={m.message_id: "zoho_cliq" for m in messages},
        window_before=5,
        window_after=5,
    )

    envelope = provider.get_context_envelope(target)

    assert [m.message_id for m in envelope.same_channel_before] == ["before"]
    assert [m.message_id for m in envelope.same_channel_after] == ["after"]


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


def test_build_context_provider_disabled_config_returns_none():
    from communication_context import build_context_provider_from_records
    from sources.base import SourceRecord

    provider = build_context_provider_from_records(
        [
            {
                "doc_id": "comm::1",
                "source_name": "comm",
                "source_type": "pg_message",
            }
        ],
        {
            "comm::1": SourceRecord(
                doc_id="1",
                source_type="pg_message",
                natural_key="zoho/1",
                mtime=1.0,
                size=6,
                metadata={
                    "_text": "Unit E",
                    "source_message_id": "s1",
                    "channel_id": "chan",
                },
            )
        },
        {"enabled": False},
    )

    assert provider is None


def test_build_context_provider_supports_source_message_id_only_records():
    from communication_context import (
        build_context_provider_from_records,
        communication_item_from_record,
    )
    from sources.base import SourceRecord

    records = [
        {"doc_id": "comm::s1", "source_name": "comm", "source_type": "pg_message"},
        {"doc_id": "comm::s2", "source_name": "comm", "source_type": "pg_message"},
    ]
    source_records = {
        "comm::s1": SourceRecord(
            doc_id="s1",
            source_type="pg_message",
            natural_key="zoho/s1",
            mtime=1.0,
            size=6,
            metadata={
                "_text": "Unit E",
                "source": "zoho_cliq",
                "source_message_id": "s1",
                "channel_id": "chan",
                "sent_at": "2026-01-01T10:00:00Z",
            },
        ),
        "comm::s2": SourceRecord(
            doc_id="s2",
            source_type="pg_message",
            natural_key="zoho/s2",
            mtime=2.0,
            size=9,
            metadata={
                "_text": "Also this",
                "source": "zoho_cliq",
                "source_message_id": "s2",
                "channel_id": "chan",
                "sent_at": "2026-01-01T10:00:10Z",
            },
        ),
    }

    provider = build_context_provider_from_records(records, source_records, {})
    assert provider is not None
    item = communication_item_from_record(
        records[1],
        source_records["comm::s2"].metadata,
    )
    assert item is not None

    envelope = provider.get_context_envelope(item)

    assert envelope.nearest_nonempty_before.text == "Unit E"
    assert [m.source_message_id for m in envelope.same_channel_before] == ["s1"]


def test_build_context_provider_matches_postgres_channel_source_id_to_sidecar():
    from communication_context import (
        build_context_provider_from_records,
        communication_item_from_record,
    )
    from sources.base import SourceRecord

    records = [
        {
            "doc_id": "comm_messages::zoho_cliq/label-1",
            "source_name": "comm_messages",
            "source_type": "pg_message",
        }
    ]
    source_records = {
        "comm_messages::zoho_cliq/label-1": SourceRecord(
            doc_id="zoho_cliq/label-1",
            source_type="pg_message",
            natural_key="zoho_cliq/label-1",
            mtime=1.0,
            size=10,
            metadata={
                "source": "zoho_cliq",
                "source_message_id": "label-1",
                "source_channel_id": "CT_renovations",
                "channel_name": "#Renovations",
                "sent_at": "2026-05-08T19:00:09Z",
                "sender": "Cesar",
                "_text": "482 # 6 as follow.....",
            },
        )
    }

    provider = build_context_provider_from_records(records, source_records, {})
    item = communication_item_from_record(
        {"doc_id": "documents::photo", "source_type": "img"},
        {
            "source": "zoho_cliq",
            "source_message_id": "photo-1",
            "source_channel_id": "CT_renovations",
            "sent_at": "2026-05-08T19:00:56Z",
        },
    )

    envelope = provider.get_context_envelope(item)

    assert [m.source_message_id for m in envelope.same_channel_before] == ["label-1"]
    assert envelope.same_channel_before[0].text == "482 # 6 as follow....."


def test_build_context_provider_distinguishes_source_and_message_id_collisions():
    from communication_context import (
        build_context_provider_from_records,
        communication_item_from_record,
    )
    from sources.base import SourceRecord

    records = [
        {"doc_id": "comm::a", "source_name": "comm", "source_type": "pg_message"},
        {"doc_id": "comm::b", "source_name": "comm", "source_type": "pg_message"},
    ]
    source_records = {
        "comm::a": SourceRecord(
            doc_id="a",
            source_type="pg_message",
            natural_key="a",
            mtime=1.0,
            size=9,
            metadata={
                "_text": "Channel A",
                "source": "origin-a",
                "source_message_id": "same",
                "message_id": "a-id",
                "channel_id": "chan-a",
                "sent_at": "2026-01-01T10:00:00Z",
            },
        ),
        "comm::b": SourceRecord(
            doc_id="b",
            source_type="pg_message",
            natural_key="b",
            mtime=2.0,
            size=9,
            metadata={
                "_text": "Channel B",
                "source": "origin-b",
                "source_message_id": "b-src",
                "message_id": "same",
                "channel_id": "chan-b",
                "sent_at": "2026-01-01T10:00:10Z",
            },
        ),
    }

    provider = build_context_provider_from_records(records, source_records, {})
    assert provider is not None
    item = communication_item_from_record(
        records[1],
        source_records["comm::b"].metadata,
    )
    assert item is not None

    envelope = provider.get_context_envelope(item)

    assert envelope.same_channel_before == []
    assert envelope.nearest_nonempty_before is None


def test_build_context_provider_distinguishes_source_message_ids_by_origin():
    from communication_context import (
        build_context_provider_from_records,
        communication_item_from_record,
    )
    from sources.base import SourceRecord

    records = [
        {"doc_id": "comm::a", "source_name": "comm", "source_type": "pg_message"},
        {"doc_id": "comm::b", "source_name": "comm", "source_type": "pg_message"},
        {"doc_id": "comm::target", "source_name": "comm", "source_type": "pg_message"},
    ]
    source_records = {
        "comm::a": SourceRecord(
            doc_id="a",
            source_type="pg_message",
            natural_key="a",
            mtime=1.0,
            size=9,
            metadata={
                "_text": "Origin A",
                "source": "origin-a",
                "source_message_id": "same",
                "message_id": "a-id",
                "channel_id": "chan-a",
                "sent_at": "2026-01-01T10:00:00Z",
            },
        ),
        "comm::b": SourceRecord(
            doc_id="b",
            source_type="pg_message",
            natural_key="b",
            mtime=2.0,
            size=9,
            metadata={
                "_text": "Origin B",
                "source": "origin-b",
                "source_message_id": "same",
                "message_id": "b-id",
                "channel_id": "chan-b",
                "sent_at": "2026-01-01T10:00:10Z",
            },
        ),
        "comm::target": SourceRecord(
            doc_id="target",
            source_type="pg_message",
            natural_key="target",
            mtime=3.0,
            size=6,
            metadata={
                "_text": "Target",
                "source": "origin-b",
                "source_message_id": "target-src",
                "message_id": "target-id",
                "channel_id": "chan-b",
                "sent_at": "2026-01-01T10:00:20Z",
            },
        ),
    }

    provider = build_context_provider_from_records(records, source_records, {})
    assert provider is not None
    item = communication_item_from_record(
        records[2],
        source_records["comm::target"].metadata,
    )
    assert item is not None

    envelope = provider.get_context_envelope(item)

    assert [m.text for m in envelope.same_channel_before] == ["Origin B"]


def test_build_context_provider_distinguishes_source_message_ids_by_channel():
    from communication_context import (
        build_context_provider_from_records,
        communication_item_from_record,
    )
    from sources.base import SourceRecord

    records = [
        {"doc_id": "comm::a", "source_name": "comm", "source_type": "pg_message"},
        {"doc_id": "comm::b", "source_name": "comm", "source_type": "pg_message"},
        {"doc_id": "comm::target", "source_name": "comm", "source_type": "pg_message"},
    ]
    source_records = {
        "comm::a": SourceRecord(
            doc_id="a",
            source_type="pg_message",
            natural_key="a",
            mtime=1.0,
            size=9,
            metadata={
                "_text": "Channel A",
                "source": "origin",
                "source_message_id": "same",
                "message_id": "a-id",
                "channel_id": "chan-a",
                "sent_at": "2026-01-01T10:00:00Z",
            },
        ),
        "comm::b": SourceRecord(
            doc_id="b",
            source_type="pg_message",
            natural_key="b",
            mtime=2.0,
            size=9,
            metadata={
                "_text": "Channel B",
                "source": "origin",
                "source_message_id": "same",
                "message_id": "b-id",
                "channel_id": "chan-b",
                "sent_at": "2026-01-01T10:00:10Z",
            },
        ),
        "comm::target": SourceRecord(
            doc_id="target",
            source_type="pg_message",
            natural_key="target",
            mtime=3.0,
            size=6,
            metadata={
                "_text": "Target",
                "source": "origin",
                "source_message_id": "target-src",
                "message_id": "target-id",
                "channel_id": "chan-b",
                "sent_at": "2026-01-01T10:00:20Z",
            },
        ),
    }

    provider = build_context_provider_from_records(records, source_records, {})
    assert provider is not None
    item = communication_item_from_record(
        records[2],
        source_records["comm::target"].metadata,
    )
    assert item is not None

    envelope = provider.get_context_envelope(item)

    assert [m.text for m in envelope.same_channel_before] == ["Channel B"]


def test_build_context_provider_excludes_records_without_message_identity():
    from communication_context import build_context_provider_from_records
    from sources.base import SourceRecord

    provider = build_context_provider_from_records(
        [
            {
                "doc_id": "comm::no-id",
                "source_name": "comm",
                "source_type": "pg_message",
            }
        ],
        {
            "comm::no-id": SourceRecord(
                doc_id="no-id",
                source_type="pg_message",
                natural_key="no-id",
                mtime=1.0,
                size=7,
                metadata={
                    "_text": "No ids",
                    "source": "origin",
                    "channel_id": "chan",
                    "sent_at": "2026-01-01T10:00:00Z",
                },
            )
        },
        {},
    )

    assert provider is None
