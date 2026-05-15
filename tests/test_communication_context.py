from pathlib import Path

from communication_context import communication_item_from_sidecar


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
    assert item.batch_key.startswith("zoho_cliq:2242125288797599446:")
