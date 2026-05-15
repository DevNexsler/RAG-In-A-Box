from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CommunicationMessage:
    message_id: str = ""
    source_message_id: str = ""
    sender: str = ""
    sent_at: str = ""
    text: str = ""


@dataclass(frozen=True)
class CommunicationItem:
    doc_id: str = ""
    rel_path: str = ""
    source_name: str = ""
    source_type: str = ""
    origin_source: str = ""
    message_id: str = ""
    source_message_id: str = ""
    channel_id: str = ""
    channel_name: str = ""
    thread_id: str = ""
    sender: str = ""
    sent_at: str = ""
    batch_key: str = ""
    attachment_index: str = ""
    sidecar_path: str = ""
    primary_text: str = ""


@dataclass(frozen=True)
class ContextEnvelope:
    primary_item: CommunicationItem
    same_channel_before: list[CommunicationMessage] = field(default_factory=list)
    same_channel_after: list[CommunicationMessage] = field(default_factory=list)
    nearest_nonempty_before: CommunicationMessage | None = None
    nearest_nonempty_after: CommunicationMessage | None = None
    same_batch: list[CommunicationItem] = field(default_factory=list)


def communication_item_from_sidecar(
    media_path: Path, sidecar_path: Path
) -> CommunicationItem:
    payload: dict[str, Any] = json.loads(sidecar_path.read_text())
    message = payload.get("message") or {}
    channel = payload.get("channel") or {}
    media = payload.get("media") or {}

    origin_source = _text(payload.get("source"))
    sent_at = _text(message.get("sent_at"))
    channel_id = _text(channel.get("source_channel_id"))

    return CommunicationItem(
        doc_id=str(media_path),
        rel_path=str(media_path),
        origin_source=origin_source,
        message_id=_text(message.get("message_id")),
        source_message_id=_text(message.get("source_message_id")),
        channel_id=channel_id,
        thread_id=_text(channel.get("thread_id")),
        sender=_text((message.get("from") or {}).get("name")),
        sent_at=sent_at,
        batch_key=_batch_key(origin_source, channel_id, sent_at),
        attachment_index=_text(media.get("media_index"), default="0"),
        sidecar_path=str(sidecar_path),
    )


def _batch_key(origin_source: str, channel_id: str, sent_at: str) -> str:
    rounded_sent_at = _round_timestamp_to_utc_second(sent_at)
    if not (origin_source or channel_id or rounded_sent_at):
        return ""
    return f"{origin_source}:{channel_id}:{rounded_sent_at}"


def _round_timestamp_to_utc_second(value: str) -> str:
    if not value:
        return ""
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc).replace(microsecond=0)
    return timestamp.isoformat().replace("+00:00", "Z")


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()
