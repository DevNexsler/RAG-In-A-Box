from __future__ import annotations

from collections.abc import Iterable
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


class SourceWindowContextProvider:
    def __init__(
        self,
        messages_by_scope: dict[tuple[str, str, str], list[CommunicationMessage]],
        *,
        window_before: int = 5,
        window_after: int = 5,
    ) -> None:
        self._messages_by_scope = {
            scope: sorted(messages, key=_message_sort_key)
            for scope, messages in messages_by_scope.items()
        }
        self._window_before = window_before
        self._window_after = window_after

    @classmethod
    def from_messages(
        cls,
        messages: list[CommunicationMessage],
        *,
        message_channels: dict[str, str],
        message_sources: dict[str, str] | None = None,
        message_threads: dict[str, str] | None = None,
        window_before: int = 5,
        window_after: int = 5,
    ) -> SourceWindowContextProvider:
        messages_by_scope: dict[tuple[str, str, str], list[CommunicationMessage]] = {}
        for message in messages:
            message_key = message.message_id
            scope = (
                _text((message_sources or {}).get(message_key)),
                _text(message_channels.get(message_key)),
                _text((message_threads or {}).get(message_key)),
            )
            messages_by_scope.setdefault(scope, []).append(message)

        return cls(
            messages_by_scope,
            window_before=window_before,
            window_after=window_after,
        )

    def get_context_envelope(self, item: CommunicationItem) -> ContextEnvelope:
        scope = (_text(item.origin_source), _text(item.channel_id), _text(item.thread_id))
        messages = self._messages_by_scope.get(scope, [])
        item_key = _context_item_sort_key(item, messages)
        before = [
            message
            for message in messages
            if _message_sort_key(message) < item_key and not _is_context_target(message, item)
        ]
        after = [
            message
            for message in messages
            if _message_sort_key(message) > item_key and not _is_context_target(message, item)
        ]
        before_window = before[-self._window_before :] if self._window_before > 0 else []
        after_window = after[: self._window_after] if self._window_after > 0 else []

        return ContextEnvelope(
            primary_item=item,
            same_channel_before=before_window,
            same_channel_after=after_window,
            nearest_nonempty_before=_nearest_nonempty(reversed(before)),
            nearest_nonempty_after=_nearest_nonempty(after),
        )


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
        sender=_sender_name(message),
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


def _message_sort_key(message: CommunicationMessage) -> tuple[str, str, str, str]:
    return (*_sent_at_sort_key(message.sent_at), message.message_id, message.source_message_id)


def _context_item_sort_key(
    item: CommunicationItem,
    messages: list[CommunicationMessage],
) -> tuple[str, str, str, str]:
    for message in messages:
        if _is_context_target(message, item):
            return _message_sort_key(message)
    return (*_sent_at_sort_key(item.sent_at), item.message_id, item.source_message_id)


def _is_context_target(message: CommunicationMessage, item: CommunicationItem) -> bool:
    if item.message_id and message.message_id == item.message_id:
        return True
    return bool(
        item.source_message_id
        and message.source_message_id
        and message.source_message_id == item.source_message_id
    )


def _sent_at_sort_key(value: str) -> tuple[str, str]:
    return (_normalize_timestamp_for_sort(value), _text(value))


def _normalize_timestamp_for_sort(value: str) -> str:
    if not value:
        return ""
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)
    return timestamp.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _nearest_nonempty(messages: Iterable[CommunicationMessage]) -> CommunicationMessage | None:
    for message in messages:
        if _text(message.text):
            return message
    return None


def _sender_name(message: dict[str, Any]) -> str:
    sender = message.get("from")
    if not isinstance(sender, dict):
        return ""
    return _text(sender.get("name"))


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()
