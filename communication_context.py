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
    origin_source: str = ""
    channel_id: str = ""
    thread_id: str = ""


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
        max_time_window_minutes: float = 15,
        max_relative_gap: float = 3.0,
        max_extra_gap_seconds: int = 60,
    ) -> None:
        self._messages_by_scope = {
            scope: sorted(messages, key=_message_sort_key)
            for scope, messages in messages_by_scope.items()
        }
        self._window_before = window_before
        self._window_after = window_after
        self._max_time_window_seconds = max(
            0.0, float(max_time_window_minutes or 0)
        ) * 60.0
        self._max_relative_gap = max(1.0, float(max_relative_gap or 1.0))
        self._max_extra_gap_seconds = max(0, int(max_extra_gap_seconds or 0))

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
        max_time_window_minutes: float = 15,
        max_relative_gap: float = 3.0,
        max_extra_gap_seconds: int = 60,
    ) -> SourceWindowContextProvider:
        messages_by_scope: dict[tuple[str, str, str], list[CommunicationMessage]] = {}
        message_sources = message_sources or {}
        message_threads = message_threads or {}
        for message in messages:
            origin_source = (
                _lookup_message_map_value(
                    message_sources,
                    message,
                    message.origin_source,
                    message.channel_id,
                    message.thread_id,
                )
                or message.origin_source
            )
            scope = (
                _text(origin_source),
                _lookup_message_map_value(
                    message_channels,
                    message,
                    origin_source,
                    message.channel_id,
                    message.thread_id,
                )
                or _text(message.channel_id),
                _lookup_message_map_value(
                    message_threads,
                    message,
                    origin_source,
                    message.channel_id,
                    message.thread_id,
                )
                or _text(message.thread_id),
            )
            messages_by_scope.setdefault(scope, []).append(message)

        return cls(
            messages_by_scope,
            window_before=window_before,
            window_after=window_after,
            max_time_window_minutes=max_time_window_minutes,
            max_relative_gap=max_relative_gap,
            max_extra_gap_seconds=max_extra_gap_seconds,
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
        before = _messages_within_time_window(
            item, before, self._max_time_window_seconds
        )
        after = _messages_within_time_window(
            item, after, self._max_time_window_seconds
        )
        nearest_before = _nearest_nonempty(reversed(before))
        nearest_after = _nearest_nonempty(after)
        before_window = before[-self._window_before :] if self._window_before > 0 else []
        after_window = after[: self._window_after] if self._window_after > 0 else []
        before_window, after_window = _prune_by_time_distance(
            item,
            before_window,
            after_window,
            max_relative_gap=self._max_relative_gap,
            max_extra_gap_seconds=self._max_extra_gap_seconds,
        )
        if self._window_before > 0:
            before_window = _include_context_message(before_window, nearest_before)
        if self._window_after > 0:
            after_window = _include_context_message(after_window, nearest_after)

        return ContextEnvelope(
            primary_item=item,
            same_channel_before=before_window,
            same_channel_after=after_window,
            nearest_nonempty_before=nearest_before,
            nearest_nonempty_after=nearest_after,
        )


def communication_item_from_sidecar(
    media_path: Path, sidecar_path: Path
) -> CommunicationItem:
    payload: dict[str, Any] = json.loads(sidecar_path.read_text())
    return _communication_item_from_sidecar_payload(media_path, sidecar_path, payload)


def communication_metadata_from_sidecar(
    media_path: Path, sidecar_path: Path
) -> dict[str, str]:
    """Return index metadata aliases extracted from a communication sidecar."""
    payload: dict[str, Any] = json.loads(sidecar_path.read_text())
    item = _communication_item_from_sidecar_payload(media_path, sidecar_path, payload)
    message = payload.get("message") or {}
    media = payload.get("media") or {}

    metadata = {
        "source": item.origin_source,
        "origin_source": item.origin_source,
        "message_id": item.message_id,
        "source_message_id": item.source_message_id,
        "channel_id": item.channel_id,
        "source_channel_id": item.channel_id,
        "thread_id": item.thread_id,
        "sender": item.sender,
        "sent_at": item.sent_at,
        "batch_key": item.batch_key,
        "attachment_index": item.attachment_index,
        "sidecar_path": item.sidecar_path,
        "message_body": _text(message.get("body")),
        "media_type": _text(media.get("media_type")),
        "original_filename": _text(media.get("original_filename")),
    }
    return {key: value for key, value in metadata.items() if value}


def _communication_item_from_sidecar_payload(
    media_path: Path,
    sidecar_path: Path,
    payload: dict[str, Any],
) -> CommunicationItem:
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


def communication_item_from_record(
    doc: dict,
    metadata: dict,
    primary_text: str = "",
) -> CommunicationItem | None:
    """Build a communication item from generic source metadata aliases."""
    metadata = metadata if isinstance(metadata, dict) else {}
    origin_source = _first_metadata_text(
        metadata,
        ("source", "origin_source"),
        nested_keys=("source", "origin_source", "id"),
    )
    channel_id = _first_metadata_text(
        metadata,
        ("channel_id", "source_channel_id", "channel"),
        nested_keys=("channel_id", "source_channel_id", "id"),
    )
    sent_at = _first_metadata_text(metadata, ("sent_at", "created_at", "timestamp"))
    message_id = _first_metadata_text(
        metadata,
        ("message_id", "message"),
        nested_keys=("message_id", "id"),
    )
    source_message_id = _first_metadata_text(
        metadata,
        ("source_message_id", "message"),
        nested_keys=("source_message_id",),
        scalar_keys=("source_message_id",),
    )
    thread_id = _first_metadata_text(
        metadata,
        ("thread_id", "source_thread_id", "thread"),
        nested_keys=("thread_id", "source_thread_id", "id"),
    )
    sender = _sender_from_metadata(metadata)
    batch_key = _first_metadata_text(
        metadata, ("batch_key", "batch"), nested_keys=("batch_key", "id")
    )
    attachment_index = _first_metadata_text(
        metadata,
        ("attachment_index", "attachment"),
        nested_keys=("attachment_index", "index", "id"),
    )
    sidecar_path = _first_metadata_text(
        metadata, ("sidecar_path", "sidecar"), nested_keys=("sidecar_path", "path")
    )

    identity_values = (
        origin_source,
        channel_id,
        sent_at,
        message_id,
        source_message_id,
        thread_id,
        sender,
        batch_key,
        attachment_index,
        sidecar_path,
    )
    if not any(identity_values):
        return None

    return CommunicationItem(
        doc_id=_text(doc.get("doc_id")),
        rel_path=_text(doc.get("rel_path")),
        source_name=_text(doc.get("source_name")),
        source_type=_text(doc.get("source_type")),
        origin_source=origin_source,
        message_id=message_id,
        source_message_id=source_message_id,
        channel_id=channel_id,
        channel_name=_first_metadata_text(metadata, ("channel_name",)),
        thread_id=thread_id,
        sender=sender,
        sent_at=sent_at,
        batch_key=batch_key,
        attachment_index=attachment_index,
        sidecar_path=sidecar_path,
        primary_text=primary_text,
    )


def build_context_provider_from_records(
    records: Iterable[dict[str, Any]],
    source_records_by_ns_doc_id: dict[str, Any],
    config: dict[str, Any] | None,
) -> SourceWindowContextProvider | None:
    """Build an in-memory communication context provider from scanned records."""
    config = config if isinstance(config, dict) else {}
    if config.get("enabled") is False:
        return None

    messages: list[CommunicationMessage] = []
    message_channels: dict[str, str] = {}
    message_sources: dict[str, str] = {}
    message_threads: dict[str, str] = {}

    for doc in records:
        if not isinstance(doc, dict):
            continue
        doc_id = _text(doc.get("doc_id"))
        if not doc_id:
            continue
        record = source_records_by_ns_doc_id.get(doc_id)
        if record is None:
            continue
        metadata = getattr(record, "metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}

        primary_text = _record_primary_text(doc, metadata)
        item = communication_item_from_record(doc, metadata, primary_text)
        if item is None:
            continue

        message = CommunicationMessage(
            message_id=item.message_id,
            source_message_id=item.source_message_id,
            sender=item.sender,
            sent_at=item.sent_at,
            text=_text(metadata.get("_text")) or _text(item.primary_text),
            origin_source=item.origin_source,
            channel_id=item.channel_id,
            thread_id=item.thread_id,
        )
        if not _record_message_has_context_scope(item, message):
            continue

        messages.append(message)
        for key in _scoped_message_map_keys(
            message,
            item.origin_source,
            item.channel_id,
            item.thread_id,
        ):
            message_channels[key] = item.channel_id
            message_sources[key] = item.origin_source
            message_threads[key] = item.thread_id

    if not messages:
        return None

    return SourceWindowContextProvider.from_messages(
        messages,
        message_channels=message_channels,
        message_sources=message_sources,
        message_threads=message_threads,
        window_before=_configured_window(config, "window_before", 5),
        window_after=_configured_window(config, "window_after", 5),
        max_time_window_minutes=_configured_float(
            config, "max_time_window_minutes", 15.0
        ),
        max_relative_gap=_configured_float(config, "max_relative_gap", 3.0),
        max_extra_gap_seconds=_configured_window(
            config, "max_extra_gap_seconds", 60
        ),
    )


def format_context_envelope_for_prompt(envelope: ContextEnvelope) -> str:
    """Format nonempty nearby messages for the enrichment prompt."""
    sections: list[str] = []
    for label, messages in (
        ("BEFORE", envelope.same_channel_before),
        ("AFTER", envelope.same_channel_after),
    ):
        lines = [_format_context_message(label, message) for message in messages]
        lines = [line for line in lines if line]
        if lines:
            sections.append(f"{label} MESSAGES")
            sections.extend(lines)
    return "\n".join(sections)


def envelope_metadata(envelope: ContextEnvelope) -> dict[str, str]:
    """Return raw provenance fields for stored metadata."""
    nearest_before = envelope.nearest_nonempty_before or _nearest_nonempty(
        reversed(envelope.same_channel_before)
    )
    nearest_after = envelope.nearest_nonempty_after or _nearest_nonempty(
        envelope.same_channel_after
    )
    metadata = {
        "context_before_message_ids": _message_ids(envelope.same_channel_before),
        "context_before_source_message_ids": _source_message_ids(
            envelope.same_channel_before
        ),
        "context_after_message_ids": _message_ids(envelope.same_channel_after),
        "context_after_source_message_ids": _source_message_ids(
            envelope.same_channel_after
        ),
        "context_nearest_before_message_id": _message_id(nearest_before),
        "context_nearest_before_source_message_id": _source_message_id(
            nearest_before
        ),
        "context_nearest_after_message_id": _message_id(nearest_after),
        "context_nearest_after_source_message_id": _source_message_id(nearest_after),
    }
    return {key: value for key, value in metadata.items() if value}


def repair_sidecar_context(
    sidecar_path: Path | str,
    envelope: ContextEnvelope,
    *,
    generated_at: str | None = None,
) -> bool:
    """Merge same-channel context into an attachment sidecar without data loss."""
    sidecar_path = Path(sidecar_path)
    payload = json.loads(sidecar_path.read_text())
    if not isinstance(payload, dict):
        return False

    existing_context = payload.get("context")
    context = dict(existing_context) if isinstance(existing_context, dict) else {}
    if existing_context is not None and not isinstance(existing_context, dict):
        context["legacy_context"] = existing_context
    context.update(_sidecar_context_payload(envelope, generated_at=generated_at))

    updated_payload = dict(payload)
    updated_payload["context"] = context
    if _sidecar_stable_payload(updated_payload) == _sidecar_stable_payload(payload):
        return False

    rendered = json.dumps(updated_payload, indent=2, ensure_ascii=False) + "\n"
    tmp_path = sidecar_path.with_name(f".{sidecar_path.name}.tmp")
    try:
        tmp_path.write_text(rendered)
        tmp_path.replace(sidecar_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return True


def _message_from_sidecar_entry(entry: Any) -> CommunicationMessage:
    if not isinstance(entry, dict):
        return CommunicationMessage()
    return CommunicationMessage(
        message_id=_text(entry.get("message_id")),
        source_message_id=_text(entry.get("source_message_id")),
        sender=_text(entry.get("sender")),
        sent_at=_text(entry.get("sent_at")),
        text=_text(entry.get("text")),
        origin_source=_text(entry.get("origin_source")),
        channel_id=_text(entry.get("channel_id")),
        thread_id=_text(entry.get("thread_id")),
    )


def _sidecar_nearest_message(entry: Any) -> CommunicationMessage | None:
    if not isinstance(entry, dict):
        return None
    message = _message_from_sidecar_entry(entry)
    return message if _text(message.text) else None


def context_envelope_from_sidecar_payload(
    payload: Any,
    item: CommunicationItem,
    *,
    max_time_window_minutes: float = 15,
) -> ContextEnvelope:
    """Reconstruct a ContextEnvelope from an attachment sidecar's stored
    `context` block (written earlier by repair_sidecar_context on a full sweep).

    Lets the targeted single-document index path reuse already-computed
    same-channel context without re-scanning the whole message corpus (which is
    why that path could not build context before)."""
    context = payload.get("context") if isinstance(payload, dict) else None
    if not isinstance(context, dict):
        return ContextEnvelope(primary_item=item)
    before = [_message_from_sidecar_entry(m) for m in (context.get("same_channel_before") or [])]
    after = [_message_from_sidecar_entry(m) for m in (context.get("same_channel_after") or [])]
    max_seconds = max(0.0, float(max_time_window_minutes or 0)) * 60.0
    before = _messages_within_time_window(item, before, max_seconds)
    after = _messages_within_time_window(item, after, max_seconds)
    explicit_before = _sidecar_nearest_message(context.get("nearest_nonempty_before"))
    explicit_after = _sidecar_nearest_message(context.get("nearest_nonempty_after"))
    explicit_before = _message_within_time_window(item, explicit_before, max_seconds)
    explicit_after = _message_within_time_window(item, explicit_after, max_seconds)
    before = _include_context_message(before, explicit_before)
    after = _include_context_message(after, explicit_after)
    return ContextEnvelope(
        primary_item=item,
        same_channel_before=before,
        same_channel_after=after,
        nearest_nonempty_before=explicit_before or _nearest_nonempty(reversed(before)),
        nearest_nonempty_after=explicit_after or _nearest_nonempty(after),
    )


class SidecarContextProvider:
    """Context provider that reads a doc's OWN sidecar's stored context block.

    Cheap and per-doc — for the targeted single-document index path, which
    cannot afford the full-corpus scan the sweep's SourceWindowContextProvider
    performs. Returns an empty envelope when the item has no sidecar or no
    stored context, so non-communication docs are unaffected."""

    def __init__(self, *, max_time_window_minutes: float = 15) -> None:
        self._max_time_window_minutes = max_time_window_minutes

    def get_context_envelope(self, item: CommunicationItem) -> ContextEnvelope:
        path = getattr(item, "sidecar_path", "") or ""
        if not path:
            return ContextEnvelope(primary_item=item)
        try:
            payload = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            return ContextEnvelope(primary_item=item)
        return context_envelope_from_sidecar_payload(
            payload,
            item,
            max_time_window_minutes=self._max_time_window_minutes,
        )


def _sidecar_context_payload(
    envelope: ContextEnvelope,
    *,
    generated_at: str | None,
) -> dict[str, Any]:
    nearest_before = envelope.nearest_nonempty_before or _nearest_nonempty(
        reversed(envelope.same_channel_before)
    )
    nearest_after = envelope.nearest_nonempty_after or _nearest_nonempty(
        envelope.same_channel_after
    )
    primary = envelope.primary_item
    return {
        "schema_version": 1,
        "source": "rag-in-a-box",
        "generated_at": generated_at or _utc_now_iso(),
        "scope": _compact_dict(
            {
                "origin_source": primary.origin_source,
                "channel_id": primary.channel_id,
                "thread_id": primary.thread_id,
            }
        ),
        "same_channel_before": _sidecar_context_messages(
            envelope.same_channel_before
        ),
        "same_channel_after": _sidecar_context_messages(envelope.same_channel_after),
        "nearest_nonempty_before": _sidecar_context_message(nearest_before),
        "nearest_nonempty_after": _sidecar_context_message(nearest_after),
    }


def _sidecar_context_message(
    message: CommunicationMessage | None,
) -> dict[str, str] | None:
    if message is None:
        return None
    rendered = _compact_dict(
        {
            "message_id": message.message_id,
            "source_message_id": message.source_message_id,
            "sender": message.sender,
            "sent_at": message.sent_at,
            "text": message.text,
            "origin_source": message.origin_source,
            "channel_id": message.channel_id,
            "thread_id": message.thread_id,
        }
    )
    return rendered or None


def _sidecar_context_messages(
    messages: Iterable[CommunicationMessage],
) -> list[dict[str, str]]:
    return [
        rendered
        for message in messages
        if (rendered := _sidecar_context_message(message)) is not None
    ]


def _sidecar_stable_payload(payload: dict[str, Any]) -> dict[str, Any]:
    stable_payload = json.loads(json.dumps(payload, sort_keys=True, default=str))
    context = stable_payload.get("context")
    if isinstance(context, dict):
        context.pop("generated_at", None)
    return stable_payload


def _compact_dict(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if _text(value)}


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
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


def _prune_by_time_distance(
    item: CommunicationItem,
    before: list[CommunicationMessage],
    after: list[CommunicationMessage],
    *,
    max_relative_gap: float,
    max_extra_gap_seconds: int,
) -> tuple[list[CommunicationMessage], list[CommunicationMessage]]:
    item_time = _parse_timestamp(item.sent_at)
    if item_time is None:
        return before, after

    candidates: list[tuple[str, CommunicationMessage, float | None]] = []
    for message in before:
        candidates.append(("before", message, _distance_seconds(item_time, message)))
    for message in after:
        candidates.append(("after", message, _distance_seconds(item_time, message)))

    distances = [distance for _, _, distance in candidates if distance is not None]
    if not distances:
        return before, after

    nearest = min(distances)
    threshold = max(nearest * max_relative_gap, nearest + max_extra_gap_seconds)
    keep = {
        id(message)
        for _, message, distance in candidates
        if distance is None or distance <= threshold
    }
    return (
        [message for message in before if id(message) in keep],
        [message for message in after if id(message) in keep],
    )


def _messages_within_time_window(
    item: CommunicationItem,
    messages: list[CommunicationMessage],
    max_seconds: float,
) -> list[CommunicationMessage]:
    return [
        message
        for message in messages
        if _message_within_time_window(item, message, max_seconds) is not None
    ]


def _message_within_time_window(
    item: CommunicationItem,
    message: CommunicationMessage | None,
    max_seconds: float,
) -> CommunicationMessage | None:
    if message is None:
        return None
    item_time = _parse_timestamp(item.sent_at)
    if item_time is None:
        return message
    distance = _distance_seconds(item_time, message)
    if distance is None or distance <= max_seconds:
        return message
    return None


def _include_context_message(
    messages: list[CommunicationMessage],
    required: CommunicationMessage | None,
) -> list[CommunicationMessage]:
    if required is None or required in messages:
        return messages
    return sorted([*messages, required], key=_message_sort_key)


def _distance_seconds(
    item_time: datetime,
    message: CommunicationMessage,
) -> float | None:
    message_time = _parse_timestamp(message.sent_at)
    if message_time is None:
        return None
    return abs((item_time - message_time).total_seconds())


def _parse_timestamp(value: str) -> datetime | None:
    value = _text(value)
    if not value:
        return None
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


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


def _sender_from_metadata(metadata: dict[str, Any]) -> str:
    for key in ("sender", "sender_name", "from"):
        if key not in metadata:
            continue
        value = metadata.get(key)
        if isinstance(value, dict):
            sender = _scalar_metadata_text(value.get("name")) or _scalar_metadata_text(
                value.get("display_name")
            )
        else:
            sender = _scalar_metadata_text(value)
        if sender:
            return sender
    return ""


def _first_metadata_text(
    metadata: dict[str, Any],
    keys: tuple[str, ...],
    *,
    nested_keys: tuple[str, ...] = (),
    scalar_keys: tuple[str, ...] | None = None,
) -> str:
    scalar_keys = keys if scalar_keys is None else scalar_keys
    for key in keys:
        if key not in metadata:
            continue
        raw_value = metadata.get(key)
        value = _nested_metadata_text(raw_value, nested_keys)
        if not value and key in scalar_keys:
            value = _scalar_metadata_text(raw_value)
        if value:
            return value
    return ""


def _nested_metadata_text(value: Any, nested_keys: tuple[str, ...]) -> str:
    if not isinstance(value, dict):
        return ""
    for key in nested_keys:
        if key in value:
            nested_value = _scalar_metadata_text(value.get(key))
            if nested_value:
                return nested_value
    return ""


def _scalar_metadata_text(value: Any) -> str:
    if isinstance(value, (dict, list, tuple, set)):
        return ""
    return _text(value)


def _format_context_message(label: str, message: CommunicationMessage) -> str:
    text = _text(message.text)
    if not text:
        return ""
    attrs = [
        _text(message.sent_at),
        f"message_id={message.message_id}" if message.message_id else "",
        (
            f"source_message_id={message.source_message_id}"
            if message.source_message_id
            else ""
        ),
        f"sender={message.sender}" if message.sender else "",
    ]
    rendered_attrs = " ".join(attr for attr in attrs if attr)
    prefix = f"[{label} {rendered_attrs}]" if rendered_attrs else f"[{label}]"
    return f"{prefix} {text}"


def _message_ids(messages: list[CommunicationMessage]) -> str:
    return ",".join(
        message_id
        for message_id in (_message_id(message) for message in messages)
        if message_id
    )


def _source_message_ids(messages: list[CommunicationMessage]) -> str:
    return ",".join(
        source_message_id
        for source_message_id in (_source_message_id(message) for message in messages)
        if source_message_id
    )


def _message_id(message: CommunicationMessage | None) -> str:
    if message is None:
        return ""
    return message.message_id


def _source_message_id(message: CommunicationMessage | None) -> str:
    if message is None:
        return ""
    return message.source_message_id


def _record_primary_text(doc: dict[str, Any], metadata: dict[str, Any]) -> str:
    return (
        _text(metadata.get("_text"))
        or _text(doc.get("primary_text"))
        or _text(doc.get("text"))
    )


def _record_message_has_context_scope(
    item: CommunicationItem,
    message: CommunicationMessage,
) -> bool:
    return bool(
        any(_text(key) for key in _message_identity_keys(message))
        and _text(message.text)
        and (_text(item.channel_id) or _text(item.thread_id))
    )


def _configured_window(config: dict[str, Any], key: str, default: int) -> int:
    try:
        value = int(config.get(key, default))
    except (TypeError, ValueError):
        return default
    return max(0, value)


def _configured_float(config: dict[str, Any], key: str, default: float) -> float:
    try:
        value = float(config.get(key, default))
    except (TypeError, ValueError):
        return default
    return max(1.0, value)


def _lookup_message_map_value(
    values: dict[str, str],
    message: CommunicationMessage,
    origin_source: str = "",
    channel_id: str = "",
    thread_id: str = "",
) -> str:
    for key in (
        _scoped_message_map_keys(message, origin_source, channel_id, thread_id)
        + _legacy_message_map_keys(message)
    ):
        value = _text(values.get(key))
        if value:
            return value
    return ""


def _scoped_message_map_keys(
    message: CommunicationMessage,
    origin_source: str = "",
    channel_id: str = "",
    thread_id: str = "",
) -> list[str]:
    origin_source = _text(origin_source)
    channel_id = _text(channel_id)
    thread_id = _text(thread_id)
    if not origin_source:
        return []
    keys: list[str] = []
    for kind, value in (
        ("source_message_id", message.source_message_id),
        ("message_id", message.message_id),
    ):
        value = _text(value)
        if value:
            keys.append(
                json.dumps(
                    [kind, origin_source, channel_id, thread_id, value],
                    separators=(",", ":"),
                )
            )
    return keys


def _legacy_message_map_keys(message: CommunicationMessage) -> list[str]:
    keys: list[str] = []
    for key in _message_identity_keys(message):
        key = _text(key)
        if key and key not in keys:
            keys.append(key)
    return keys


def _message_identity_keys(message: CommunicationMessage) -> tuple[str, str]:
    return (message.source_message_id, message.message_id)


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()
