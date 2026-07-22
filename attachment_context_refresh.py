"""Cheap attachment context refresh using stored text and sidecars only."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from communication_context import (
    communication_item_from_sidecar,
    context_envelope_from_sidecar_payload,
    format_context_envelope_for_prompt,
)


CONVERSATION_CONTEXT_MARKER = "[Conversation context]"


@dataclass(frozen=True)
class ContextRefreshPlan:
    doc_id: str
    loc: str
    current_text: str
    desired_text: str

    @property
    def needs_refresh(self) -> bool:
        return self.current_text != self.desired_text


def context_text_from_sidecar(
    sidecar_path: str | Path,
    *,
    doc_id: str,
    max_time_window_minutes: float = 15,
) -> str:
    """Render stored sidecar context without reading attachment media."""
    sidecar_path = Path(sidecar_path)
    payload = json.loads(sidecar_path.read_text())
    item = communication_item_from_sidecar(Path(doc_id), sidecar_path)
    envelope = context_envelope_from_sidecar_payload(
        payload,
        item,
        max_time_window_minutes=max_time_window_minutes,
    )
    return format_context_envelope_for_prompt(envelope)


def build_contextualized_text(text: str, context_text: str) -> str:
    """Replace standardized conversation suffix without duplicating it."""
    base, marker, _existing = (text or "").partition(CONVERSATION_CONTEXT_MARKER)
    if marker:
        base = base.rstrip()
    else:
        base = (text or "").rstrip()
    context_text = (context_text or "").strip()
    if not context_text:
        return base
    return f"{base}\n\n{CONVERSATION_CONTEXT_MARKER}\n{context_text}"


def refresh_document_context(
    store: Any,
    embed_provider: Any,
    doc_id: str,
    context_text: str,
) -> bool:
    """Refresh one anchor chunk. Returns true only when stored row changed."""
    plan = plan_document_context(store, doc_id, context_text)
    if plan is None or not plan.needs_refresh:
        return False
    vectors = embed_provider.embed_texts([plan.desired_text])
    if len(vectors) != 1 or not vectors[0]:
        raise RuntimeError(f"Embedding provider returned no vector for {doc_id}")
    return bool(
        store.replace_chunk_text_and_vector(
            doc_id,
            plan.loc,
            plan.current_text,
            plan.desired_text,
            vectors[0],
        )
    )


def plan_document_context(
    store: Any,
    doc_id: str,
    context_text: str,
) -> ContextRefreshPlan | None:
    """Build read-only refresh plan for dry-run and write paths."""
    chunks = store.get_doc_chunks(doc_id)
    if not chunks:
        return None
    anchors = [
        chunk for chunk in chunks if CONVERSATION_CONTEXT_MARKER in chunk.text
    ]
    anchor = anchors[-1] if anchors else chunks[-1]
    updated_text = build_contextualized_text(anchor.text, context_text)
    return ContextRefreshPlan(
        doc_id=doc_id,
        loc=anchor.loc,
        current_text=anchor.text,
        desired_text=updated_text,
    )
