"""Production-shaped audio route and fallback coverage through candidate container."""

import os

import httpx
import pytest

from tests.e2e.client import E2E_SIM_URL, get_hook_events, search_hits

pytestmark = pytest.mark.anyio
E2E_REAL = os.environ.get("E2E_REAL", "").strip() == "1"


async def _arm_429(route_prefix: str, times: int) -> None:
    async with httpx.AsyncClient(timeout=10) as sim:
        response = await sim.post(
            f"{E2E_SIM_URL}/admin/fault",
            json={"route_prefix": route_prefix, "fault": "429", "times": times},
        )
        response.raise_for_status()


@pytest.mark.parametrize(
    ("expected_model", "transcription_faults", "chat_faults"),
    [
        ("openai/whisper-1", 0, 0),
        ("mistralai/voxtral-small-24b-2507", 1, 0),
        ("google/gemini-2.5-flash-lite", 1, 1),
    ],
)
@pytest.mark.skipif(
    E2E_REAL,
    reason="provider-simulator fault injection is unavailable in real mode",
)
async def test_each_audio_candidate_persists_and_callbacks_after_ordered_fallback(
    indexed_corpus,
    mcp_session,
    expected_model: str,
    transcription_faults: int,
    chat_faults: int,
):
    """Same WAV reaches each ordered candidate; output crosses persistence and hook seams."""
    if transcription_faults:
        await _arm_429("/api/v1/audio/transcriptions", transcription_faults)
    if chat_faults:
        await _arm_429("/api/v1/chat/completions", chat_faults)

    result = await mcp_session.call_tool_json(
        "file_index_document",
        {"target": "clip.wav", "source_name": "documents", "force": True},
    )
    assert result.get("status") == "indexed", result

    events = await get_hook_events()
    matching = [
        event
        for event in events
        if event.get("event") == "document.indexed"
        and event.get("rel_path", "").endswith(".wav")
    ]
    assert matching and matching[0].get("chunks"), events

    payload = await mcp_session.call_tool_json(
        "file_search", {"query": expected_model, "top_k": 50}
    )
    hits = search_hits(payload, "clip")
    assert hits, payload
    chunks = await mcp_session.call_tool_json(
        "file_get_doc_chunks", {"doc_id": hits[0]["doc_id"]}
    )
    assert isinstance(chunks, list) and chunks, chunks
    stored = "\n".join(chunk.get("text", "") for chunk in chunks)
    assert f"model {expected_model}" in stored, stored[:2000]
