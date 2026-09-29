"""Android MMS media (.amr voice note, .3gp video) through the per-attachment path (#3711).

Quo tenants send Android's default MMS formats. The comm hooks deposit each one
under quo-attachments/ and call REST /api/index/document; before #3711 the
source include and the extractor both rejected the extensions, so the file was
never described. Runs after the corpus tests (zz) because it adds documents.
"""

import os
import subprocess

import httpx
import pytest

from tests.e2e.client import AUTH_HEADERS, E2E_BASE, get_hook_events
from tests.e2e.conftest import COMPOSE_FILE, FIXTURES, ROOT

pytestmark = pytest.mark.anyio
E2E_REAL = os.environ.get("E2E_REAL", "").strip() == "1"
DEPOSIT_DIR = "quo-attachments/e2e-tenant/2026-09"


def _deposit(name: str) -> str:
    rel_path = f"{DEPOSIT_DIR}/{name}"
    for argv in (
        ["exec", "-T", "doc-organizer-staging",
         "mkdir", "-p", f"/data/documents/{DEPOSIT_DIR}"],
        ["cp", str(FIXTURES / name), f"doc-organizer-staging:/data/documents/{rel_path}"],
    ):
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), *argv],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
    return rel_path


@pytest.mark.skipif(
    E2E_REAL,
    reason=(
        "real mode's OpenRouter audio routes are not the production primary; "
        "production transcribes AMR through the LiteLLM omni route (#3711)"
    ),
)
@pytest.mark.parametrize(
    ("name", "media_kind"),
    [("voice.amr", "input_audio"), ("clip.3gp", "video_url")],
)
async def test_mms_attachment_indexes_with_media_description(
    indexed_corpus, mcp_session, name: str, media_kind: str
):
    rel_path = _deposit(name)

    async with httpx.AsyncClient(
        base_url=E2E_BASE, headers=dict(AUTH_HEADERS), timeout=120
    ) as api:
        resp = await api.post(
            "/api/index/document",
            json={"rel_path": rel_path, "source_name": "documents"},
        )
    assert resp.status_code == 200, resp.text
    result = resp.json()
    assert result.get("status") == "indexed", result

    events = [
        event
        for event in await get_hook_events()
        if event.get("event") == "document.indexed"
        and event.get("doc_id") == result["doc_id"]
    ]
    assert events and events[0].get("chunks"), events

    chunks = await mcp_session.call_tool_json(
        "file_get_doc_chunks", {"doc_id": result["doc_id"]}
    )
    assert isinstance(chunks, list) and chunks, chunks
    stored = "\n".join(chunk.get("text", "") for chunk in chunks)
    assert f"simulated {media_kind} transcript" in stored, stored[:2000]
