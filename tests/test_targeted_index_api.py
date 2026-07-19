"""HTTP contract for the targeted single-document index endpoint (TICKET-6).

POST /api/index/document is what comm-data-store-hooks calls per new
attachment. index_document_flow itself is covered by test_targeted_index.py;
here we only pin the route's request/response contract.
"""

from unittest.mock import patch

import httpx
import pytest

from api_server import build_api_app


@pytest.fixture
async def client(tmp_path):
    app = build_api_app(tmp_path)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver/") as c:
        yield c


@pytest.mark.anyio
async def test_index_document_route_dispatches_to_flow(client):
    fake = {
        "status": "indexed",
        "doc_id": "documents::00abc",
        "rel_path": "email-attachments/x@00abc@.pdf",
    }
    with patch("flow_index_vault.index_document_flow", return_value=fake) as flow:
        resp = await client.post(
            "/index/document",
            json={"rel_path": "email-attachments/x@00abc@.pdf"},
        )

    assert resp.status_code == 200
    assert resp.json() == fake
    flow.assert_called_once()
    _, kwargs = flow.call_args
    assert kwargs["target"] == "email-attachments/x@00abc@.pdf"
    assert kwargs["source_name"] == "documents"
    assert kwargs["force"] is False


@pytest.mark.anyio
async def test_index_document_route_accepts_doc_id_and_force(client):
    with patch("flow_index_vault.index_document_flow", return_value={"status": "indexed"}) as flow:
        await client.post(
            "/index/document",
            json={"doc_id": "documents::00abc", "source_name": "documents", "force": True},
        )
    _, kwargs = flow.call_args
    assert kwargs["target"] == "documents::00abc"
    assert kwargs["force"] is True


@pytest.mark.anyio
async def test_index_document_route_requires_target(client):
    resp = await client.post("/index/document", json={"source_name": "documents"})
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_index_document_route_maps_not_found_to_404(client):
    with patch(
        "flow_index_vault.index_document_flow",
        return_value={"status": "error", "reason": "not_found"},
    ):
        resp = await client.post("/index/document", json={"rel_path": "nope.pdf"})
    assert resp.status_code == 404
    assert resp.json()["reason"] == "not_found"


@pytest.mark.anyio
async def test_index_document_route_maps_durable_queue_acceptance_to_202(client):
    queued = {
        "status": "queued",
        "reason": "index_write_in_progress",
        "target": "queued.pdf",
        "source_name": "documents",
        "revision": 3,
    }
    with patch("flow_index_vault.index_document_flow", return_value=queued):
        resp = await client.post(
            "/index/document", json={"rel_path": "queued.pdf"}
        )

    assert resp.status_code == 202
    assert resp.json() == queued


@pytest.mark.anyio
async def test_index_document_route_never_returns_202_when_enqueue_fails(client):
    with patch(
        "flow_index_vault.index_document_flow",
        side_effect=OSError("queue disk full"),
    ):
        resp = await client.post(
            "/index/document", json={"rel_path": "queued.pdf"}
        )

    assert resp.status_code == 500
    assert resp.json()["reason"] == "index_failed"
