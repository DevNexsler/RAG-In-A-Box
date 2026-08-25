# REST API Integration Tests - Design Doc: system-design-doc.md
# Generated: 2026-03-08 | Budget Used: 3/3 integration, 0/2 E2E
#
# These tests exercise the Starlette REST API (api_server.py) via HTTPX
# async test client. No external services needed -- uses a temp directory
# as documents_root.
#
# Framework: pytest + httpx (async)
# Run with: pytest tests/test_api_server.int.test.py -v

import hmac
from pathlib import Path

import httpx
import pytest

import mcp_server
from api_server import build_api_app
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_docs_root(tmp_path):
    """Create a temporary directory to serve as documents_root.
    Pre-populate with a sample .md file and a subdirectory containing a .pdf-like file.
    """
    # Create sample .md file
    sample_md = tmp_path / "sample.md"
    sample_md.write_text("# Sample Document\n\nThis is a test markdown file.")

    # Create subdirectory with a .pdf placeholder
    subdir = tmp_path / "reports"
    subdir.mkdir()
    sample_pdf = subdir / "report.pdf"
    sample_pdf.write_bytes(b"%PDF-1.4 fake pdf content for testing")

    return tmp_path


@pytest.fixture
async def api_client(tmp_docs_root):
    """Build the Starlette app via build_api_app(tmp_docs_root) and wrap it
    in httpx.AsyncClient(transport=ASGITransport(app)).
    """
    app = build_api_app(tmp_docs_root)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver/") as client:
        yield client


def _build_auth_app(documents_root: Path, api_key: str):
    """Build the composed app with auth middleware matching server.py logic."""
    api_app = build_api_app(documents_root)
    app = Starlette(routes=[
        Mount("/api", app=api_app),
    ])

    expected = f"Bearer {api_key}".encode()

    async def auth_app(scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            auth_value = b""
            for name, value in scope.get("headers", []):
                if name == b"authorization":
                    auth_value = value
                    break
            if not hmac.compare_digest(auth_value, expected):
                response = JSONResponse({"error": "Unauthorized"}, status_code=401)
                await response(scope, receive, send)
                return
        await app(scope, receive, send)

    return auth_app


@pytest.fixture
async def api_client_with_auth(tmp_docs_root):
    """Build unified server with auth middleware, yields (client, api_key)."""
    api_key = "test-secret-key-12345"
    auth_app = _build_auth_app(tmp_docs_root, api_key)
    transport = httpx.ASGITransport(app=auth_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver/") as client:
        yield client, api_key


# ===================================================================
# TEST 1: File Upload -- Happy Path and Validation
# ===================================================================


@pytest.mark.anyio
async def test_upload_valid_md_file(api_client, tmp_docs_root):
    """AC-REST-1: Upload a valid .md file returns 201 with correct rel_path and size."""
    file_content = b"# My Upload\n\nNew content here."
    files = {"file": ("upload_test.md", file_content, "text/markdown")}

    resp = await api_client.post("/upload", files=files)

    assert resp.status_code == 201
    body = resp.json()
    assert body["uploaded"] is True
    assert body["rel_path"] == "upload_test.md"
    assert body["size"] == len(file_content)

    # File actually exists on disk
    on_disk = tmp_docs_root / "upload_test.md"
    assert on_disk.exists()
    assert on_disk.read_bytes() == file_content


@pytest.mark.anyio
async def test_upload_rejects_disallowed_extension(api_client):
    """AC-REST-1: Upload .exe file returns 400 with invalid_file_type error."""
    files = {"file": ("malware.exe", b"MZ evil content", "application/octet-stream")}

    resp = await api_client.post("/upload", files=files)

    assert resp.status_code == 400
    body = resp.json()
    assert body["error"] is True
    assert body["code"] == "invalid_file_type"


@pytest.mark.anyio
async def test_upload_rejects_empty_file(api_client, tmp_docs_root):
    """AC-REST-1: Upload empty file returns 400 and writes nothing."""
    files = {"file": ("empty.pdf", b"", "application/pdf")}

    resp = await api_client.post("/upload", files=files)

    assert resp.status_code == 400
    body = resp.json()
    assert body["error"] is True
    assert body["code"] == "empty_file"
    assert not (tmp_docs_root / "empty.pdf").exists()


@pytest.mark.anyio
async def test_upload_rejects_path_traversal_in_directory(api_client):
    """AC-REST-1: Directory field with ../../ returns 400 with invalid_directory."""
    files = {"file": ("ok.md", b"# OK", "text/markdown")}
    data = {"directory": "../../etc"}

    resp = await api_client.post("/upload", files=files, data=data)

    assert resp.status_code == 400
    body = resp.json()
    assert body["error"] is True
    assert body["code"] == "invalid_directory"


@pytest.mark.anyio
async def test_upload_rejects_missing_file_field(api_client):
    """AC-REST-1: Upload without file field returns 400 with missing_file."""
    # Send multipart with no "file" field -- send an empty form field instead
    resp = await api_client.post(
        "/upload",
        content=b"",
        headers={"content-type": "multipart/form-data; boundary=boundary123"},
    )

    # The server expects multipart/form-data with a "file" field
    # Sending malformed multipart should result in 400
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_upload_to_subdirectory(api_client, tmp_docs_root):
    """AC-REST-1: Upload with directory field creates file in subdirectory."""
    file_content = b"# Note\n\nSubfolder note."
    files = {"file": ("note.md", file_content, "text/markdown")}
    data = {"directory": "subfolder"}

    resp = await api_client.post("/upload", files=files, data=data)

    assert resp.status_code == 201
    body = resp.json()
    assert "subfolder/" in body["rel_path"] or body["rel_path"] == "subfolder/note.md"

    # File on disk in subfolder
    on_disk = tmp_docs_root / "subfolder" / "note.md"
    assert on_disk.exists()
    assert on_disk.read_bytes() == file_content


@pytest.mark.anyio
async def test_upload_response_names_the_path_rel_path(api_client):
    """#1230: the upload response addresses the file by ``rel_path``.

    The value has always been a documents_root-relative path, never an index
    ``doc_id`` (those are minted by DocIDStore and namespaced per source, e.g.
    ``documents::00001``), so the canonical key is ``rel_path`` and ``doc_id``
    survives only as a deprecated alias carrying the same value.
    """
    files = {"file": ("naming.md", b"# Naming\n", "text/markdown")}

    resp = await api_client.post("/upload", files=files)

    assert resp.status_code == 201
    body = resp.json()
    assert body["rel_path"] == "naming.md"
    assert body["doc_id"] == body["rel_path"], "deprecated alias must mirror rel_path"


@pytest.mark.anyio
async def test_upload_rel_path_downloads_verbatim(api_client):
    """#1230: the upload response's rel_path is exactly what download takes."""
    file_content = b"# Round Trip\n\nUploaded then fetched back."
    files = {"file": ("trip.md", file_content, "text/markdown")}

    upload = await api_client.post("/upload", files=files, data={"directory": "subfolder"})
    assert upload.status_code == 201
    rel_path = upload.json()["rel_path"]
    assert rel_path == "subfolder/trip.md"

    resp = await api_client.get(f"/documents/{rel_path}")

    assert resp.status_code == 200
    assert resp.content == file_content


# ===================================================================
# TEST 2: File Download and Path Traversal Protection
# ===================================================================


@pytest.mark.anyio
async def test_download_existing_file(api_client, tmp_docs_root):
    """AC-REST-2: Download a pre-existing file returns 200 with correct content."""
    expected_content = "# Sample Document\n\nThis is a test markdown file."

    resp = await api_client.get("/documents/sample.md")

    assert resp.status_code == 200
    assert resp.text == expected_content


@pytest.mark.anyio
async def test_download_nested_file(api_client, tmp_docs_root):
    """AC-REST-2: The catch-all still serves multi-segment doc_ids."""
    resp = await api_client.get("/documents/reports/report.pdf")

    assert resp.status_code == 200
    assert resp.content == (tmp_docs_root / "reports" / "report.pdf").read_bytes()


@pytest.mark.anyio
async def test_download_nonexistent_file(api_client):
    """AC-REST-2: Download a nonexistent file returns 404 with error body."""
    resp = await api_client.get("/documents/does_not_exist.md")

    assert resp.status_code == 404
    body = resp.json()
    assert body["error"] is True
    assert body["code"] == "not_found"


@pytest.mark.anyio
async def test_download_path_traversal_blocked(api_client):
    """AC-REST-2: Path traversal in doc_id returns 400.

    Uses URL-encoded path components because httpx normalizes bare ../
    sequences before sending the request.
    """
    resp = await api_client.get("/documents/..%2F..%2Fetc%2Fpasswd")

    assert resp.status_code == 400
    body = resp.json()
    assert body["error"] is True
    assert body["code"] == "invalid_path"


# ===================================================================
# TEST 3: Authentication Enforcement
# ===================================================================


@pytest.mark.anyio
async def test_auth_rejects_missing_token(api_client_with_auth):
    """AC-REST-4: When API_KEY is set, request without auth header returns 401."""
    client, _api_key = api_client_with_auth

    resp = await client.get("/api/documents")

    assert resp.status_code == 401


@pytest.mark.anyio
async def test_auth_rejects_wrong_token(api_client_with_auth):
    """AC-REST-4: When API_KEY is set, wrong Bearer token returns 401."""
    client, _api_key = api_client_with_auth

    resp = await client.get(
        "/api/documents",
        headers={"Authorization": "Bearer wrong-token"},
    )

    assert resp.status_code == 401


@pytest.mark.anyio
async def test_auth_accepts_correct_token(api_client_with_auth, tmp_docs_root):
    """AC-REST-4: When API_KEY is set, correct Bearer token returns 200."""
    client, api_key = api_client_with_auth

    resp = await client.get(
        "/api/documents",
        headers={"Authorization": f"Bearer {api_key}"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert "files" in body


@pytest.mark.anyio
async def test_auth_not_enforced_without_api_key(api_client):
    """AC-REST-4: When API_KEY is not set, requests pass without auth."""
    # api_client fixture has no auth middleware
    resp = await api_client.get("/documents")

    assert resp.status_code == 200
    body = resp.json()
    assert "files" in body


# ===================================================================
# Directory Listing
# ===================================================================


@pytest.mark.anyio
async def test_list_documents_root(api_client, tmp_docs_root):
    """AC-REST-3: Listing root directory returns files with correct structure."""
    resp = await api_client.get("/documents")

    assert resp.status_code == 200
    body = resp.json()
    assert "directory" in body
    assert "files" in body
    assert "total" in body
    assert "offset" in body
    assert "limit" in body

    # Should contain sample.md and reports/ directory
    names = {f["name"] for f in body["files"]}
    assert "sample.md" in names

    # Verify file entry structure
    for f in body["files"]:
        assert "name" in f
        assert "type" in f
        assert "path" in f
        if f["type"] == "file":
            assert "size" in f


@pytest.mark.anyio
async def test_list_documents_filters_extensions(api_client, tmp_docs_root):
    """AC-REST-3: Listing only shows files with allowed extensions."""
    # Create files with disallowed extensions
    (tmp_docs_root / "notes.txt").write_text("text file")
    (tmp_docs_root / "script.py").write_text("python file")
    (tmp_docs_root / "data.json").write_text("{}")

    resp = await api_client.get("/documents")

    body = resp.json()
    names = {f["name"] for f in body["files"] if f["type"] == "file"}

    # .md and .txt are allowed, .py and .json are NOT
    assert "sample.md" in names
    assert "notes.txt" in names
    assert "script.py" not in names
    assert "data.json" not in names


@pytest.mark.anyio
async def test_list_documents_pagination(api_client, tmp_docs_root):
    """AC-REST-3: Offset and limit params paginate results correctly."""
    # Create 5 .md files
    for i in range(5):
        (tmp_docs_root / f"doc{i}.md").write_text(f"# Doc {i}")

    # Get total count first
    resp_all = await api_client.get("/documents")
    total = resp_all.json()["total"]
    assert total >= 5  # at least our 5 + the pre-existing sample.md

    # Paginate with limit=2, offset=1
    resp = await api_client.get("/documents?limit=2&offset=1")

    body = resp.json()
    assert len(body["files"]) == 2
    assert body["total"] == total
    assert body["offset"] == 1


@pytest.mark.anyio
async def test_list_documents_trailing_slash(api_client):
    """AC-REST-3: /documents/ lists like /documents instead of hitting download."""
    resp = await api_client.get("/documents/")

    assert resp.status_code == 200
    assert resp.json() == (await api_client.get("/documents")).json()


@pytest.mark.anyio
async def test_list_documents_trailing_slash_with_directory(api_client):
    """AC-REST-3: The documented /documents/?directory=... form lists that subdirectory."""
    resp = await api_client.get("/documents/?directory=reports&limit=50")

    assert resp.status_code == 200
    body = resp.json()
    assert body["directory"] == "reports"
    assert {f["name"] for f in body["files"]} == {"report.pdf"}


@pytest.mark.anyio
async def test_search_post_returns_mcp_search_results(api_client, monkeypatch):
    """AC-REST-5: POST /search forwards JSON body to file_search implementation."""
    calls = []

    def fake_search(**kwargs):
        calls.append(kwargs)
        return {
            "results": [
                {
                    "doc_id": "notes/ml.md",
                    "loc": "c:0",
                    "snippet": "Neural search notes",
                    "score": 0.42,
                    "title": "ML Notes",
                }
            ],
            "diagnostics": {"degraded": False},
        }

    monkeypatch.setattr(mcp_server, "_file_search_impl", fake_search)

    resp = await api_client.post(
        "/search",
        json={"query": "neural search", "top_k": 3, "folder": "notes"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["results"][0]["doc_id"] == "notes/ml.md"
    assert body["diagnostics"]["degraded"] is False
    assert calls == [
        {
            "query": "neural search",
            "top_k": 3,
            "doc_id_prefix": None,
            "source_type": None,
            "source_name": None,
            "tags": None,
            "status": None,
            "folder": "notes",
            "prefer_recent": False,
            "metadata_filters": None,
            "enr_doc_type": None,
            "enr_topics": None,
            "filter": None,
        }
    ]


async def _post_search(client):
    return await client.post("/search", json={"query": "neural"})


@pytest.mark.anyio
async def test_search_post_does_not_block_the_serving_loop(api_client, monkeypatch):
    """A slow /search must leave the loop free for the liveness route (#1086).

    /api/search is the REST twin of the file_search MCP tool and runs the same
    blocking implementation (embed → LanceDB → rerank, routinely 10-20s in
    production). Called inline on the serving event loop it starves every other
    route — including the unauthenticated /health probe the container
    healthcheck polls with a 5s timeout — so it must be dispatched to a thread.
    """
    import threading
    import time

    import anyio

    search_started = threading.Event()
    release_search = threading.Event()
    probe_bound_s = 1.0
    search_block_s = 3.0

    def blocking_search(**_kwargs):
        search_started.set()
        release_search.wait(timeout=search_block_s)  # bounded: never hang the suite
        return {"results": [], "diagnostics": {}}

    monkeypatch.setattr(mcp_server, "_file_search_impl", blocking_search)

    try:
        # Measured from before the search starts: a probe timed only after the
        # blocking call returned would look fast even on a loop starved
        # throughout it.
        started_at = time.monotonic()
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(_post_search, api_client)
            assert await anyio.to_thread.run_sync(search_started.wait, search_block_s)
            resp = await api_client.get("/documents")
            elapsed = time.monotonic() - started_at
            release_search.set()
    finally:
        release_search.set()

    assert resp.status_code == 200
    assert elapsed < probe_bound_s, (
        f"a sibling request waited {elapsed:.1f}s behind a blocking /search "
        "— the serving event loop was starved"
    )
