"""REST API for document upload and download.

Separate from MCP server — mounted alongside it in server.py.
Requires the same API_KEY auth when set.
"""

import logging
import os
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from core.config import load_config

logger = logging.getLogger(__name__)

# Max upload size: 100 MB
_MAX_UPLOAD_BYTES = 100 * 1024 * 1024

_ALLOWED_EXTENSIONS = {".md", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp"}


def _get_documents_root() -> Path:
    config = load_config()
    return Path(config["documents_root"])


async def upload(request: Request) -> JSONResponse:
    """Upload a file to documents_root.

    Multipart form data:
        file: The file to upload (required).
        directory: Subdirectory within documents_root (optional, default: root).
    """
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        return JSONResponse(
            {"error": True, "code": "invalid_request", "message": "Expected multipart/form-data"},
            status_code=400,
        )

    form = await request.form()
    file = form.get("file")
    if file is None:
        return JSONResponse(
            {"error": True, "code": "missing_file", "message": "No file field in upload"},
            status_code=400,
        )

    filename = file.filename
    if not filename:
        return JSONResponse(
            {"error": True, "code": "missing_filename", "message": "File has no filename"},
            status_code=400,
        )

    # Validate extension
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        return JSONResponse(
            {
                "error": True,
                "code": "invalid_file_type",
                "message": f"File type '{ext}' not allowed. Allowed: {sorted(_ALLOWED_EXTENSIONS)}",
            },
            status_code=400,
        )

    # Sanitize filename — prevent path traversal
    safe_name = Path(filename).name
    if not safe_name or safe_name.startswith("."):
        return JSONResponse(
            {"error": True, "code": "invalid_filename", "message": "Invalid filename"},
            status_code=400,
        )

    # Resolve target directory
    docs_root = _get_documents_root()
    directory = form.get("directory", "")
    if isinstance(directory, str) and directory.strip():
        # Sanitize directory — no .. traversal
        dir_path = Path(directory.strip().strip("/"))
        if ".." in dir_path.parts:
            return JSONResponse(
                {"error": True, "code": "invalid_directory", "message": "Directory must not contain '..'"},
                status_code=400,
            )
        target_dir = docs_root / dir_path
    else:
        target_dir = docs_root

    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / safe_name

    # Read and write file
    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        return JSONResponse(
            {"error": True, "code": "file_too_large", "message": f"File exceeds {_MAX_UPLOAD_BYTES // (1024*1024)} MB limit"},
            status_code=413,
        )

    target_path.write_bytes(content)

    doc_id = str(target_path.relative_to(docs_root)).replace("\\", "/")
    logger.info("Uploaded: %s (%d bytes)", doc_id, len(content))

    return JSONResponse(
        {
            "uploaded": True,
            "doc_id": doc_id,
            "size": len(content),
            "path": str(target_path),
        },
        status_code=201,
    )


async def download(request: Request) -> FileResponse | JSONResponse:
    """Download a file by doc_id (path relative to documents_root).

    GET /api/documents/{doc_id:path}
    """
    doc_id = request.path_params.get("doc_id", "")
    if not doc_id:
        return JSONResponse(
            {"error": True, "code": "missing_doc_id", "message": "doc_id path parameter required"},
            status_code=400,
        )

    # Prevent path traversal
    if ".." in Path(doc_id).parts:
        return JSONResponse(
            {"error": True, "code": "invalid_path", "message": "Path must not contain '..'"},
            status_code=400,
        )

    docs_root = _get_documents_root()
    file_path = docs_root / doc_id

    # Ensure resolved path is still under documents_root
    try:
        file_path.resolve().relative_to(docs_root.resolve())
    except ValueError:
        return JSONResponse(
            {"error": True, "code": "invalid_path", "message": "Path escapes documents root"},
            status_code=400,
        )

    if not file_path.is_file():
        return JSONResponse(
            {"error": True, "code": "not_found", "message": f"File not found: {doc_id}"},
            status_code=404,
        )

    return FileResponse(file_path, filename=file_path.name)


async def list_documents(request: Request) -> JSONResponse:
    """List files in a directory within documents_root.

    GET /api/documents/?directory=optional/subdir
    """
    docs_root = _get_documents_root()
    directory = request.query_params.get("directory", "")

    if directory:
        dir_path = Path(directory.strip().strip("/"))
        if ".." in dir_path.parts:
            return JSONResponse(
                {"error": True, "code": "invalid_directory", "message": "Directory must not contain '..'"},
                status_code=400,
            )
        target_dir = docs_root / dir_path
    else:
        target_dir = docs_root

    if not target_dir.is_dir():
        return JSONResponse(
            {"error": True, "code": "not_found", "message": f"Directory not found: {directory}"},
            status_code=404,
        )

    files = []
    for item in sorted(target_dir.iterdir()):
        rel = str(item.relative_to(docs_root)).replace("\\", "/")
        if item.is_dir():
            files.append({"name": item.name, "type": "directory", "path": rel})
        elif item.suffix.lower() in _ALLOWED_EXTENSIONS:
            files.append({
                "name": item.name,
                "type": "file",
                "path": rel,
                "size": item.stat().st_size,
            })

    return JSONResponse({"directory": directory or ".", "files": files, "count": len(files)})


def build_api_app() -> Starlette:
    """Build the REST API Starlette app."""
    routes = [
        Route("/upload", upload, methods=["POST"]),
        Route("/documents/{doc_id:path}", download, methods=["GET"]),
        Route("/documents/", list_documents, methods=["GET"]),
        Route("/documents", list_documents, methods=["GET"]),
    ]
    return Starlette(routes=routes)
