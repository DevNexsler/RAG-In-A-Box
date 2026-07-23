"""e2e fixtures: drives the staging compose stack from the OUTSIDE.

Requires `docker compose -f docker-compose.staging.yml up -d --build --wait`
(scripts/gate.py does this for the staging-e2e tier). Real MCP streamable-HTTP
+ real REST against localhost:17788; provider-sim admin API on localhost:19999.
"""
import json
import subprocess
import time
from pathlib import Path

import anyio
import httpx
import pytest

from tests.e2e.client import (
    AUTH_HEADERS,
    COVERAGE_FILE,
    E2E_BASE,
    E2E_SIM_URL,
    RecordingSession,
    get_hook_events,
    open_mcp_session,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "e2e"
COMPOSE_FILE = ROOT / "docker-compose.staging.yml"

# 5 uploaded files + 5 seeded sor messages
EXPECTED_CORPUS_DOCS = 10
INDEX_TIMEOUT_S = 180
POLL_INTERVAL_S = 3

NOTE_PHRASE = "quixotic manganese lighthouse"


@pytest.fixture(scope="session")
def anyio_backend():
    # Session scope (anyio's default fixture is narrower) so session-scoped
    # async machinery and per-test anyio tests agree on one backend.
    return "asyncio"


@pytest.fixture(scope="session", autouse=True)
def staging_stack_reachable():
    """Fail the whole session fast, with instructions, if the stack is down."""
    try:
        health = httpx.get(f"{E2E_BASE}/health", timeout=10)
        sim = httpx.get(f"{E2E_SIM_URL}/", timeout=10)
        if health.status_code >= 500 or sim.status_code != 200:
            raise RuntimeError(f"health={health.status_code} sim={sim.status_code}")
    except Exception as exc:  # noqa: BLE001 — any failure means "stack not up"
        pytest.exit(
            "staging stack not up — run: "
            "docker compose -f docker-compose.staging.yml up -d --build --wait "
            f"(probe error: {exc})",
            returncode=3,
        )
    if COVERAGE_FILE.exists():
        COVERAGE_FILE.unlink()


@pytest.fixture(autouse=True)
def sim_reset(staging_stack_reachable):
    """Reset provider-sim (sink + armed faults) before AND after every test."""
    httpx.post(f"{E2E_SIM_URL}/admin/reset", timeout=10)
    yield
    httpx.post(f"{E2E_SIM_URL}/admin/reset", timeout=10)


@pytest.fixture
async def mcp_session(request):
    async with open_mcp_session(request.node.name) as session:
        yield session


@pytest.fixture
async def api():
    async with httpx.AsyncClient(
        base_url=E2E_BASE, headers=dict(AUTH_HEADERS), timeout=60
    ) as client:
        yield client


async def wait_for_index(
    session: RecordingSession,
    *,
    min_docs: int = 1,
    timeout_s: float = INDEX_TIMEOUT_S,
    require_idle: bool = True,
) -> dict:
    """Poll file_status until the index reaches min_docs and the indexer is idle.

    Deadline-based (no naked sleeps as sync); raises TimeoutError with the last
    status payload so failures are diagnosable.
    """
    deadline = time.monotonic() + timeout_s
    status: dict = {}
    while time.monotonic() < deadline:
        status = await session.call_tool_json("file_status", {})
        if isinstance(status, dict) and not status.get("error"):
            done = status.get("doc_count", 0) >= min_docs and status.get("last_run_at")
            if done and (not require_idle or not status.get("indexer_running")):
                return status
        await anyio.sleep(POLL_INTERVAL_S)
    raise TimeoutError(
        f"index did not reach {min_docs} docs within {timeout_s}s; "
        f"last file_status: {json.dumps(status, default=str)[:2000]}"
    )


def _compose_cp_into_documents(local: Path) -> None:
    """Deposit a file into the container's /data/documents volume.

    /api/upload allowlists document extensions only (api_server._ALLOWED_EXTENSIONS
    has no .wav/.mp4) — in production, media lands in the documents tree via the
    deposit path (comm hooks writing to the shared volume), which `docker compose
    cp` faithfully emulates from outside the container.
    """
    subprocess.run(
        [
            "docker", "compose", "-f", str(COMPOSE_FILE), "cp",
            str(local), f"doc-organizer-staging:/data/documents/{local.name}",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )


async def _build_corpus() -> dict:
    uploaded = {}
    async with httpx.AsyncClient(
        base_url=E2E_BASE, headers=dict(AUTH_HEADERS), timeout=60
    ) as api:
        for name in ("note.md", "report.pdf", "diagram.png"):
            path = FIXTURES / name
            resp = await api.post(
                "/api/upload", files={"file": (name, path.read_bytes())}
            )
            assert resp.status_code == 201, f"upload {name}: {resp.status_code} {resp.text}"
            uploaded[name] = resp.json()["doc_id"]

    for name in ("clip.wav", "clip.mp4", "clip.json"):
        _compose_cp_into_documents(FIXTURES / name)
        if name != "clip.json":
            uploaded[name] = name

    async with open_mcp_session("indexed_corpus") as session:
        started = await session.call_tool_json("file_index_update", {})
        assert started.get("status") == "started", f"file_index_update: {started}"
        status = await wait_for_index(session, min_docs=EXPECTED_CORPUS_DOCS)

    return {
        "uploaded": uploaded,
        "status": status,
        "hook_events": await get_hook_events(),
    }


@pytest.fixture(scope="session")
def indexed_corpus(staging_stack_reachable):
    """Upload all fixtures, run the full sweep (documents + sor), wait until indexed.

    Sync session fixture driving its own event loop: keeps it independent of
    per-test loops, and captures the webhook-sink snapshot BEFORE any per-test
    /admin/reset wipes it.
    """
    return anyio.run(_build_corpus)
