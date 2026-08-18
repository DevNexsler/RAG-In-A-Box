"""Liveness under load — /health must answer while the server is busy (#1086).

The image healthcheck is
``urlopen('http://127.0.0.1:7788/health', timeout=5)`` with ``--retries=3``, so
a probe that takes longer than 5s is a miss and three misses in a row flip the
container to ``unhealthy`` — 81 such episodes in five days while the indexer
was progressing normally. These tests put real in-process load on the candidate
container and assert the probe latency bound from outside, the way docker does.
"""
import time

import anyio
import httpx
import pytest

from tests.e2e.client import E2E_BASE, E2E_SIM_URL
from tests.e2e.conftest import wait_for_index

pytestmark = pytest.mark.anyio

HEALTHCHECK_TIMEOUT_S = 5.0  # the image healthcheck's own urlopen timeout
HEALTHCHECK_RETRIES = 3  # consecutive misses that flip the container unhealthy
PROBE_BOUND_S = 2.0  # asserted latency bound, well inside the timeout
SLOW_EMBED_S = 8.0  # longer than the timeout, so a starved loop is provable


async def _arm_slow_provider(route_prefix: str, seconds: float, times: int = 1) -> None:
    """Make provider-sim stall a route, so a real tool call blocks in-process."""
    async with httpx.AsyncClient(timeout=10) as sim:
        resp = await sim.post(
            f"{E2E_SIM_URL}/admin/fault",
            json={
                "route_prefix": route_prefix,
                "fault": "timeout",
                "seconds": seconds,
                "times": times,
            },
        )
        assert resp.status_code == 200, resp.text


async def _probe_health(probe: httpx.AsyncClient) -> tuple[httpx.Response, float]:
    started = time.monotonic()
    resp = await probe.get("/health")
    return resp, time.monotonic() - started


async def _assert_health_stays_answerable(probe: httpx.AsyncClient, *, probes: int) -> None:
    """Poll /health like the healthcheck does; every answer must beat the bound.

    Status is not asserted: 200 and 503 are both *answers* (the probe reports
    real index conditions). What flips the container unhealthy here is latency.
    """
    for attempt in range(probes):
        resp, elapsed = await _probe_health(probe)
        assert resp.status_code in (200, 503), f"probe {attempt}: {resp.status_code} {resp.text}"
        assert elapsed < PROBE_BOUND_S, (
            f"/health took {elapsed:.1f}s on probe {attempt} — the healthcheck "
            f"gives it {HEALTHCHECK_TIMEOUT_S}s and flips unhealthy after "
            f"{HEALTHCHECK_RETRIES} misses"
        )
        await anyio.sleep(1)


async def test_health_answers_while_a_slow_tool_call_is_in_flight(indexed_corpus, mcp_session):
    """A tool call stuck on a slow provider must not starve the probe.

    file_search is a synchronous tool body (embed → LanceDB → rerank); run on
    the serving event loop it holds the whole process for its entire duration,
    which in production is routinely 10-20s.
    """
    await _arm_slow_provider("/api/v1/embeddings", seconds=SLOW_EMBED_S)

    search: dict = {}
    search_elapsed = 0.0

    async def _slow_search() -> None:
        nonlocal search, search_elapsed
        started = time.monotonic()
        search = await mcp_session.call_tool_json("file_search", {"query": "lighthouse"})
        search_elapsed = time.monotonic() - started

    async with httpx.AsyncClient(base_url=E2E_BASE, timeout=HEALTHCHECK_TIMEOUT_S) as probe:
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(_slow_search)
            await _assert_health_stays_answerable(probe, probes=HEALTHCHECK_RETRIES + 1)

    assert not search.get("error"), search
    assert search_elapsed >= SLOW_EMBED_S, (
        f"search returned in {search_elapsed:.1f}s — the injected "
        f"{SLOW_EMBED_S}s provider stall never applied, so the probes above "
        "did not overlap a long in-process call"
    )


async def test_health_answers_while_a_full_index_run_is_in_progress(indexed_corpus, mcp_session):
    """The liveness path must stay responsive for a whole sweep, not just when idle."""
    started = await mcp_session.call_tool_json("file_index_update", {})
    assert started.get("status") in ("started", "already_running"), started

    async with httpx.AsyncClient(base_url=E2E_BASE, timeout=HEALTHCHECK_TIMEOUT_S) as probe:
        await _assert_health_stays_answerable(probe, probes=HEALTHCHECK_RETRIES + 1)

    # Leave the stack idle for the next test rather than racing a live sweep.
    await wait_for_index(mcp_session, min_docs=1)
