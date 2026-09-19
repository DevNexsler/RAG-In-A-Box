"""Terminal degraded state stays visible through real candidate sweeps (#2022)."""
import json
import subprocess
import time

import anyio
import pytest

from tests.e2e.conftest import COMPOSE_FILE, ROOT, indexer_log_lines

pytestmark = pytest.mark.anyio


def _candidate_python(script):
    return subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "exec", "-T",
         "doc-organizer-staging", "python", "-c", script],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout


async def _sweep(session):
    before = await session.call_tool_json("file_status", {})
    started = await session.call_tool_json("file_index_update", {})
    assert started.get("status") == "started", started
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        status = await session.call_tool_json("file_status", {})
        if (status.get("last_run_at") != before.get("last_run_at")
                and not status.get("indexer_running")):
            return status
        await anyio.sleep(1)
    raise AssertionError("candidate sweep did not finish")


async def test_terminal_backlog_survives_next_sweep(indexed_corpus, mcp_session):
    # Real SQLite retirements, including legacy terminal residue. No prod state.
    _candidate_python('''
import json
from pathlib import Path
from doc_id_store import DocIDStore
root = Path('/data/index')
registry = DocIDStore(root / 'doc_registry.db')
retired = [f'documents::ticket2022-retired-{i}' for i in range(16)]
for doc_id in retired:
    registry.register(doc_id, doc_id)
    registry.delete(doc_id)
registry.close()
(root / 'degraded_docs.json').write_text(json.dumps({'version': 2, 'docs': {
    retired[0]: {'attempts': 1, 'unresolved_runs': 2},
    'documents::ticket2022-missing': {'attempts': 1, 'unresolved_runs': 2},
}}))
(root / 'degraded_unresolved.json').write_text(json.dumps({'docs': {
    doc_id: {'escalated_at': 1000} for doc_id in retired
}}))
''')
    try:
        for _ in range(2):
            status = await _sweep(mcp_session)
            assert "degraded_unresolved:1" in status["health"]["last_index_warnings"]
            state = json.loads(_candidate_python('''
import json
from pathlib import Path
root = Path('/data/index')
print(json.dumps({name: json.loads((root / name).read_text())
                  for name in ('degraded_docs.json', 'degraded_unresolved.json')}))
'''))
            assert not state["degraded_docs.json"]["docs"]
            assert set(state["degraded_unresolved.json"]["docs"]) == {
                "documents::ticket2022-missing"
            }
            summaries = [line for line in indexer_log_lines() if "Degraded ledger:" in line]
            assert "1 terminal, oldest" in summaries[-1]
        assert status["doc_count"] >= indexed_corpus["status"]["doc_count"]
    finally:
        _candidate_python('''
from pathlib import Path
for name in ('degraded_docs.json', 'degraded_unresolved.json'):
    (Path('/data/index') / name).write_text('{"version": 2, "docs": {}}')
''')
        await _sweep(mcp_session)
