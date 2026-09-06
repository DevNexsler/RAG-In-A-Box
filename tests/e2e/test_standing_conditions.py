"""Standing operator conditions, end to end against the candidate container.

Ticket #2101: two docs parked at the degraded ledger's terminal cap since
2026-08-11 re-emitted a byte-identical ERROR on every one of ~116 daily runs
(117 of a 24h window's 118 ERROR lines), and `index_health.json` called them
`retry_pending` — a permanent state described in transient language.

Both halves are asserted at the service boundary here: the log the nightly
review actually reads (`/data/index/indexer.log`, not container stdout), and
the `file_status` deep-check payload an operator reads.
"""
import subprocess
import time

import anyio
import pytest

from tests.e2e.conftest import (
    COMPOSE_FILE,
    ROOT,
    indexer_log_lines,
    wait_for_index,
)

pytestmark = pytest.mark.anyio

CAPPED_DOC_ID = "documents::e2e2101cap"
CAPPED_REASON = "vision_sidecar_failed:blocked_on_upstream"
SKIPPED_DOC_ID = "documents::e2e2101skip"


def _exec_in_container(script: str) -> str:
    completed = subprocess.run(
        [
            "docker", "compose", "-f", str(COMPOSE_FILE), "exec", "-T",
            "doc-organizer-staging", "python", "-c", script,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _seed_standing_conditions(capped_doc_ids: list[str]) -> None:
    """Park docs at the terminal cap and record one permanent skip.

    Written straight into the ledgers because reaching this state naturally
    needs a provider-error artifact plus three unchanged sweeps — the state is
    the subject here, not the path that produces it (which
    tests/test_degraded_ledger.py covers).
    """
    _exec_in_container(
        "import json;"
        "docs={doc_id: {'reasons': [%r], 'attempts': 0, 'blocked_attempts': 3,"
        " 'change_key': 'e2e-2101', 'last_attempt_at': 1.0}"
        " for doc_id in %r};"
        "json.dump({'version': 2, 'docs': docs},"
        " open('/data/index/degraded_docs.json','w'));"
        "skip=json.load(open('/data/index/skip_docs.json'))"
        " if __import__('os').path.exists('/data/index/skip_docs.json')"
        " else {'docs': {}};"
        "skip['docs'][%r]={'reasons': ['corrupt_mangled_binary'],"
        " 'change_key': 'e2e-2101', 'skipped_at': 1000.0};"
        "json.dump(skip, open('/data/index/skip_docs.json','w'))"
        % (CAPPED_REASON, capped_doc_ids, SKIPPED_DOC_ID)
    )


def _snapshot_ledgers() -> str:
    """Both ledgers verbatim, so this test leaves the shared stack as it found it."""
    return _exec_in_container(
        "import json, os;"
        "read=lambda p: json.load(open(p)) if os.path.exists(p) else None;"
        "print(json.dumps({'degraded': read('/data/index/degraded_docs.json'),"
        " 'skip': read('/data/index/skip_docs.json')}))"
    )


def _restore_ledgers(snapshot: str) -> None:
    _exec_in_container(
        "import json, os;"
        "snap=json.loads(%r);"
        "paths={'degraded': '/data/index/degraded_docs.json',"
        " 'skip': '/data/index/skip_docs.json'};"
        "[json.dump(snap[k], open(p,'w')) if snap[k] is not None"
        " else (os.path.exists(p) and os.remove(p))"
        " for k, p in paths.items()];"
        "os.path.exists('/data/index/standing_conditions.json')"
        " and os.remove('/data/index/standing_conditions.json')"
        % snapshot
    )


def _clear_standing_conditions() -> None:
    """Empty both sets — but leave the announcer's own state file alone, or the
    next run has nothing to notice the clearing against."""
    _exec_in_container(
        "import json;"
        "json.dump({'version': 2, 'docs': {}},"
        " open('/data/index/degraded_docs.json','w'));"
        "skip=json.load(open('/data/index/skip_docs.json'));"
        "skip['docs'].pop(%r, None);"
        "json.dump(skip, open('/data/index/skip_docs.json','w'))"
        % SKIPPED_DOC_ID
    )


async def _sweep(session) -> None:
    started = await session.call_tool_json("file_index_update", {})
    assert started.get("status") == "started", f"file_index_update: {started}"
    # file_status can still read "idle" for a moment after launch, which would
    # let wait_for_index return before the sweep has processed anything.
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if (await session.call_tool_json("file_status", {})).get("indexer_running"):
            break
        await anyio.sleep(1)
    await wait_for_index(session, min_docs=1)


def _lines_matching(fragment: str, since: int) -> list[str]:
    return [line for line in indexer_log_lines()[since:] if fragment in line]


async def test_standing_conditions_report_on_transition_not_every_run(
    indexed_corpus, mcp_session
):
    snapshot = _snapshot_ledgers()
    try:
        _seed_standing_conditions([CAPPED_DOC_ID])
        first_line = len(indexer_log_lines())

        await _sweep(mcp_session)
        entered_cap = _lines_matching("parked at terminal cap", first_line)
        entered_skip = _lines_matching("permanent actionable skips", first_line)
        assert len(entered_cap) == 1, entered_cap
        assert "ERROR" in entered_cap[0] and CAPPED_DOC_ID in entered_cap[0]
        assert len(entered_skip) == 1, entered_skip

        # The state an operator reads must call it what it is, and must not be
        # the only thing `overall` can ever say.
        status = await mcp_session.call_tool_json("file_status", {})
        deep_check = status["health"]["deep_check"]
        assert deep_check["standing_actions"]["by_reason"][CAPPED_REASON] == [
            CAPPED_DOC_ID
        ]
        assert SKIPPED_DOC_ID in (
            deep_check["standing_actions"]["by_reason"]["corrupt_mangled_binary"]
        )

        # Second sweep, unchanged sets: no new ERROR, no new WARNING — but the
        # per-run INFO summary still carries the standing count.
        second_line = len(indexer_log_lines())
        await _sweep(mcp_session)
        assert _lines_matching("parked at terminal cap", second_line) == []
        assert _lines_matching("permanent actionable skips", second_line) == []
        summaries = _lines_matching("Degraded ledger:", second_line)
        assert summaries and "1 capped" in summaries[-1], summaries

        # A changed set is news again.
        third_line = len(indexer_log_lines())
        _seed_standing_conditions([CAPPED_DOC_ID, CAPPED_DOC_ID + "2"])
        await _sweep(mcp_session)
        changed = _lines_matching("parked at terminal cap", third_line)
        assert len(changed) == 1, changed
        assert CAPPED_DOC_ID + "2" in changed[0]

        # And so is the set clearing.
        fourth_line = len(indexer_log_lines())
        _clear_standing_conditions()
        await _sweep(mcp_session)
        assert _lines_matching("Terminal cap cleared", fourth_line)
        assert _lines_matching("parked at terminal cap", fourth_line) == []
    finally:
        _restore_ledgers(snapshot)
