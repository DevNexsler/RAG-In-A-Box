"""Production entrypoint Prefect lifecycle contract (#0325).

Root cause: production runs ``python server.py``, which imported and called
``mcp_server.run_server()`` directly — bypassing the ``PrefectServer`` context
that ``run_index.py`` and ``mcp_server.__main__`` use. With no
``PREFECT_API_URL`` set and Prefect's ephemeral mode enabled, every index flow
launched by ``file_index_update`` auto-started a throwaway Prefect
``SubprocessASGIServer`` ("Starting temporary server on ...") that orphaned when
earlyoom killed the indexer before teardown — 49+ orphan servers, ~19 GiB
permanent baseline.

Fixes proven here:
1. ``server.main()`` runs the server strictly inside one ``PrefectServer`` so a
   single stable server exists and index subprocesses inherit its
   ``PREFECT_API_URL`` instead of each spawning their own.
2. With ephemeral auto-start disabled (the deployment default), running a flow
   with no ``PREFECT_API_URL`` fails loudly and spawns ZERO temporary servers —
   so a killed indexer can never orphan one.
"""

import subprocess
import sys
import textwrap


def test_main_runs_server_within_prefect_lifecycle(monkeypatch):
    """server.main() must run the HTTP server inside a PrefectServer context.

    Before the fix, server.py had no main()/PrefectServer at all and called
    run_server() at import time — so the production entrypoint never
    established Prefect lifecycle and PREFECT_API_URL stayed empty.
    """
    import server

    order: list[str] = []

    class FakePrefectServer:
        def __enter__(self):
            order.append("prefect_enter")
            return self

        def __exit__(self, *exc):
            order.append("prefect_exit")
            return False

    monkeypatch.setattr(server, "load_config", lambda *a, **k: {})
    monkeypatch.setattr(server, "setup_tracing", lambda *a, **k: None)
    monkeypatch.setattr(server, "PrefectServer", FakePrefectServer)
    monkeypatch.setattr(server, "run_server", lambda **k: order.append("run_server"))

    server.main()

    assert order == ["prefect_enter", "run_server", "prefect_exit"], (
        "run_server must execute strictly within the PrefectServer context so "
        "the one persistent server is up (and PREFECT_API_URL exported) for the "
        "whole serving lifetime"
    )


# The child mirrors how an index subprocess would run a Prefect flow if it ever
# lacked an inherited PREFECT_API_URL: with ephemeral auto-start disabled it
# must NOT start a SubprocessASGIServer (which would orphan on SIGKILL) — it
# must fail loudly instead. Runs in a subprocess so Prefect's process-global
# settings can't leak into the rest of the suite.
_CHILD = textwrap.dedent(
    """
    import os
    os.environ["PREFECT_API_URL"] = ""
    os.environ["PREFECT_SERVER_ALLOW_EPHEMERAL_MODE"] = "false"
    try:
        from prefect.settings.models.root import Settings as _S
        import prefect.context
        prefect.context.get_settings_context().settings = _S()
    except Exception:
        pass

    import prefect.server.api.server as _srv
    _starts = {"n": 0}
    _orig = _srv.SubprocessASGIServer.start
    def _spy(self, *a, **k):
        _starts["n"] += 1
        return _orig(self, *a, **k)
    _srv.SubprocessASGIServer.start = _spy

    from prefect import flow

    @flow
    def _noop():
        return 42

    raised = None
    try:
        _noop()
    except Exception as exc:
        raised = type(exc).__name__
    print(f"STARTS={_starts['n']} RAISED={raised}")
    """
)


def test_ephemeral_disabled_spawns_no_temporary_server():
    """With ephemeral auto-start disabled and no API URL, a flow run must not
    spawn any Prefect temporary server — proving a killed indexer under the
    production config leaves zero orphan servers behind."""
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD],
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = proc.stdout.strip()
    assert "STARTS=0" in out, (
        "A Prefect SubprocessASGIServer was started despite ephemeral mode being "
        f"disabled — it would orphan on kill. Child output: {out!r} / {proc.stderr[-500:]!r}"
    )
