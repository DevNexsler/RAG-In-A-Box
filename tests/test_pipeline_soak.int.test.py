"""Bounded local stress tests fail on work loss, replay duplicates, or budget breach."""
from scripts.pipeline_soak import run_soak


def test_small_soak_drains_and_preserves_latest_documents(tmp_path):
    report = run_soak(tmp_path, rounds=2, documents=4, workers=2)
    assert report['passed'], report
    assert report['writes'] == 8
    assert report['chunks'] == 4
    assert report['pending_requests'] == report['pending_callbacks'] == 0
    assert report['callback_retries'] > 0
    assert len(report['samples']) == 2


import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


@pytest.mark.parametrize('interrupt', [signal.SIGINT, signal.SIGTERM])
def test_soak_interruption_reaps_worker_before_removing_index(tmp_path, interrupt):
    root = Path(__file__).resolve().parents[1]
    child = None
    with (tmp_path / 'soak.log').open('w') as log:
        parent = subprocess.Popen([sys.executable, str(root / 'scripts/pipeline_soak.py'),
                                   '--rounds', '10000', '--output', str(tmp_path / 'result.json')],
                                  env={**os.environ, 'TMPDIR': str(tmp_path)},
                                  stdout=log, stderr=log, start_new_session=True)
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                children = Path(f'/proc/{parent.pid}/task/{parent.pid}/children').read_text().split()
                if children:
                    child = int(children[0])
                    break
                time.sleep(.02)
            assert child is not None, 'worker did not start'
            parent.send_signal(interrupt)
            parent.wait(timeout=10)
            assert not Path(f'/proc/{child}').exists(), 'orphaned soak worker'
            assert not list(tmp_path.glob('rag-soak-*')), 'temporary index leaked'
        finally:
            if child is not None and Path(f'/proc/{child}').exists():
                try:
                    os.killpg(child, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if parent.poll() is None:
                parent.kill()
            parent.wait(timeout=10)


def test_soak_deadline_reports_failure_and_cleans_owned_index(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / 'scripts/pipeline_soak.py'),
                             '--rounds', '10000', '--deadline', '.1',
                             '--output', str(tmp_path / 'report.json')],
                            env={**os.environ, 'TMPDIR': str(tmp_path)},
                            capture_output=True, text=True, timeout=15)
    import json
    assert result.returncode == 1
    assert json.loads((tmp_path / 'report.json').read_text()) == {'passed': False, 'error': 'deadline_exceeded'}
    assert not list(tmp_path.glob('rag-soak-*'))


def test_soak_budget_breach_fails(tmp_path):
    report = run_soak(tmp_path, rounds=1, documents=2, workers=1, max_p95_ms=.000001)
    assert not report['passed']
    assert report['write_p95_ms'] > .000001
