"""Mutation runner distinguishes assertion kills from broken test infrastructure."""
import subprocess
from pathlib import Path

from scripts.mutation_gate import Mutation, run_mutations


def test_losing_collected_cases_cannot_count_as_a_mutation_kill(tmp_path):
    (tmp_path / 'sample.py').write_text('CASES = [1, 2]\n')
    (tmp_path / 'test_sample.py').write_text(
        'import pytest\nfrom sample import CASES\n'
        '@pytest.mark.parametrize("value", CASES)\n'
        'def test_case(value):\n    assert value > 0\n'
    )
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    subprocess.run(['git', '-C', str(tmp_path), 'add', '.'], check=True)
    mutation = Mutation('lost-case', 'sample.py', '[1, 2]', '[-1]', ('test_sample.py',))
    report = run_mutations(tmp_path, [mutation], timeout=30)
    assert not report['passed']
    assert report['mutations'][0]['status'] == 'error'
    assert report['mutations'][0]['reason'] == 'test inventory changed'


def test_mutations_report_killed_survived_and_invalid_without_editing_checkout(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'sample.py').write_text('def accept(value):\n    return value > 0\n')
    (repo / 'test_sample.py').write_text('from sample import accept\ndef test_positive():\n    assert accept(1)\n')
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
    source = (repo / 'sample.py').read_text()
    mutations = [
        Mutation('killed', 'sample.py', 'value > 0', 'value < 0', ('test_sample.py',)),
        Mutation('survived', 'sample.py', 'value > 0', 'value >= 0', ('test_sample.py',)),
        Mutation('invalid', 'sample.py', 'value > 0', 'value +', ('test_sample.py',)),
        Mutation('test-error', 'sample.py', 'value > 0', 'missing_name', ('test_sample.py',)),
    ]
    report = run_mutations(repo, mutations, timeout=30)
    assert not report['passed']
    assert [row['status'] for row in report['mutations']] == ['killed', 'survived', 'invalid', 'killed']
    assert (repo / 'sample.py').read_text() == source


def test_process_crash_cannot_reuse_stale_junit(tmp_path):
    from scripts.mutation_gate import run_test_process
    (tmp_path / 'conftest.py').write_text('import os\nos._exit(1)\n')
    (tmp_path / 'test_sample.py').write_text('def test_ok():\n    assert True\n')
    report = tmp_path / 'result.xml'
    report.write_text('<testsuites><testsuite tests="1" failures="1" errors="0" skipped="0"/></testsuites>')
    assert run_test_process(tmp_path, ('test_sample.py',), 10, report)['status'] == 'error'


def _pytest_child(parent_pid):
    """Return the parent's pytest child, not the short-lived git ls-files subprocess."""
    for value in Path(f'/proc/{parent_pid}/task/{parent_pid}/children').read_text().split():
        try:
            if b'pytest' in Path(f'/proc/{value}/cmdline').read_bytes():
                return int(value)
        except (FileNotFoundError, ProcessLookupError):
            pass  # The child was reaped before (ENOENT) or after (ESRCH) its cmdline opened.
    return None


def test_child_scan_skips_a_child_that_exits_between_listing_and_read(monkeypatch):
    import errno
    import os
    import signal
    import sys
    # A parent with two pytest-like children, the first of which is reaped after
    # its pid was listed: reading its cmdline then raises ESRCH, not ENOENT (#4119).
    script = ('import subprocess, sys, time\n'
              'kids = [subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "pytest"])'
              ' for _ in range(2)]\n'
              'print(*[kid.pid for kid in kids], flush=True)\n'
              'time.sleep(30)\n')
    parent = subprocess.Popen([sys.executable, '-c', script], stdout=subprocess.PIPE, text=True,
                              start_new_session=True)
    try:
        exiting, survivor = map(int, parent.stdout.readline().split())
        read_bytes = Path.read_bytes

        def read_bytes_racing_exit(path):
            if path == Path(f'/proc/{exiting}/cmdline'):
                raise ProcessLookupError(errno.ESRCH, os.strerror(errno.ESRCH))
            return read_bytes(path)

        monkeypatch.setattr(Path, 'read_bytes', read_bytes_racing_exit)
        assert _pytest_child(parent.pid) == survivor
        assert _pytest_child(survivor) is None
    finally:
        os.killpg(parent.pid, signal.SIGKILL)
        parent.wait(timeout=10)


def test_mutation_cli_termination_reaps_child_and_removes_copy(tmp_path):
    import os
    from pathlib import Path
    import signal
    import sys
    import time
    import shutil
    root = Path(__file__).resolve().parents[1]
    child = None
    with (tmp_path / 'mutation.log').open('w') as log:
        parent = subprocess.Popen([sys.executable, str(root / 'scripts/mutation_gate.py'),
                                   '--output', str(tmp_path / 'report.json')],
                                  env={**os.environ, 'TMPDIR': str(tmp_path)},
                                  stdout=log, stderr=log, start_new_session=True)
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                child = _pytest_child(parent.pid)
                if child is not None:
                    break
                time.sleep(.02)
            assert child is not None, 'mutation baseline did not start'
            parent.send_signal(signal.SIGTERM)
            parent.wait(timeout=10)
            assert not Path(f'/proc/{child}').exists(), 'orphaned mutation test process'
            assert not list(tmp_path.glob('rag-mutations-*')), 'mutation copy leaked'
        finally:
            if child is not None and Path(f'/proc/{child}').exists():
                try:
                    os.killpg(child, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if parent.poll() is None:
                parent.kill()
            parent.wait(timeout=10)
            for directory in tmp_path.glob('rag-mutations-*'):
                shutil.rmtree(directory)
